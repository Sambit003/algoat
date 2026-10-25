/**
 * @file python_fastpath.hpp
 * @brief High-performance homogeneous Python list sorting fast-paths.
 *
 * Implements cache-friendly contiguous 16-byte key-pointer pairs, fine-grained
 * GIL release with multi-core parallel sorting (ips4o / pdqsort), and zero-allocation
 * in-place pointer rewrites preserving exact Python object identities (id(x)).
 */

#pragma once

#include <Python.h>
#include <algoat/sorting/ips4o.hpp>
#include <algoat/sorting/pdqsort.hpp>
#include <cmath>
#include <cstdint>
#include <nanobind/nanobind.h>
#include <optional>
#include <span>
#include <vector>

namespace nb = nanobind;

namespace algoat::python_fastpath {

/**
 * @struct FastPathEntry
 * @brief Cache-friendly 16-byte key-pointer pair.
 *
 * Fits exactly 4 entries per 64-byte CPU cache line, enabling hardware
 * stream prefetching and single 128-bit register moves.
 */
template <typename T> struct alignas(16) FastPathEntry {
    T val;         ///< Unboxed primitive key for hardware comparison.
    PyObject* obj; ///< Borrowed pointer to original Python object.

    bool operator<(const FastPathEntry& other) const noexcept {
        return val < other.val;
    }
    bool operator>(const FastPathEntry& other) const noexcept {
        return val > other.val;
    }
    bool operator<=(const FastPathEntry& other) const noexcept {
        return val <= other.val;
    }
    bool operator>=(const FastPathEntry& other) const noexcept {
        return val >= other.val;
    }
    bool operator==(const FastPathEntry& other) const noexcept {
        return val == other.val;
    }
    bool operator!=(const FastPathEntry& other) const noexcept {
        return val != other.val;
    }
};

using IntEntry = FastPathEntry<int64_t>;
using FloatEntry = FastPathEntry<double>;

static_assert(sizeof(IntEntry) == 16, "IntEntry must be exactly 16 bytes");
static_assert(sizeof(FloatEntry) == 16, "FloatEntry must be exactly 16 bytes");

/**
 * @brief Multi-threaded sorting engine selecting between parallel ips4o and pdqsort.
 */
template <typename Entry> inline void sort_entries_parallel(std::vector<Entry>& buf) {
    if (buf.size() >= 10000) {
        algoat::sorting::ips4o_parallel_sort(buf);
    } else {
        algoat::sorting::pdqsort(buf);
    }
}

/**
 * @brief Fast-path in-place sort for homogeneous 64-bit integer lists.
 *
 * Validates that all elements are exact PyLong objects fitting within int64_t.
 * Releases the GIL and performs parallel multi-core sort via ips4o / pdqsort.
 * Reacquires the GIL and rewrites ob_item pointers in-place with zero heap allocations,
 * zero refcount modifications, and 100% object identity (id(x)) preservation.
 *
 * @param data Target Python list.
 * @return true if fast-path successfully sorted the list, false if fallback is required.
 */
inline bool try_sort_inplace_int_fastpath(nb::list data) {
    if (!PyList_Check(data.ptr())) {
        return false;
    }
    size_t n = static_cast<size_t>(PyList_GET_SIZE(data.ptr()));
    if (n <= 1) {
        return true;
    }

    PyObject* const* items = ((PyListObject*)data.ptr())->ob_item;
    std::vector<IntEntry> buf(n);
    for (size_t i = 0; i < n; ++i) {
        PyObject* item = items[i];
        if (!PyLong_CheckExact(item)) {
            return false;
        }
        int overflow = 0;
        long long val = PyLong_AsLongLongAndOverflow(item, &overflow);
        if (overflow != 0) {
            return false;
        }
        buf[i] = IntEntry{static_cast<int64_t>(val), item};
    }

    PyListObject* list_obj = reinterpret_cast<PyListObject*>(data.ptr());
    Py_ssize_t original_allocated = list_obj->allocated;
    list_obj->allocated = -1;

    {
        nb::gil_scoped_release release;
        sort_entries_parallel(buf);
    }

    list_obj->allocated = original_allocated;
    PyObject** target_items = list_obj->ob_item;
    for (size_t i = 0; i < n; ++i) {
        target_items[i] = buf[i].obj;
    }
    return true;
}

/**
 * @brief Fast-path in-place sort for homogeneous double-precision float lists.
 *
 * Validates that all elements are exact PyFloat objects with non-NaN values.
 * Releases the GIL and sorts via ips4o / pdqsort across multiple threads.
 *
 * @param data Target Python list.
 * @return true if successfully sorted, false if fallback is required.
 */
inline bool try_sort_inplace_float_fastpath(nb::list data) {
    if (!PyList_Check(data.ptr())) {
        return false;
    }
    size_t n = static_cast<size_t>(PyList_GET_SIZE(data.ptr()));
    if (n <= 1) {
        return true;
    }

    PyObject* const* items = ((PyListObject*)data.ptr())->ob_item;
    std::vector<FloatEntry> buf(n);
    for (size_t i = 0; i < n; ++i) {
        PyObject* item = items[i];
        if (!PyFloat_CheckExact(item)) {
            return false;
        }
        double val = PyFloat_AS_DOUBLE(item);
        if (std::isnan(val)) {
            return false;
        }
        buf[i] = FloatEntry{val, item};
    }

    PyListObject* list_obj = reinterpret_cast<PyListObject*>(data.ptr());
    Py_ssize_t original_allocated = list_obj->allocated;
    list_obj->allocated = -1;

    {
        nb::gil_scoped_release release;
        sort_entries_parallel(buf);
    }

    list_obj->allocated = original_allocated;
    PyObject** target_items = list_obj->ob_item;
    for (size_t i = 0; i < n; ++i) {
        target_items[i] = buf[i].obj;
    }
    return true;
}

/**
 * @brief Fast-path out-of-place sort returning a new Python list.
 *
 * Inspects list homogeneity, creates a new list, and populates sorted elements
 * while preserving pointer identity and incrementing reference counts.
 *
 * @param data Target Python list.
 * @return New sorted Python list, or std::nullopt if heterogeneous / fallback needed.
 */
inline std::optional<nb::list> try_sort_out_of_place_fastpath(nb::list data) {
    if (!PyList_Check(data.ptr())) {
        return std::nullopt;
    }
    size_t n = static_cast<size_t>(PyList_GET_SIZE(data.ptr()));
    if (n == 0) {
        return nb::list();
    }

    PyObject* const* items = ((PyListObject*)data.ptr())->ob_item;
    PyObject* first = items[0];

    if (PyLong_CheckExact(first)) {
        std::vector<IntEntry> buf(n);
        for (size_t i = 0; i < n; ++i) {
            PyObject* item = items[i];
            if (!PyLong_CheckExact(item)) {
                return std::nullopt;
            }
            int overflow = 0;
            long long val = PyLong_AsLongLongAndOverflow(item, &overflow);
            if (overflow != 0) {
                return std::nullopt;
            }
            buf[i] = IntEntry{static_cast<int64_t>(val), item};
        }

        {
            nb::gil_scoped_release release;
            sort_entries_parallel(buf);
        }

        nb::list result = nb::steal<nb::list>(PyList_New(n));
        for (size_t i = 0; i < n; ++i) {
            PyObject* obj = buf[i].obj;
            Py_INCREF(obj);
            PyList_SET_ITEM(result.ptr(), i, obj);
        }
        return result;
    } else if (PyFloat_CheckExact(first)) {
        std::vector<FloatEntry> buf(n);
        for (size_t i = 0; i < n; ++i) {
            PyObject* item = items[i];
            if (!PyFloat_CheckExact(item)) {
                return std::nullopt;
            }
            double val = PyFloat_AS_DOUBLE(item);
            if (std::isnan(val)) {
                return std::nullopt;
            }
            buf[i] = FloatEntry{val, item};
        }

        {
            nb::gil_scoped_release release;
            sort_entries_parallel(buf);
        }

        nb::list result = nb::steal<nb::list>(PyList_New(n));
        for (size_t i = 0; i < n; ++i) {
            PyObject* obj = buf[i].obj;
            Py_INCREF(obj);
            PyList_SET_ITEM(result.ptr(), i, obj);
        }
        return result;
    }

    return std::nullopt;
}

} // namespace algoat::python_fastpath
