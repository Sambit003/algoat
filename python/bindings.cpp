/**
 * @file bindings.cpp
 * @brief Nanobind Python C++ extension module for Algoat.
 *
 * Exposes C++ sorting and searching primitives to Python with zero-copy NumPy ndarray
 * buffer views, mixed-type list sorting with GIL release, and Morton Z-order curve
 * spatial sorting for complex numbers.
 */

#include "python_fastpath.hpp"
#include "python_types.hpp"
#include "python_wrappers.hpp"

#include <algoat/algoat.hpp>
#include <algoat/dp/alignment.hpp>
#include <algoat/dp/levenshtein.hpp>
#include <algoat/hash/hash.hpp>
#include <algoat/numerics/bigint.hpp>
#include <algoat/numerics/float16_sort.hpp>
#include <algoat/sorting/boolean_sort.hpp>
#include <algoat/sorting/ips4o.hpp>
#include <algoat/sorting/pdqsort.hpp>
#include <algoat/strings/burstsort.hpp>
#include <algoat/strings/multikey_quicksort.hpp>
#include <cmath>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <span>
#include <stdexcept>
#include <vector>

namespace nb = nanobind;

/**
 * @brief Unboxes Python list elements into C++ wrappers, sorts in C++ releasing the GIL, and
 * returns a new list.
 * @tparam Wrapper Type of wrapper struct (e.g., `PyFloatWrapper`, `PyBigIntWrapper`).
 * @param data Input Python list.
 * @return New sorted Python list.
 */
template <typename Wrapper> nb::list sort_list_direct(nb::list data) {
    size_t n = nb::len(data);
    std::vector<Wrapper> buf(n);
    for (size_t i = 0; i < n; ++i) {
        buf[i] = Wrapper(data[i].ptr());
    }

    {
        nb::gil_scoped_release release;
        algoat::sort<Wrapper>(std::span<Wrapper>{buf});
    }

    nb::list result = nb::steal<nb::list>(PyList_New(n));
    for (size_t i = 0; i < n; ++i) {
        PyObject* obj = buf[i].obj;
        Py_INCREF(obj);
        PyList_SET_ITEM(result.ptr(), i, obj);
    }
    return result;
}

/**
 * @brief Sorts Python list elements in-place by mutating the underlying `PyListObject`.
 * @tparam Wrapper Type of wrapper struct.
 * @param data Python list to sort in-place.
 */
template <typename Wrapper> void sort_list_inplace_impl(nb::list data) {
    size_t n = nb::len(data);
    if (n == 0)
        return;

    std::vector<Wrapper> buf(n);
    for (size_t i = 0; i < n; ++i) {
        buf[i] = Wrapper(data[i].ptr());
    }

    {
        nb::gil_scoped_release release;
        algoat::sort<Wrapper>(std::span<Wrapper>{buf});
    }

    for (size_t i = 0; i < n; ++i) {
        PyList_SET_ITEM(data.ptr(), i, buf[i].obj);
    }
}

/**
 * @brief Specialized O(N) branchless counting sort for Python boolean lists.
 * @param data Input Python list.
 * @param out_result Populated with sorted boolean list if all elements are booleans.
 * @return `true` if list contained exclusively booleans, `false` otherwise.
 */
bool try_sort_bool_list(nb::list data, nb::list& out_result) {
    size_t n = nb::len(data);
    if (n == 0) {
        out_result = nb::list();
        return true;
    }
    if (!PyBool_Check(data[0].ptr())) {
        return false;
    }

    size_t count_false = 0;
    for (size_t i = 0; i < n; ++i) {
        PyObject* ptr = data[i].ptr();
        if (!PyBool_Check(ptr)) {
            return false;
        }
        if (ptr == Py_False) {
            count_false++;
        }
    }

    nb::list result = nb::steal<nb::list>(PyList_New(n));
    for (size_t i = 0; i < count_false; ++i) {
        Py_INCREF(Py_False);
        PyList_SET_ITEM(result.ptr(), i, Py_False);
    }
    for (size_t i = count_false; i < n; ++i) {
        Py_INCREF(Py_True);
        PyList_SET_ITEM(result.ptr(), i, Py_True);
    }

    out_result = result;
    return true;
}

/**
 * @brief Primary entry point for out-of-place Python list sorting.
 *
 * Inspects the type of the first element to dispatch to specialized fast-path wrappers.
 */
nb::list sort_dispatch(nb::list data, const std::string& curve = "morton") {
    if (nb::len(data) == 0)
        return nb::list();

    nb::list bool_res;
    if (try_sort_bool_list(data, bool_res)) {
        return bool_res;
    }

    auto fast_res = algoat::python_fastpath::try_sort_out_of_place_fastpath(data);
    if (fast_res.has_value()) {
        return *fast_res;
    }

    PyObject* first = data[0].ptr();
    if (PyFloat_Check(first)) {
        return sort_list_direct<algoat::pybind::PyFloatWrapper>(data);
    } else if (PyUnicode_Check(first)) {
        return sort_list_direct<algoat::pybind::PyStringWrapper>(data);
    } else if (PyComplex_Check(first)) {
        if (curve == "hilbert") {
            return sort_list_direct<algoat::pybind::PyComplexHilbertWrapper>(data);
        } else if (curve == "hybrid") {
            return sort_list_direct<algoat::pybind::PyComplexHybridWrapper>(data);
        } else {
            return sort_list_direct<algoat::pybind::PyComplexWrapper>(data);
        }
    } else if (PyLong_Check(first)) {
        return sort_list_direct<algoat::pybind::PyBigIntWrapper>(data);
    } else {
        return sort_list_direct<algoat::pybind::PyGenericWrapper>(data);
    }
}

/**
 * @brief Primary entry point for in-place Python list sorting.
 */
void sort_inplace_dispatch(nb::list data, const std::string& curve = "morton") {
    if (nb::len(data) == 0)
        return;

    if (PyBool_Check(data[0].ptr())) {
        size_t n = nb::len(data);
        size_t count_false = 0;
        bool all_bool = true;
        for (size_t i = 0; i < n; ++i) {
            PyObject* ptr = data[i].ptr();
            if (!PyBool_Check(ptr)) {
                all_bool = false;
                break;
            }
            if (ptr == Py_False)
                count_false++;
        }
        if (all_bool) {
            for (size_t i = 0; i < count_false; ++i) {
                Py_INCREF(Py_False);
                PyObject* old = PyList_GET_ITEM(data.ptr(), i);
                PyList_SET_ITEM(data.ptr(), i, Py_False);
                Py_DECREF(old);
            }
            for (size_t i = count_false; i < n; ++i) {
                Py_INCREF(Py_True);
                PyObject* old = PyList_GET_ITEM(data.ptr(), i);
                PyList_SET_ITEM(data.ptr(), i, Py_True);
                Py_DECREF(old);
            }
            return;
        }
    }

    PyObject* first = data[0].ptr();
    if (PyFloat_CheckExact(first)) {
        if (algoat::python_fastpath::try_sort_inplace_float_fastpath(data)) {
            return;
        }
        sort_list_inplace_impl<algoat::pybind::PyGenericWrapper>(data);
    } else if (PyLong_CheckExact(first)) {
        if (algoat::python_fastpath::try_sort_inplace_int_fastpath(data)) {
            return;
        }
        sort_list_inplace_impl<algoat::pybind::PyGenericWrapper>(data);
    } else if (PyFloat_Check(first)) {
        sort_list_inplace_impl<algoat::pybind::PyFloatWrapper>(data);
    } else if (PyUnicode_Check(first)) {
        sort_list_inplace_impl<algoat::pybind::PyStringWrapper>(data);
    } else if (PyComplex_Check(first)) {
        if (curve == "hilbert") {
            sort_list_inplace_impl<algoat::pybind::PyComplexHilbertWrapper>(data);
        } else if (curve == "hybrid") {
            sort_list_inplace_impl<algoat::pybind::PyComplexHybridWrapper>(data);
        } else {
            sort_list_inplace_impl<algoat::pybind::PyComplexWrapper>(data);
        }
    } else if (PyLong_Check(first)) {
        sort_list_inplace_impl<algoat::pybind::PyBigIntWrapper>(data);
    } else {
        sort_list_inplace_impl<algoat::pybind::PyGenericWrapper>(data);
    }
}

/**
 * @brief Zero-allocation direct random access view for Python list elements.
 */
template <typename T> struct PyListRandomAccess {
    PyObject* const* items;
    explicit PyListRandomAccess(PyObject* const* item_array) : items(item_array) {}

    T operator[](std::size_t idx) const noexcept {
        if constexpr (std::is_same_v<T, int64_t>) {
            return PyLong_AsLongLong(items[idx]);
        } else if constexpr (std::is_same_v<T, double>) {
            return PyFloat_AsDouble(items[idx]);
        } else {
            return nb::cast<T>(nb::handle(items[idx]));
        }
    }
};

/**
 * @brief Direct typed search implementation on Python lists with zero allocation.
 *
 * Delegates search directly to core C++ algoat::searching::BinarySearch::search_indexed,
 * evaluating only O(log N) elements without full list copying.
 */
template <typename T>
std::optional<std::size_t> search_list_direct(PyObject* const* items, size_t n, const T& target) {
    PyListRandomAccess<T> seq(items);
    return algoat::searching::BinarySearch{}.search_indexed(seq, n, target);
}

/**
 * @brief Searches for target value in Python list with fast-path dynamic type inference.
 */
std::optional<std::size_t> search_dispatch(nb::list data, nb::object target) {
    PyObject* data_ptr = data.ptr();
    Py_ssize_t n = PyList_GET_SIZE(data_ptr);
    if (n == 0)
        return std::nullopt;

    PyObject* const* items = ((PyListObject*)data_ptr)->ob_item;
    PyObject* target_ptr = target.ptr();

    if (PyLong_CheckExact(target_ptr) && PyLong_CheckExact(items[0])) {
        int64_t target_val = PyLong_AsLongLong(target_ptr);
        return search_list_direct<int64_t>(items, static_cast<size_t>(n), target_val);
    } else if (PyFloat_Check(target_ptr) || PyFloat_Check(items[0])) {
        double target_val = PyFloat_AsDouble(target_ptr);
        return search_list_direct<double>(items, static_cast<size_t>(n), target_val);
    } else if (PyUnicode_Check(target_ptr) || PyUnicode_Check(items[0])) {
        std::string target_val = nb::cast<std::string>(target);
        return search_list_direct<std::string>(items, static_cast<size_t>(n), target_val);
    } else {
        int64_t target_val = nb::cast<int64_t>(target);
        return search_list_direct<int64_t>(items, static_cast<size_t>(n), target_val);
    }
}

template <typename T> nb::list search_many_list_typed(nb::list data, nb::list targets) {
    PyObject* data_ptr = data.ptr();
    size_t n = static_cast<size_t>(PyList_GET_SIZE(data_ptr));
    size_t num_targets = nb::len(targets);
    PyListRandomAccess<T> data_seq(((PyListObject*)data_ptr)->ob_item);
    algoat::searching::BinarySearch algo;

    nb::list res_list;
    for (size_t i = 0; i < num_targets; ++i) {
        T target_val = nb::cast<T>(targets[i]);
        auto res = algo.search_indexed(data_seq, n, target_val);
        if (res.has_value()) {
            res_list.append(nb::cast(res.value()));
        } else {
            res_list.append(nb::none());
        }
    }
    return res_list;
}

/**
 * @brief Searches multiple targets across a Python list with amortized FFI calls.
 */
nb::list search_many_dispatch(nb::list data, nb::list targets) {
    size_t num_targets = nb::len(targets);
    if (num_targets == 0) {
        return nb::list();
    }
    if (nb::len(data) == 0) {
        nb::list results;
        for (size_t i = 0; i < num_targets; ++i) {
            results.append(nb::none());
        }
        return results;
    }

    PyObject* first = data[0].ptr();
    if (PyFloat_Check(first)) {
        return search_many_list_typed<double>(data, targets);
    } else if (PyUnicode_Check(first)) {
        return search_many_list_typed<std::string>(data, targets);
    } else {
        return search_many_list_typed<int64_t>(data, targets);
    }
}

template <typename T>
std::optional<std::size_t> search_ndarray_typed(nb::ndarray<T, nb::ndim<1>, nb::c_contig> array,
                                                T target) {
    return algoat::search<T>(std::span<const T>(array.data(), array.size()), target);
}

template <typename T>
std::vector<std::optional<std::size_t>>
search_many_ndarray_typed(nb::ndarray<T, nb::ndim<1>, nb::c_contig> array,
                          nb::ndarray<T, nb::ndim<1>, nb::c_contig> targets) {
    std::span<const T> data_span(array.data(), array.size());
    size_t num_targets = targets.size();
    const T* targets_ptr = targets.data();

    std::vector<std::optional<std::size_t>> results(num_targets);
    algoat::searching::BinarySearch algo;
    for (size_t i = 0; i < num_targets; ++i) {
        results[i] = algo.search(data_span, targets_ptr[i]);
    }
    return results;
}

/**
 * @brief Zero-copy sorting wrapper for NumPy Float16 arrays viewed as uint16.
 */
void sort_ndarray_float16_buffer(nb::ndarray<uint16_t, nb::ndim<1>, nb::c_contig> array) {
    if (reinterpret_cast<std::uintptr_t>(array.data()) % alignof(uint16_t) != 0) {
        throw std::invalid_argument(
            "Input array memory buffer is not aligned to alignof(uint16_t). Use np.require(arr, "
            "requirements=['A']) in Python.");
    }
    algoat::numerics::sort_float16(std::span<uint16_t>(array.data(), array.size()));
}

/**
 * @brief Zero-copy sorting wrapper for NumPy boolean arrays.
 */
void sort_ndarray_bool_buffer(nb::ndarray<uint8_t, nb::ndim<1>, nb::c_contig> array) {
    algoat::sorting::sort_boolean(std::span<uint8_t>(array.data(), array.size()));
}

template <typename T> void sort_ndarray(nb::ndarray<T, nb::ndim<1>, nb::c_contig> array) {
    algoat::sort(std::span<T>(array.data(), array.size()));
}

namespace {

algoat::hash::HighwayKey parse_highway_key(nb::object key_obj) {
    if (nb::isinstance<nb::sequence>(key_obj)) {
        nb::sequence seq = nb::cast<nb::sequence>(key_obj);
        if (nb::len(seq) != 4) {
            throw nb::value_error("HighwayKey must contain exactly 4 64-bit integers");
        }
        return algoat::hash::HighwayKey{nb::cast<uint64_t>(seq[0]), nb::cast<uint64_t>(seq[1]),
                                        nb::cast<uint64_t>(seq[2]), nb::cast<uint64_t>(seq[3])};
    }
    throw nb::type_error("HighwayKey must be a sequence of 4 integers");
}

struct PyBufferGuard {
    Py_buffer view;
    bool active = false;

    ~PyBufferGuard() noexcept {
        if (active) {
            PyBuffer_Release(&view);
        }
    }
};

template <typename Func> auto invoke_with_buffer(nb::handle obj, Func&& func) {
    PyObject* p = obj.ptr();
    if (PyUnicode_Check(p)) {
        Py_ssize_t len = 0;
        const char* s = PyUnicode_AsUTF8AndSize(p, &len);
        if (!s)
            throw nb::python_error();
        auto span = std::as_bytes(std::span{s, static_cast<size_t>(len)});
        decltype(func(span)) res;
        {
            nb::gil_scoped_release release;
            res = func(span);
        }
        return res;
    }

    PyBufferGuard guard;
    if (PyObject_GetBuffer(p, &guard.view, PyBUF_SIMPLE) != 0) {
        PyErr_Clear();
        throw nb::type_error("Expected a bytes-like object or str");
    }
    guard.active = true;

    auto span = std::as_bytes(
        std::span{static_cast<const char*>(guard.view.buf), static_cast<size_t>(guard.view.len)});
    decltype(func(span)) res;
    {
        nb::gil_scoped_release release;
        res = func(span);
    }
    return res;
}

} // namespace

NB_MODULE(_algoat_impl, m) {
    m.doc() = "Algoat C++ algorithm library bindings";

    m.def("load_global_config", &algoat::load_global_config, nb::arg("filepath"),
          "Load algorithm configuration from a JSON file");

    m.def("sort", &sort_dispatch, nb::arg("data"), nb::arg("curve") = "morton",
          "Sort a list of mixed types");
    m.def("sort_inplace", &sort_inplace_dispatch, nb::arg("data"), nb::arg("curve") = "morton",
          "Sort a list in-place");
    m.def("search", &search_dispatch, nb::arg("data"), nb::arg("target"),
          "Search for a target in a list");
    m.def("search_many", &search_many_dispatch, nb::arg("data"), nb::arg("targets"),
          "Search for multiple targets in a list");

    // Fast-path homogeneous list sorting functions
    m.def("try_sort_inplace_int_fastpath", &algoat::python_fastpath::try_sort_inplace_int_fastpath,
          nb::arg("data"), "Fast-path in-place sort for homogeneous 64-bit int lists");
    m.def("try_sort_inplace_float_fastpath",
          &algoat::python_fastpath::try_sort_inplace_float_fastpath, nb::arg("data"),
          "Fast-path in-place sort for homogeneous float lists");
    m.def("try_sort_out_of_place_fastpath",
          &algoat::python_fastpath::try_sort_out_of_place_fastpath, nb::arg("data"),
          "Fast-path out-of-place sort returning a new Python list");

    // Zero-copy array sorting bindings with fine-grained GIL release
    m.def("sort_numpy_f16", &sort_ndarray_float16_buffer, nb::arg("array").noconvert(),
          nb::call_guard<nb::gil_scoped_release>());
    m.def("sort_numpy_bool", &sort_ndarray_bool_buffer, nb::arg("array").noconvert(),
          nb::call_guard<nb::gil_scoped_release>());
    m.def("sort_numpy", &sort_ndarray<int8_t>, nb::arg("array").noconvert(),
          nb::call_guard<nb::gil_scoped_release>());
    m.def("sort_numpy", &sort_ndarray<uint8_t>, nb::arg("array").noconvert(),
          nb::call_guard<nb::gil_scoped_release>());
    m.def("sort_numpy", &sort_ndarray<int16_t>, nb::arg("array").noconvert(),
          nb::call_guard<nb::gil_scoped_release>());
    m.def("sort_numpy", &sort_ndarray<uint16_t>, nb::arg("array").noconvert(),
          nb::call_guard<nb::gil_scoped_release>());
    m.def("sort_numpy", &sort_ndarray<int32_t>, nb::arg("array").noconvert(),
          nb::call_guard<nb::gil_scoped_release>());
    m.def("sort_numpy", &sort_ndarray<int64_t>, nb::arg("array").noconvert(),
          nb::call_guard<nb::gil_scoped_release>());
    m.def("sort_numpy", &sort_ndarray<uint32_t>, nb::arg("array").noconvert(),
          nb::call_guard<nb::gil_scoped_release>());
    m.def("sort_numpy", &sort_ndarray<uint64_t>, nb::arg("array").noconvert(),
          nb::call_guard<nb::gil_scoped_release>());
    m.def("sort_numpy", &sort_ndarray<float>, nb::arg("array").noconvert(),
          nb::call_guard<nb::gil_scoped_release>());
    m.def("sort_numpy", &sort_ndarray<double>, nb::arg("array").noconvert(),
          nb::call_guard<nb::gil_scoped_release>());

    m.def(
        "sort_numpy_c64_morton",
        [](nb::ndarray<std::complex<float>, nb::ndim<1>, nb::c_contig> array) {
            algoat::numerics::sort_complex_morton(
                std::span<std::complex<float>>(array.data(), array.size()));
        },
        nb::arg("array").noconvert(), nb::call_guard<nb::gil_scoped_release>());
    m.def(
        "sort_numpy_c128_morton",
        [](nb::ndarray<std::complex<double>, nb::ndim<1>, nb::c_contig> array) {
            algoat::numerics::sort_complex_morton(
                std::span<std::complex<double>>(array.data(), array.size()));
        },
        nb::arg("array").noconvert(), nb::call_guard<nb::gil_scoped_release>());

    m.def(
        "sort_numpy_c64_hilbert",
        [](nb::ndarray<std::complex<float>, nb::ndim<1>, nb::c_contig> array) {
            algoat::numerics::sort_complex_hilbert(
                std::span<std::complex<float>>(array.data(), array.size()));
        },
        nb::arg("array").noconvert(), nb::call_guard<nb::gil_scoped_release>());
    m.def(
        "sort_numpy_c128_hilbert",
        [](nb::ndarray<std::complex<double>, nb::ndim<1>, nb::c_contig> array) {
            algoat::numerics::sort_complex_hilbert(
                std::span<std::complex<double>>(array.data(), array.size()));
        },
        nb::arg("array").noconvert(), nb::call_guard<nb::gil_scoped_release>());

    m.def(
        "sort_numpy_c64_hybrid",
        [](nb::ndarray<std::complex<float>, nb::ndim<1>, nb::c_contig> array) {
            algoat::numerics::sort_complex_hybrid(
                std::span<std::complex<float>>(array.data(), array.size()));
        },
        nb::arg("array").noconvert(), nb::call_guard<nb::gil_scoped_release>());
    m.def(
        "sort_numpy_c128_hybrid",
        [](nb::ndarray<std::complex<double>, nb::ndim<1>, nb::c_contig> array) {
            algoat::numerics::sort_complex_hybrid(
                std::span<std::complex<double>>(array.data(), array.size()));
        },
        nb::arg("array").noconvert(), nb::call_guard<nb::gil_scoped_release>());

    // Zero-copy array search bindings with fine-grained GIL release
    m.def("search_numpy", &search_ndarray_typed<int8_t>, nb::arg("array").noconvert(),
          nb::arg("target"), nb::call_guard<nb::gil_scoped_release>());
    m.def("search_numpy", &search_ndarray_typed<uint8_t>, nb::arg("array").noconvert(),
          nb::arg("target"), nb::call_guard<nb::gil_scoped_release>());
    m.def("search_numpy", &search_ndarray_typed<int16_t>, nb::arg("array").noconvert(),
          nb::arg("target"), nb::call_guard<nb::gil_scoped_release>());
    m.def("search_numpy", &search_ndarray_typed<uint16_t>, nb::arg("array").noconvert(),
          nb::arg("target"), nb::call_guard<nb::gil_scoped_release>());
    m.def("search_numpy", &search_ndarray_typed<int32_t>, nb::arg("array").noconvert(),
          nb::arg("target"), nb::call_guard<nb::gil_scoped_release>());
    m.def("search_numpy", &search_ndarray_typed<int64_t>, nb::arg("array").noconvert(),
          nb::arg("target"), nb::call_guard<nb::gil_scoped_release>());
    m.def("search_numpy", &search_ndarray_typed<uint32_t>, nb::arg("array").noconvert(),
          nb::arg("target"), nb::call_guard<nb::gil_scoped_release>());
    m.def("search_numpy", &search_ndarray_typed<uint64_t>, nb::arg("array").noconvert(),
          nb::arg("target"), nb::call_guard<nb::gil_scoped_release>());
    m.def("search_numpy", &search_ndarray_typed<float>, nb::arg("array").noconvert(),
          nb::arg("target"), nb::call_guard<nb::gil_scoped_release>());
    m.def("search_numpy", &search_ndarray_typed<double>, nb::arg("array").noconvert(),
          nb::arg("target"), nb::call_guard<nb::gil_scoped_release>());

    m.def("search_many_numpy", &search_many_ndarray_typed<int8_t>, nb::arg("array").noconvert(),
          nb::arg("targets").noconvert(), nb::call_guard<nb::gil_scoped_release>());
    m.def("search_many_numpy", &search_many_ndarray_typed<uint8_t>, nb::arg("array").noconvert(),
          nb::arg("targets").noconvert(), nb::call_guard<nb::gil_scoped_release>());
    m.def("search_many_numpy", &search_many_ndarray_typed<int16_t>, nb::arg("array").noconvert(),
          nb::arg("targets").noconvert(), nb::call_guard<nb::gil_scoped_release>());
    m.def("search_many_numpy", &search_many_ndarray_typed<uint16_t>, nb::arg("array").noconvert(),
          nb::arg("targets").noconvert(), nb::call_guard<nb::gil_scoped_release>());
    m.def("search_many_numpy", &search_many_ndarray_typed<int32_t>, nb::arg("array").noconvert(),
          nb::arg("targets").noconvert(), nb::call_guard<nb::gil_scoped_release>());
    m.def("search_many_numpy", &search_many_ndarray_typed<int64_t>, nb::arg("array").noconvert(),
          nb::arg("targets").noconvert(), nb::call_guard<nb::gil_scoped_release>());
    m.def("search_many_numpy", &search_many_ndarray_typed<uint32_t>, nb::arg("array").noconvert(),
          nb::arg("targets").noconvert(), nb::call_guard<nb::gil_scoped_release>());
    m.def("search_many_numpy", &search_many_ndarray_typed<uint64_t>, nb::arg("array").noconvert(),
          nb::arg("targets").noconvert(), nb::call_guard<nb::gil_scoped_release>());
    m.def("search_many_numpy", &search_many_ndarray_typed<float>, nb::arg("array").noconvert(),
          nb::arg("targets").noconvert(), nb::call_guard<nb::gil_scoped_release>());
    m.def("search_many_numpy", &search_many_ndarray_typed<double>, nb::arg("array").noconvert(),
          nb::arg("targets").noconvert(), nb::call_guard<nb::gil_scoped_release>());

    nb::class_<algoat::Rational>(m, "Rational")
        .def(nb::init<int64_t, int64_t>())
        .def_rw("num", &algoat::Rational::num)
        .def_rw("den", &algoat::Rational::den)
        .def("__lt__", [](const algoat::Rational& a, const algoat::Rational& b) { return a < b; })
        .def("__eq__", [](const algoat::Rational& a, const algoat::Rational& b) { return a == b; });

    nb::class_<algoat::numerics::BigInt>(m, "BigInt")
        .def(nb::init<>())
        .def(nb::init<int64_t>())
        .def("__init__", [](algoat::numerics::BigInt* self,
                            const std::string& str) { new (self) algoat::numerics::BigInt(str); })
        .def("__init__", [](algoat::numerics::BigInt* self, const std::string& str,
                            int base) { new (self) algoat::numerics::BigInt(str, base); })
        .def("__add__", [](const algoat::numerics::BigInt& a,
                           const algoat::numerics::BigInt& b) { return a + b; })
        .def("__sub__", [](const algoat::numerics::BigInt& a,
                           const algoat::numerics::BigInt& b) { return a - b; })
        .def(
            "__mul__",
            [](const algoat::numerics::BigInt& a, const algoat::numerics::BigInt& b) {
                return a * b;
            },
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "__floordiv__",
            [](const algoat::numerics::BigInt& a, const algoat::numerics::BigInt& b) {
                return a / b;
            },
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "__mod__",
            [](const algoat::numerics::BigInt& a, const algoat::numerics::BigInt& b) {
                return a % b;
            },
            nb::call_guard<nb::gil_scoped_release>())
        .def("__neg__", [](const algoat::numerics::BigInt& a) { return -a; })
        .def("__invert__", [](const algoat::numerics::BigInt& a) { return ~a; })
        .def("__and__", [](const algoat::numerics::BigInt& a,
                           const algoat::numerics::BigInt& b) { return a & b; })
        .def("__or__", [](const algoat::numerics::BigInt& a,
                          const algoat::numerics::BigInt& b) { return a | b; })
        .def("__xor__", [](const algoat::numerics::BigInt& a,
                           const algoat::numerics::BigInt& b) { return a ^ b; })
        .def("__lshift__", [](const algoat::numerics::BigInt& a, size_t bits) { return a << bits; })
        .def("__rshift__", [](const algoat::numerics::BigInt& a, size_t bits) { return a >> bits; })
        .def("__eq__", [](const algoat::numerics::BigInt& a,
                          const algoat::numerics::BigInt& b) { return a == b; })
        .def("__lt__", [](const algoat::numerics::BigInt& a,
                          const algoat::numerics::BigInt& b) { return a < b; })
        .def("__le__", [](const algoat::numerics::BigInt& a,
                          const algoat::numerics::BigInt& b) { return a <= b; })
        .def("__gt__", [](const algoat::numerics::BigInt& a,
                          const algoat::numerics::BigInt& b) { return a > b; })
        .def("__ge__", [](const algoat::numerics::BigInt& a,
                          const algoat::numerics::BigInt& b) { return a >= b; })
        .def("to_string", &algoat::numerics::BigInt::to_string, nb::arg("base") = 10,
             nb::call_guard<nb::gil_scoped_release>())
        .def("to_hex", &algoat::numerics::BigInt::to_hex)
        .def("to_int64", &algoat::numerics::BigInt::to_int64)
        .def("fits_int64", &algoat::numerics::BigInt::fits_int64)
        .def("num_bits", &algoat::numerics::BigInt::num_bits)
        .def("num_limbs", &algoat::numerics::BigInt::num_limbs)
        .def("is_zero", &algoat::numerics::BigInt::is_zero)
        .def("is_negative", &algoat::numerics::BigInt::is_negative)
        .def("sign", &algoat::numerics::BigInt::sign)
        .def("mod_pow", &algoat::numerics::BigInt::mod_pow, nb::arg("exp"), nb::arg("modulus"),
             nb::call_guard<nb::gil_scoped_release>())
        .def("mod_inverse", &algoat::numerics::BigInt::mod_inverse, nb::arg("modulus"),
             nb::call_guard<nb::gil_scoped_release>())
        .def("__str__", [](const algoat::numerics::BigInt& a) { return a.to_string(10); })
        .def("__repr__", [](const algoat::numerics::BigInt& a) {
            return "BigInt(\"" + a.to_string(10) + "\")";
        });

    auto strings_m =
        m.def_submodule("strings", "String sorting suite (Burstsort & Multi-Key Quicksort)");
    strings_m.def(
        "burstsort",
        [](nb::list data) -> nb::list {
            size_t n = nb::len(data);
            std::vector<algoat::pybind::PyStringWrapper> buf(n);
            for (size_t i = 0; i < n; ++i) {
                buf[i] = algoat::pybind::PyStringWrapper(data[i].ptr());
            }
            {
                nb::gil_scoped_release release;
                algoat::strings::burstsort(std::span<algoat::pybind::PyStringWrapper>{buf});
            }
            nb::list result = nb::steal<nb::list>(PyList_New(n));
            for (size_t i = 0; i < n; ++i) {
                PyObject* obj = buf[i].obj;
                Py_INCREF(obj);
                PyList_SET_ITEM(result.ptr(), i, obj);
            }
            return result;
        },
        nb::arg("data"));

    strings_m.def(
        "burstsort_inplace",
        [](nb::list data) {
            size_t n = nb::len(data);
            std::vector<algoat::pybind::PyStringWrapper> buf(n);
            for (size_t i = 0; i < n; ++i) {
                buf[i] = algoat::pybind::PyStringWrapper(data[i].ptr());
            }
            {
                nb::gil_scoped_release release;
                algoat::strings::burstsort(std::span<algoat::pybind::PyStringWrapper>{buf});
            }
            for (size_t i = 0; i < n; ++i) {
                PyList_SET_ITEM(data.ptr(), i, buf[i].obj);
            }
        },
        nb::arg("data"));

    strings_m.def(
        "multikey_quicksort",
        [](nb::list data) -> nb::list {
            size_t n = nb::len(data);
            std::vector<algoat::pybind::PyStringWrapper> buf(n);
            for (size_t i = 0; i < n; ++i) {
                buf[i] = algoat::pybind::PyStringWrapper(data[i].ptr());
            }
            {
                nb::gil_scoped_release release;
                algoat::strings::multikey_quicksort(
                    std::span<algoat::pybind::PyStringWrapper>{buf});
            }
            nb::list result = nb::steal<nb::list>(PyList_New(n));
            for (size_t i = 0; i < n; ++i) {
                PyObject* obj = buf[i].obj;
                Py_INCREF(obj);
                PyList_SET_ITEM(result.ptr(), i, obj);
            }
            return result;
        },
        nb::arg("data"));

    strings_m.def(
        "multikey_quicksort_inplace",
        [](nb::list data) {
            size_t n = nb::len(data);
            std::vector<algoat::pybind::PyStringWrapper> buf(n);
            for (size_t i = 0; i < n; ++i) {
                buf[i] = algoat::pybind::PyStringWrapper(data[i].ptr());
            }
            {
                nb::gil_scoped_release release;
                algoat::strings::multikey_quicksort(
                    std::span<algoat::pybind::PyStringWrapper>{buf});
            }
            for (size_t i = 0; i < n; ++i) {
                PyList_SET_ITEM(data.ptr(), i, buf[i].obj);
            }
        },
        nb::arg("data"));

    auto hash_m = m.def_submodule(
        "hash", "Non-cryptographic fast hashing suite (xxHash3, WyHash, HighwayHash)");

    hash_m.def(
        "xxhash3_64",
        [](nb::handle data, uint64_t seed) -> uint64_t {
            return invoke_with_buffer(data, [seed](std::span<const std::byte> sp) {
                return algoat::hash::xxhash3_64(sp, seed);
            });
        },
        nb::arg("data"), nb::arg("seed") = 0, "Compute 64-bit xxHash3 hash");

    hash_m.def(
        "xxhash3_128",
        [](nb::handle data, uint64_t seed) -> nb::tuple {
            auto res = invoke_with_buffer(data, [seed](std::span<const std::byte> sp) {
                return algoat::hash::xxhash3_128(sp, seed);
            });
            return nb::make_tuple(res[0], res[1]);
        },
        nb::arg("data"), nb::arg("seed") = 0, "Compute 128-bit xxHash3 hash as (low64, high64)");

    hash_m.def(
        "wyhash",
        [](nb::handle data, uint64_t seed) -> uint64_t {
            return invoke_with_buffer(data, [seed](std::span<const std::byte> sp) {
                return algoat::hash::wyhash(sp, seed);
            });
        },
        nb::arg("data"), nb::arg("seed") = 0, "Compute 64-bit WyHash hash");

    hash_m.def(
        "highway_hash_64",
        [](nb::handle data, nb::object key) -> uint64_t {
            auto hw_key = parse_highway_key(key);
            return invoke_with_buffer(data, [&hw_key](std::span<const std::byte> sp) {
                return algoat::hash::highway_hash_64(sp, hw_key);
            });
        },
        nb::arg("data"), nb::arg("key"), "Compute 64-bit HighwayHash hash with 256-bit key");

    hash_m.def(
        "highway_hash_128",
        [](nb::handle data, nb::object key) -> nb::tuple {
            auto hw_key = parse_highway_key(key);
            auto res = invoke_with_buffer(data, [&hw_key](std::span<const std::byte> sp) {
                return algoat::hash::highway_hash_128(sp, hw_key);
            });
            return nb::make_tuple(res[0], res[1]);
        },
        nb::arg("data"), nb::arg("key"), "Compute 128-bit HighwayHash hash as (h0, h1)");

    hash_m.def(
        "highway_hash_256",
        [](nb::handle data, nb::object key) -> nb::tuple {
            auto hw_key = parse_highway_key(key);
            auto res = invoke_with_buffer(data, [&hw_key](std::span<const std::byte> sp) {
                return algoat::hash::highway_hash_256(sp, hw_key);
            });
            return nb::make_tuple(res[0], res[1], res[2], res[3]);
        },
        nb::arg("data"), nb::arg("key"), "Compute 256-bit HighwayHash hash as (h0, h1, h2, h3)");

    auto dp_m = m.def_submodule(
        "dp",
        "Dynamic programming & edit distance suite (Myers bit-parallel, Ukkonen banded DP, LCS)");

    nb::enum_<algoat::dp::EditOp>(dp_m, "EditOp")
        .value("Match", algoat::dp::EditOp::Match)
        .value("Substitute", algoat::dp::EditOp::Substitute)
        .value("Insert", algoat::dp::EditOp::Insert)
        .value("Delete", algoat::dp::EditOp::Delete);

    nb::class_<algoat::dp::Alignment>(dp_m, "Alignment")
        .def_ro("distance", &algoat::dp::Alignment::distance)
        .def_ro("operations", &algoat::dp::Alignment::operations);

    dp_m.def(
        "levenshtein_distance",
        [](const std::string& pattern, const std::string& text) -> size_t {
            nb::gil_scoped_release release;
            return algoat::dp::levenshtein_distance(pattern, text);
        },
        nb::arg("pattern"), nb::arg("text"),
        "Compute exact Levenshtein edit distance using Myers' bit-parallel algorithm");

    dp_m.def(
        "levenshtein_within_k",
        [](const std::string& pattern, const std::string& text, size_t k) -> size_t {
            nb::gil_scoped_release release;
            return algoat::dp::levenshtein_within_k(pattern, text, k);
        },
        nb::arg("pattern"), nb::arg("text"), nb::arg("k"),
        "Compute edit distance within bound k using Ukkonen's banded DP");

    dp_m.def(
        "levenshtein_align",
        [](const std::string& pattern, const std::string& text) -> algoat::dp::Alignment {
            nb::gil_scoped_release release;
            return algoat::dp::levenshtein_align(pattern, text);
        },
        nb::arg("pattern"), nb::arg("text"),
        "Compute full alignment recovering step-by-step EditOp operations");

    dp_m.def(
        "levenshtein_batch",
        [](const std::string& pattern,
           const std::vector<std::string>& texts) -> std::vector<size_t> {
            std::vector<std::string_view> views;
            views.reserve(texts.size());
            for (const auto& s : texts)
                views.emplace_back(s);

            nb::gil_scoped_release release;
            return algoat::dp::levenshtein_batch(pattern, views);
        },
        nb::arg("pattern"), nb::arg("texts"),
        "Compute batch edit distance from pattern to multiple texts using SIMD");

    dp_m.def(
        "lcs_length",
        [](const std::string& a, const std::string& b) -> size_t {
            nb::gil_scoped_release release;
            return algoat::dp::lcs_length(a, b);
        },
        nb::arg("a"), nb::arg("b"),
        "Compute Longest Common Subsequence length using bit-parallel algorithm");

    nb::class_<algoat::dp::ScoringMatrix>(dp_m, "ScoringMatrix")
        .def(nb::init<int32_t, int32_t>(), nb::arg("match") = 2, nb::arg("mismatch") = -1)
        .def_static("blosum62", &algoat::dp::ScoringMatrix::blosum62)
        .def_static("pam250", &algoat::dp::ScoringMatrix::pam250)
        .def_static("dna", &algoat::dp::ScoringMatrix::dna, nb::arg("match") = 2,
                    nb::arg("mismatch") = -3)
        .def(
            "score",
            [](const algoat::dp::ScoringMatrix& m, char a, char b) -> int32_t {
                return m.score(a, b);
            },
            nb::arg("a"), nb::arg("b"));

    nb::class_<algoat::dp::GapPenalty>(dp_m, "GapPenalty")
        .def(nb::init<int32_t, int32_t>(), nb::arg("open") = -5, nb::arg("extend") = -1)
        .def_rw("open", &algoat::dp::GapPenalty::open)
        .def_rw("extend", &algoat::dp::GapPenalty::extend);

    nb::class_<algoat::dp::AlignmentResult>(dp_m, "AlignmentResult")
        .def_ro("score", &algoat::dp::AlignmentResult::score)
        .def_ro("cigar", &algoat::dp::AlignmentResult::cigar)
        .def_ro("query_start", &algoat::dp::AlignmentResult::query_start)
        .def_ro("query_end", &algoat::dp::AlignmentResult::query_end)
        .def_ro("ref_start", &algoat::dp::AlignmentResult::ref_start)
        .def_ro("ref_end", &algoat::dp::AlignmentResult::ref_end)
        .def_ro("aligned_query", &algoat::dp::AlignmentResult::aligned_query)
        .def_ro("aligned_ref", &algoat::dp::AlignmentResult::aligned_ref);

    dp_m.def(
        "needleman_wunsch",
        [](const std::string& query, const std::string& ref,
           const algoat::dp::ScoringMatrix& matrix,
           const algoat::dp::GapPenalty& gap) -> algoat::dp::AlignmentResult {
            nb::gil_scoped_release release;
            return algoat::dp::needleman_wunsch(query, ref, matrix, gap);
        },
        nb::arg("query"), nb::arg("ref"), nb::arg("matrix") = algoat::dp::ScoringMatrix{},
        nb::arg("gap") = algoat::dp::GapPenalty{},
        "Global alignment using Hirschberg linear-space algorithm with Gotoh affine gaps");

    dp_m.def(
        "smith_waterman",
        [](const std::string& query, const std::string& ref,
           const algoat::dp::ScoringMatrix& matrix,
           const algoat::dp::GapPenalty& gap) -> algoat::dp::AlignmentResult {
            nb::gil_scoped_release release;
            return algoat::dp::smith_waterman(query, ref, matrix, gap);
        },
        nb::arg("query"), nb::arg("ref"), nb::arg("matrix") = algoat::dp::ScoringMatrix{},
        nb::arg("gap") = algoat::dp::GapPenalty{},
        "Local alignment using Gotoh dynamic programming with full traceback");

    dp_m.def(
        "smith_waterman_score",
        [](const std::string& query, const std::string& ref,
           const algoat::dp::ScoringMatrix& matrix, const algoat::dp::GapPenalty& gap) -> int64_t {
            nb::gil_scoped_release release;
            return algoat::dp::smith_waterman_score(query, ref, matrix, gap);
        },
        nb::arg("query"), nb::arg("ref"), nb::arg("matrix") = algoat::dp::ScoringMatrix{},
        nb::arg("gap") = algoat::dp::GapPenalty{}, "Farrar Striped SIMD local alignment score");

    dp_m.def(
        "smith_waterman_batch",
        [](const std::string& query, const std::vector<std::string>& refs,
           const algoat::dp::ScoringMatrix& matrix,
           const algoat::dp::GapPenalty& gap) -> std::vector<int64_t> {
            std::vector<std::string_view> ref_views;
            ref_views.reserve(refs.size());
            for (const auto& r : refs) {
                ref_views.emplace_back(r);
            }
            nb::gil_scoped_release release;
            return algoat::dp::smith_waterman_batch(query, ref_views, matrix, gap);
        },
        nb::arg("query"), nb::arg("refs"), nb::arg("matrix") = algoat::dp::ScoringMatrix{},
        nb::arg("gap") = algoat::dp::GapPenalty{},
        "Batch Farrar Striped SIMD local alignment scoring against multiple reference targets");
}
