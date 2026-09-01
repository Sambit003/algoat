/**
 * @file bindings.cpp
 * @brief Nanobind Python C++ extension module for Algoat.
 *
 * Exposes C++ sorting and searching primitives to Python with zero-copy NumPy ndarray
 * buffer views, mixed-type list sorting with GIL release, and Morton Z-order curve
 * spatial sorting for complex numbers.
 */

#include "python_types.hpp"
#include "python_wrappers.hpp"

#include <algoat/algoat.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <span>
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
nb::list sort_dispatch(nb::list data) {
    if (nb::len(data) == 0)
        return nb::list();

    nb::list bool_res;
    if (try_sort_bool_list(data, bool_res)) {
        return bool_res;
    }

    PyObject* first = data[0].ptr();
    if (PyFloat_Check(first)) {
        return sort_list_direct<algoat::pybind::PyFloatWrapper>(data);
    } else if (PyUnicode_Check(first)) {
        return sort_list_direct<algoat::pybind::PyStringWrapper>(data);
    } else if (PyComplex_Check(first)) {
        return sort_list_direct<algoat::pybind::PyComplexWrapper>(data);
    } else if (PyLong_Check(first)) {
        return sort_list_direct<algoat::pybind::PyBigIntWrapper>(data);
    } else {
        return sort_list_direct<algoat::pybind::PyGenericWrapper>(data);
    }
}

/**
 * @brief Primary entry point for in-place Python list sorting.
 */
void sort_inplace_dispatch(nb::list data) {
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
    if (PyFloat_Check(first)) {
        sort_list_inplace_impl<algoat::pybind::PyFloatWrapper>(data);
    } else if (PyUnicode_Check(first)) {
        sort_list_inplace_impl<algoat::pybind::PyStringWrapper>(data);
    } else if (PyComplex_Check(first)) {
        sort_list_inplace_impl<algoat::pybind::PyComplexWrapper>(data);
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

#include <algoat/numerics/float16_sort.hpp>
#include <algoat/sorting/boolean_sort.hpp>
#include <stdexcept>

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

void sort_ndarray_float32(nb::ndarray<float, nb::ndim<1>, nb::c_contig> array) {
    algoat::sort(std::span<float>(array.data(), array.size()));
}
void sort_ndarray_float64(nb::ndarray<double, nb::ndim<1>, nb::c_contig> array) {
    algoat::sort(std::span<double>(array.data(), array.size()));
}
void sort_ndarray_int32(nb::ndarray<int32_t, nb::ndim<1>, nb::c_contig> array) {
    algoat::sort(std::span<int32_t>(array.data(), array.size()));
}
void sort_ndarray_int64(nb::ndarray<int64_t, nb::ndim<1>, nb::c_contig> array) {
    algoat::sort(std::span<int64_t>(array.data(), array.size()));
}
void sort_ndarray_uint32(nb::ndarray<uint32_t, nb::ndim<1>, nb::c_contig> array) {
    algoat::sort(std::span<uint32_t>(array.data(), array.size()));
}
void sort_ndarray_uint64(nb::ndarray<uint64_t, nb::ndim<1>, nb::c_contig> array) {
    algoat::sort(std::span<uint64_t>(array.data(), array.size()));
}

NB_MODULE(_algoat_impl, m) {
    m.doc() = "Algoat C++ algorithm library bindings";

    m.def("load_global_config", &algoat::load_global_config, nb::arg("filepath"),
          "Load algorithm configuration from a JSON file");

    m.def("sort", &sort_dispatch, nb::arg("data"), "Sort a list of mixed types");
    m.def("sort_inplace", &sort_inplace_dispatch, nb::arg("data"), "Sort a list in-place");
    m.def("search", &search_dispatch, nb::arg("data"), nb::arg("target"),
          "Search for a target in a list");
    m.def("search_many", &search_many_dispatch, nb::arg("data"), nb::arg("targets"),
          "Search for multiple targets in a list");

    m.def("sort_numpy_f16", &sort_ndarray_float16_buffer, nb::arg("array").noconvert());
    m.def("sort_numpy_bool", &sort_ndarray_bool_buffer, nb::arg("array").noconvert());
    m.def("sort_numpy", &sort_ndarray_float32, nb::arg("array").noconvert());
    m.def("sort_numpy", &sort_ndarray_float64, nb::arg("array").noconvert());
    m.def("sort_numpy", &sort_ndarray_int32, nb::arg("array").noconvert());
    m.def("sort_numpy", &sort_ndarray_int64, nb::arg("array").noconvert());
    m.def("sort_numpy", &sort_ndarray_uint32, nb::arg("array").noconvert());
    m.def("sort_numpy", &sort_ndarray_uint64, nb::arg("array").noconvert());

    m.def("search_numpy", &search_ndarray_typed<float>, nb::arg("array").noconvert(),
          nb::arg("target"), nb::call_guard<nb::gil_scoped_release>());
    m.def("search_numpy", &search_ndarray_typed<double>, nb::arg("array").noconvert(),
          nb::arg("target"), nb::call_guard<nb::gil_scoped_release>());
    m.def("search_numpy", &search_ndarray_typed<int32_t>, nb::arg("array").noconvert(),
          nb::arg("target"), nb::call_guard<nb::gil_scoped_release>());
    m.def("search_numpy", &search_ndarray_typed<int64_t>, nb::arg("array").noconvert(),
          nb::arg("target"), nb::call_guard<nb::gil_scoped_release>());
    m.def("search_numpy", &search_ndarray_typed<uint32_t>, nb::arg("array").noconvert(),
          nb::arg("target"), nb::call_guard<nb::gil_scoped_release>());
    m.def("search_numpy", &search_ndarray_typed<uint64_t>, nb::arg("array").noconvert(),
          nb::arg("target"), nb::call_guard<nb::gil_scoped_release>());

    m.def("search_many_numpy", &search_many_ndarray_typed<float>, nb::arg("array").noconvert(),
          nb::arg("targets").noconvert(), nb::call_guard<nb::gil_scoped_release>());
    m.def("search_many_numpy", &search_many_ndarray_typed<double>, nb::arg("array").noconvert(),
          nb::arg("targets").noconvert(), nb::call_guard<nb::gil_scoped_release>());
    m.def("search_many_numpy", &search_many_ndarray_typed<int32_t>, nb::arg("array").noconvert(),
          nb::arg("targets").noconvert(), nb::call_guard<nb::gil_scoped_release>());
    m.def("search_many_numpy", &search_many_ndarray_typed<int64_t>, nb::arg("array").noconvert(),
          nb::arg("targets").noconvert(), nb::call_guard<nb::gil_scoped_release>());
    m.def("search_many_numpy", &search_many_ndarray_typed<uint32_t>, nb::arg("array").noconvert(),
          nb::arg("targets").noconvert(), nb::call_guard<nb::gil_scoped_release>());
    m.def("search_many_numpy", &search_many_ndarray_typed<uint64_t>, nb::arg("array").noconvert(),
          nb::arg("targets").noconvert(), nb::call_guard<nb::gil_scoped_release>());

    m.def("sort_numpy_c64", [](nb::ndarray<std::complex<float>, nb::ndim<1>, nb::c_contig> array) {
        algoat::numerics::sort_complex_morton(
            std::span<std::complex<float>>(array.data(), array.size()));
    });
    m.def("sort_numpy_c128",
          [](nb::ndarray<std::complex<double>, nb::ndim<1>, nb::c_contig> array) {
              algoat::numerics::sort_complex_morton(
                  std::span<std::complex<double>>(array.data(), array.size()));
          });

    nb::class_<algoat::Rational>(m, "Rational")
        .def(nb::init<int64_t, int64_t>())
        .def_rw("num", &algoat::Rational::num)
        .def_rw("den", &algoat::Rational::den)
        .def("__lt__", [](const algoat::Rational& a, const algoat::Rational& b) { return a < b; })
        .def("__eq__", [](const algoat::Rational& a, const algoat::Rational& b) { return a == b; });
}
