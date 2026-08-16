/**
 * @file python_wrappers.hpp
 * @brief Zero-overhead C++ comparator wrappers for Python objects.
 * 
 * Provides type-specialized wrapper structs around raw `PyObject*` pointers to allow
 * standard C++ sorting algorithms (`algoat::sort`, `IntroSort`, etc.) to operate directly
 * on Python sequences while maintaining full compatibility with Python comparison protocols
 * and GIL state management.
 */

#pragma once

#include <Python.h>
#include <vector>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <algoat/numerics/morton.hpp>

namespace algoat {
namespace pybind {

/**
 * @struct PyComplexWrapper
 * @brief Unboxes `PyComplexObject` into a `std::complex<float>` for 2D Morton Z-order curve comparisons.
 */
struct PyComplexWrapper {
    PyObject* obj;           ///< Borrowed reference to Python complex object.
    std::complex<float> val; ///< Cached floating-point complex value.

    PyComplexWrapper() : obj(nullptr), val(0.0f, 0.0f) {}
    explicit PyComplexWrapper(PyObject* o) : obj(o) {
        if (PyComplex_Check(o)) {
            val = std::complex<float>(
                static_cast<float>(PyComplex_RealAsDouble(o)),
                static_cast<float>(PyComplex_ImagAsDouble(o))
            );
        } else {
            val = std::complex<float>(0.0f, 0.0f);
        }
    }

    bool operator<(const PyComplexWrapper& other) const { return algoat::numerics::MortonCompare{}(val, other.val); }
    bool operator==(const PyComplexWrapper& other) const { return val == other.val; }
    bool operator>(const PyComplexWrapper& other) const { return other < *this; }
    bool operator<=(const PyComplexWrapper& other) const { return !(*this > other); }
    bool operator>=(const PyComplexWrapper& other) const { return !(*this < other); }
};

/**
 * @struct PyFloatWrapper
 * @brief Unboxes Python `float` and `int` objects into native C++ `double` values.
 * 
 * Avoids repeated Python API calls during sorting comparisons by unboxing once.
 */
struct PyFloatWrapper {
    PyObject* obj; ///< Borrowed reference to original Python object.
    double val;    ///< Unboxed native double.

    PyFloatWrapper() : obj(nullptr), val(0.0) {}
    explicit PyFloatWrapper(PyObject* o) : obj(o) {
        if (PyFloat_Check(o)) val = PyFloat_AsDouble(o);
        else val = (double)PyLong_AsLongLong(o);
    }

    bool operator<(const PyFloatWrapper& other) const { return val < other.val; }
    bool operator<=(const PyFloatWrapper& other) const { return val <= other.val; }
    bool operator>(const PyFloatWrapper& other) const { return val > other.val; }
    bool operator>=(const PyFloatWrapper& other) const { return val >= other.val; }
    bool operator==(const PyFloatWrapper& other) const { return val == other.val; }
};

/**
 * @struct PyBigIntWrapper
 * @brief Optimized wrapper for Python integers with small-integer fast path.
 * 
 * Fits integers within `int64_t` into `small_val` for branchless hardware integer comparison.
 * Falls back to `PyObject_RichCompareBool` with GIL acquisition for arbitrary-precision integers.
 */
struct PyBigIntWrapper {
    PyObject* obj;     ///< Borrowed reference to Python object.
    int64_t small_val; ///< Unboxed 64-bit integer value if fitting.
    bool is_small;     ///< True if integer fits in 64 bits without overflow.

    PyBigIntWrapper() : obj(nullptr), small_val(0), is_small(true) {}
    explicit PyBigIntWrapper(PyObject* o) : obj(o), small_val(0), is_small(true) {
        if (PyLong_Check(o)) {
            int overflow = 0;
            long long v = PyLong_AsLongLongAndOverflow(o, &overflow);
            if (overflow == 0) {
                small_val = static_cast<int64_t>(v);
                is_small = true;
            } else {
                is_small = false;
            }
        } else if (PyFloat_Check(o)) {
            small_val = static_cast<int64_t>(PyFloat_AsDouble(o));
            is_small = true;
        }
    }

    /**
     * @brief Performs Python rich comparison with safe GIL state management.
     */
    static bool rich_compare(PyObject* lhs, PyObject* rhs, int op) {
        PyGILState_STATE gstate = PyGILState_Ensure();
        int res = PyObject_RichCompareBool(lhs, rhs, op);
        if (res == -1) {
            PyErr_Clear();
            res = 0;
        }
        PyGILState_Release(gstate);
        return res == 1;
    }

    bool operator<(const PyBigIntWrapper& other) const {
        if (is_small && other.is_small) return small_val < other.small_val;
        return rich_compare(obj, other.obj, Py_LT);
    }
    bool operator<=(const PyBigIntWrapper& other) const {
        if (is_small && other.is_small) return small_val <= other.small_val;
        return rich_compare(obj, other.obj, Py_LE);
    }
    bool operator>(const PyBigIntWrapper& other) const {
        if (is_small && other.is_small) return small_val > other.small_val;
        return rich_compare(obj, other.obj, Py_GT);
    }
    bool operator>=(const PyBigIntWrapper& other) const {
        if (is_small && other.is_small) return small_val >= other.small_val;
        return rich_compare(obj, other.obj, Py_GE);
    }
    bool operator==(const PyBigIntWrapper& other) const {
        if (is_small && other.is_small) return small_val == other.small_val;
        return rich_compare(obj, other.obj, Py_EQ);
    }
};

/**
 * @struct PyStringWrapper
 * @brief Zero-copy wrapper creating a `std::string_view` over Python UTF-8 string memory.
 */
struct PyStringWrapper {
    PyObject* obj;       ///< Borrowed reference to Python unicode string.
    std::string_view sv; ///< Zero-copy view into Python's internal UTF-8 buffer.

    PyStringWrapper() : obj(nullptr), sv() {}
    explicit PyStringWrapper(PyObject* o) : obj(o) {
        if (PyUnicode_Check(o)) {
            Py_ssize_t len = 0;
            const char* ptr = PyUnicode_AsUTF8AndSize(o, &len);
            if (ptr) sv = std::string_view(ptr, len);
        }
    }
    bool operator<(const PyStringWrapper& other) const {
        if (obj == other.obj) return false;
        return sv < other.sv;
    }
    bool operator<=(const PyStringWrapper& other) const {
        if (obj == other.obj) return true;
        return sv <= other.sv;
    }
    bool operator>(const PyStringWrapper& other) const {
        if (obj == other.obj) return false;
        return sv > other.sv;
    }
    bool operator>=(const PyStringWrapper& other) const {
        if (obj == other.obj) return true;
        return sv >= other.sv;
    }
    bool operator==(const PyStringWrapper& other) const {
        if (obj == other.obj) return true;
        return sv == other.sv;
    }
};

/**
 * @struct PyGenericWrapper
 * @brief Generic wrapper delegating all comparisons to Python's `PyObject_RichCompareBool`.
 */
struct PyGenericWrapper {
    PyObject* obj; ///< Borrowed reference to arbitrary Python object.

    PyGenericWrapper() : obj(nullptr) {}
    explicit PyGenericWrapper(PyObject* o) : obj(o) {}

    bool operator<(const PyGenericWrapper& other) const {
        PyGILState_STATE gstate = PyGILState_Ensure();
        int res = PyObject_RichCompareBool(obj, other.obj, Py_LT);
        if (res == -1) { PyErr_Clear(); res = 0; }
        PyGILState_Release(gstate);
        return res == 1;
    }
    bool operator<=(const PyGenericWrapper& other) const {
        PyGILState_STATE gstate = PyGILState_Ensure();
        int res = PyObject_RichCompareBool(obj, other.obj, Py_LE);
        if (res == -1) { PyErr_Clear(); res = 0; }
        PyGILState_Release(gstate);
        return res == 1;
    }
    bool operator>(const PyGenericWrapper& other) const {
        PyGILState_STATE gstate = PyGILState_Ensure();
        int res = PyObject_RichCompareBool(obj, other.obj, Py_GT);
        if (res == -1) { PyErr_Clear(); res = 0; }
        PyGILState_Release(gstate);
        return res == 1;
    }
    bool operator>=(const PyGenericWrapper& other) const {
        PyGILState_STATE gstate = PyGILState_Ensure();
        int res = PyObject_RichCompareBool(obj, other.obj, Py_GE);
        if (res == -1) { PyErr_Clear(); res = 0; }
        PyGILState_Release(gstate);
        return res == 1;
    }
    bool operator==(const PyGenericWrapper& other) const {
        PyGILState_STATE gstate = PyGILState_Ensure();
        int res = PyObject_RichCompareBool(obj, other.obj, Py_EQ);
        if (res == -1) { PyErr_Clear(); res = 0; }
        PyGILState_Release(gstate);
        return res == 1;
    }
};

} // namespace pybind
} // namespace algoat
