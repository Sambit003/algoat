#pragma once

#include <Python.h>
#include <vector>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <algoat/numerics/morton.hpp>

namespace algoat {
namespace pybind {

struct PyComplexWrapper {
    PyObject* obj;
    std::complex<float> val;

    PyComplexWrapper() : obj(nullptr), val(0.0f, 0.0f) {}
    PyComplexWrapper(PyObject* o) : obj(o) {
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

struct PyFloatWrapper {
    PyObject* obj;
    double val;

    PyFloatWrapper() : obj(nullptr), val(0.0) {}
    PyFloatWrapper(PyObject* o) : obj(o) {
        if (PyFloat_Check(o)) val = PyFloat_AsDouble(o);
        else val = (double)PyLong_AsLongLong(o);
    }

    bool operator<(const PyFloatWrapper& other) const { return val < other.val; }
    bool operator<=(const PyFloatWrapper& other) const { return val <= other.val; }
    bool operator>(const PyFloatWrapper& other) const { return val > other.val; }
    bool operator>=(const PyFloatWrapper& other) const { return val >= other.val; }
    bool operator==(const PyFloatWrapper& other) const { return val == other.val; }
};

struct PyBigIntWrapper {
    PyObject* obj;
    int64_t small_val;
    bool is_small;

    PyBigIntWrapper() : obj(nullptr), small_val(0), is_small(true) {}
    PyBigIntWrapper(PyObject* o) : obj(o), small_val(0), is_small(true) {
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

struct PyStringWrapper {
    PyObject* obj;
    std::string_view sv;

    PyStringWrapper() : obj(nullptr), sv() {}
    PyStringWrapper(PyObject* o) : obj(o) {
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

struct PyGenericWrapper {
    PyObject* obj;

    PyGenericWrapper() : obj(nullptr) {}
    PyGenericWrapper(PyObject* o) : obj(o) {}

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
