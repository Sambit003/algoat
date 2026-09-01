import pytest
import algoat

def test_sort_ints():
    data = [5, 2, 8, 1, 9]
    sorted_data = algoat.sort(data)
    assert sorted_data == [1, 2, 5, 8, 9]
    assert data == [5, 2, 8, 1, 9] # original is unmodified

def test_sort_floats():
    data = [5.5, 2.2, 8.8, 1.1, 9.9]
    sorted_data = algoat.sort(data)
    assert sorted_data == [1.1, 2.2, 5.5, 8.8, 9.9]

def test_sort_large_ints():
    data = [2**80, -(2**70), 3, 2**100, -5]
    sorted_data = algoat.sort(data)
    assert sorted_data == [-(2**70), -5, 3, 2**80, 2**100]

def test_search_found():
    data = [1, 2, 5, 8, 9]
    index = algoat.search(data, 5)
    assert index == 2

def test_search_not_found():
    data = [1, 2, 5, 8, 9]
    index = algoat.search(data, 3)
    assert index is None

def test_empty_list():
    assert algoat.sort([]) == []
    assert algoat.search([], 5) is None

def test_search_large_list():
    # A large list that would timeout/be very slow if O(N) copy is used
    n = 100_000
    data = list(range(n))
    # Should be instant with zero-copy binary search
    assert algoat.search(data, 99_999) == 99_999
    assert algoat.search(data, -1) is None
    assert algoat.search(data, n) is None

def test_search_duplicates():
    data = [1, 2, 2, 2, 5]
    index = algoat.search(data, 2)
    # Binary search could return any index of the matching element
    assert index in (1, 2, 3)

def test_rational_sort():
    r1 = algoat.Rational(1, 2)
    r2 = algoat.Rational(1, 3)
    r3 = algoat.Rational(3, 4)
    data = [r1, r2, r3]
    sorted_data = algoat.sort(data)
    assert sorted_data == [r2, r1, r3]

def test_numpy_specialized_sorts():
    import numpy as np

    # Float16
    f16 = np.array([3.5, -1.2, 0.0, 100.0, -50.5], dtype=np.float16)
    res_f16 = algoat.sort(f16)
    assert np.array_equal(res_f16, np.sort(f16))

    # Bool
    b = np.array([True, False, True, False, True], dtype=np.bool_)
    res_b = algoat.sort(b)
    assert np.array_equal(res_b, np.array([False, False, True, True, True]))

    # Complex64
    c64 = np.array([1+2j, 0+0j, -1-1j, 2+1j], dtype=np.complex64)
    res_c64 = algoat.sort(c64)
    assert len(res_c64) == 4


def test_search_many_list():
    data = [10, 20, 30, 40, 50]
    targets = [20, 5, 50, 35, 10]
    results = algoat.search_many(data, targets)
    assert results == [1, None, 4, None, 0]

    # Empty targets
    assert algoat.search_many(data, []) == []

    # Empty data
    assert algoat.search_many([], [10, 20]) == [None, None]


def test_search_many_numpy():
    import numpy as np

    # Int64
    arr_i64 = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    targets_i64 = np.array([20, 5, 50, 35, 10], dtype=np.int64)
    res_i64 = algoat.search_many(arr_i64, targets_i64)
    assert res_i64 == [1, None, 4, None, 0]

    # Int32
    arr_i32 = np.array([1, 3, 5, 7, 9], dtype=np.int32)
    targets_i32 = np.array([1, 7, 10], dtype=np.int32)
    res_i32 = algoat.search_many(arr_i32, targets_i32)
    assert res_i32 == [0, 3, None]

    # Float64
    arr_f64 = np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float64)
    targets_f64 = np.array([3.3, 0.0, 4.4], dtype=np.float64)
    res_f64 = algoat.search_many(arr_f64, targets_f64)
    assert res_f64 == [2, None, 3]


def test_search_numpy_unsigned_and_float32():
    import numpy as np

    # np.uint32
    arr_u32 = np.array([10, 25, 50, 100, 250], dtype=np.uint32)
    assert algoat.search(arr_u32, 50) == 2
    assert algoat.search(arr_u32, 5) is None
    assert algoat.search(arr_u32, 300) is None
    targets_u32 = np.array([10, 30, 250, 5], dtype=np.uint32)
    assert algoat.search_many(arr_u32, targets_u32) == [0, None, 4, None]

    # np.uint64
    arr_u64 = np.array([2**33, 2**34, 2**35, 2**36], dtype=np.uint64)
    assert algoat.search(arr_u64, 2**34) == 1
    assert algoat.search(arr_u64, 100) is None
    assert algoat.search(arr_u64, 2**37) is None
    targets_u64 = np.array([2**35, 2**33, 2**40], dtype=np.uint64)
    assert algoat.search_many(arr_u64, targets_u64) == [2, 0, None]

    # np.float32
    arr_f32 = np.array([1.5, 3.25, 7.125, 15.5], dtype=np.float32)
    assert algoat.search(arr_f32, np.float32(3.25)) == 1
    assert algoat.search(arr_f32, np.float32(2.0)) is None
    targets_f32 = np.array([15.5, 0.5, 1.5, 8.0], dtype=np.float32)
    assert algoat.search_many(arr_f32, targets_f32) == [3, None, 0, None]


def test_search_negative_numbers_list():
    data = [-100, -50, -25, -5, 0, 15, 30]
    assert algoat.search(data, -50) == 1
    assert algoat.search(data, -5) == 3
    assert algoat.search(data, 0) == 4
    assert algoat.search(data, -200) is None
    assert algoat.search(data, -10) is None
    assert algoat.search(data, 100) is None

    # Search many
    targets = [-5, -100, -30, 30, -500]
    assert algoat.search_many(data, targets) == [3, 0, None, 6, None]


def test_search_negative_numbers_numpy():
    import numpy as np

    # Int64 with negative numbers
    arr_i64 = np.array([-100, -50, -25, 0, 25, 50], dtype=np.int64)
    assert algoat.search(arr_i64, -50) == 1
    assert algoat.search(arr_i64, 0) == 3
    assert algoat.search(arr_i64, -150) is None
    targets_i64 = np.array([-100, -30, 50, 100], dtype=np.int64)
    assert algoat.search_many(arr_i64, targets_i64) == [0, None, 5, None]

    # Int32 with negative numbers
    arr_i32 = np.array([-80, -40, -10, 5, 20], dtype=np.int32)
    assert algoat.search(arr_i32, -40) == 1
    assert algoat.search(arr_i32, -50) is None
    targets_i32 = np.array([-80, 20, 0], dtype=np.int32)
    assert algoat.search_many(arr_i32, targets_i32) == [0, 4, None]

    # Float64 with negative numbers
    arr_f64 = np.array([-99.5, -45.25, -0.5, 0.0, 12.75], dtype=np.float64)
    assert algoat.search(arr_f64, -45.25) == 1
    assert algoat.search(arr_f64, -10.0) is None
    targets_f64 = np.array([12.75, -99.5, 100.0], dtype=np.float64)
    assert algoat.search_many(arr_f64, targets_f64) == [4, 0, None]

    # Float32 with negative numbers
    arr_f32 = np.array([-50.5, -20.25, -1.0, 3.5], dtype=np.float32)
    assert algoat.search(arr_f32, np.float32(-20.25)) == 1
    assert algoat.search(arr_f32, np.float32(-5.0)) is None
    targets_f32 = np.array([-50.5, 3.5, 0.0], dtype=np.float32)
    assert algoat.search_many(arr_f32, targets_f32) == [0, 3, None]


