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

