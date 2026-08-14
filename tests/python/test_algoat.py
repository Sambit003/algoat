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
