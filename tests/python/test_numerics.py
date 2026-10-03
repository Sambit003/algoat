import pytest
import numpy as np
import algoat


@pytest.fixture(params=[np.complex64, np.complex128])
def complex_dtype(request):
    return request.param


def test_complex_morton_sort(complex_dtype):
    coords = np.array(
        [10.0 + 5.0j, 2.0 + 3.0j, 5.0 + 10.0j, 1.0 + 1.0j], dtype=complex_dtype
    )

    # Sort in-place using Morton curve
    algoat.sort(coords, curve="morton")
    expected = np.array(
        [1.0 + 1.0j, 2.0 + 3.0j, 10.0 + 5.0j, 5.0 + 10.0j], dtype=complex_dtype
    )
    assert np.allclose(coords, expected)


def test_complex_hilbert_sort(complex_dtype):
    coords = np.array(
        [10.0 + 5.0j, 2.0 + 3.0j, 5.0 + 10.0j, 1.0 + 1.0j], dtype=complex_dtype
    )

    algoat.sort(coords, curve="hilbert")
    expected = np.array(
        [1.0 + 1.0j, 2.0 + 3.0j, 10.0 + 5.0j, 5.0 + 10.0j], dtype=complex_dtype
    )
    assert np.allclose(coords, expected)


def test_complex_hybrid_sort(complex_dtype):
    coords = np.array(
        [10.0 + 5.0j, 2.0 + 3.0j, 5.0 + 10.0j, 1.0 + 1.0j], dtype=complex_dtype
    )

    algoat.sort(coords, curve="hybrid")
    expected = np.array(
        [1.0 + 1.0j, 2.0 + 3.0j, 10.0 + 5.0j, 5.0 + 10.0j], dtype=complex_dtype
    )
    assert np.allclose(coords, expected)


def test_invalid_curve_mode():
    coords = np.array([1.0 + 1.0j, 2.0 + 2.0j], dtype=np.complex64)
    with pytest.raises(ValueError, match="Unknown curve type: invalid_curve"):
        algoat.sort(coords, curve="invalid_curve")


def test_non_contiguous_complex_array():
    # Create a 2D array and take a column to make it non-contiguous
    arr_2d = np.array(
        [[1.0 + 1.0j, 2.0 + 2.0j], [3.0 + 3.0j, 4.0 + 4.0j]], dtype=np.complex64
    )
    non_contiguous_col = arr_2d[:, 0]

    assert not non_contiguous_col.flags.c_contiguous
    with pytest.raises(
        ValueError, match="In-place sorting requires a contiguous buffer."
    ):
        algoat.sort(non_contiguous_col, curve="morton")


def test_large_random_complex_arrays(complex_dtype):
    np.random.seed(42)
    base_dtype = np.float32 if complex_dtype == np.complex64 else np.float64
    coords = (
        np.random.uniform(-1000.0, 1000.0, size=10000)
        .astype(base_dtype)
        .view(complex_dtype)
    )

    # We will test on copies
    algoat.sort(coords.copy(), curve="morton")
    algoat.sort(coords.copy(), curve="hilbert")
    algoat.sort(coords.copy(), curve="hybrid")
    assert len(coords) == 5000


def test_python_list_complex_sort():
    coords = [10.0 + 5.0j, 2.0 + 3.0j, 5.0 + 10.0j, 1.0 + 1.0j]

    # Sort out-of-place
    res = algoat.sort(coords, curve="hilbert")
    expected = [1.0 + 1.0j, 2.0 + 3.0j, 10.0 + 5.0j, 5.0 + 10.0j]
    assert res == expected
    assert res is not coords

    # Sort in-place
    algoat.sort_inplace(coords, curve="hybrid")
    assert coords == expected
