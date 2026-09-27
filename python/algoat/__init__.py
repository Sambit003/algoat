"""Algoat: High-performance C++20 and Python algorithm library.

Features smart dynamic fallback dispatch based on data profiles,
domain-specific non-comparative sorting for Float16, boolean, and complex arrays,
zero-copy PEP 3118 memory views, and fine-grained GIL release for multi-core parallelism.
"""

from . import _algoat_impl
from ._algoat_impl import load_global_config, Rational
import numpy as np
from typing import Union, List, Any, Optional

def sort(data: Union[np.ndarray, List[Any], Any]) -> Union[np.ndarray, List[Any], Any]:
    """Sort an array, PEP 3118 buffer, or list using smart dynamic algorithm dispatch.

    If given a standard Python list, returns a newly allocated sorted list without mutating the original.
    If given a NumPy ndarray or PEP 3118 buffer (array.array, bytearray), operates in-place with
    zero-copy C++ execution and GIL release, and returns the sorted object:
    - `np.float16`: Dispatches to O(N) bit-flipping counting/radix sort.
    - `np.bool_`: Dispatches to branchless O(N) counting pass and memset.
    - `np.complex64` / `np.complex128`: Dispatches to 2D Morton Z-order curve spatial radix sort.
    - Other numeric dtypes and native buffers: Dispatches to dynamic C++ Dispatcher.

    Args:
        data: A NumPy 1D array, PEP 3118 buffer, or Python list of comparable elements.

    Returns:
        The sorted array/buffer (in-place) or a new sorted Python list.
    """
    if isinstance(data, list):
        return _algoat_impl.sort(data)
    sort_inplace(data)
    return data


def sort_inplace(data: Union[np.ndarray, List[Any], Any]) -> None:
    """Sort a NumPy array, PEP 3118 buffer (array.array, bytearray, memoryview), or Python list in-place with zero memory allocation.

    If given a NumPy ndarray:
    - Sorts directly in-place across the contiguous memory buffer with zero heap allocations.
    - Releases the Python GIL during compute to enable concurrent multi-core speedup across worker threads.
    - Requires contiguous memory (C-contiguous). Non-contiguous slices raise a ValueError.

    If given a PEP 3118 buffer (e.g., array.array, bytearray, memoryview):
    - Sorts directly in-place via zero-copy C++ backend with GIL release.

    If given a standard Python list:
    - Mutates the list in-place by rearranging its internal item pointers.

    Args:
        data: A mutable NumPy 1D array, PEP 3118 buffer, or Python list.
    """
    if isinstance(data, np.ndarray):
        if not data.flags.c_contiguous:
            raise ValueError("In-place sorting requires a contiguous array buffer.")
        if data.dtype == np.float16:
            view_arr = data.view(np.uint16)
            _algoat_impl.sort_numpy_f16(view_arr)
        elif data.dtype == np.bool_:
            view_arr = data.view(np.uint8)
            _algoat_impl.sort_numpy_bool(view_arr)
        elif data.dtype == np.complex64:
            view_arr = data.view(np.complex64)
            _algoat_impl.sort_numpy_c64(view_arr)
        elif data.dtype == np.complex128:
            view_arr = data.view(np.complex128)
            _algoat_impl.sort_numpy_c128(view_arr)
        else:
            _algoat_impl.sort_numpy(data)
    elif isinstance(data, list):
        _algoat_impl.sort_inplace(data)
    else:
        # Support PEP 3118 buffer protocol types (e.g. array.array, bytearray, memoryview)
        try:
            mv = memoryview(data)
        except TypeError:
            raise TypeError(f"sort_inplace is not supported for type {type(data).__name__}")

        if mv.readonly:
            raise TypeError(f"Cannot sort read-only buffer of type {type(data).__name__} in-place.")
        if not mv.c_contiguous:
            raise ValueError("In-place sorting requires a contiguous buffer.")

        try:
            _algoat_impl.sort_numpy(data)
        except TypeError as e:
            raise TypeError(f"Unsupported buffer element type for {type(data).__name__}: {e}")


_search_impl = _algoat_impl.search
_search_numpy_impl = _algoat_impl.search_numpy
_search_many_impl = _algoat_impl.search_many
_search_many_numpy_impl = _algoat_impl.search_many_numpy


def search(data: Union[np.ndarray, List[Any]], target: Any) -> Optional[int]:
    """Search for a target value within a sorted array or list using branchless bisection.

    If given a NumPy ndarray, performs zero-copy branchless C++ search directly on contiguous memory with GIL release.
    If given a standard Python list, searches using fast branchless traversal.

    Args:
        data: A sorted NumPy 1D array or sorted Python list.
        target: The value to locate.

    Returns:
        Index of the matching element if found, or None.
    """
    if isinstance(data, list):
        return _search_impl(data, target)
    elif isinstance(data, np.ndarray):
        return _search_numpy_impl(data, target)
    return _search_impl(data, target)


def search_many(
    data: Union[np.ndarray, List[Any]], targets: Union[np.ndarray, List[Any]]
) -> List[Optional[int]]:
    """Batch search for multiple target values with amortized FFI overhead and GIL release.

    Args:
        data: A sorted array or list to search within.
        targets: An array or list of target values to locate.

    Returns:
        A list of matching indices (or None for targets not found).
    """
    if isinstance(data, list) and isinstance(targets, list):
        return _search_many_impl(data, targets)
    elif isinstance(data, np.ndarray) and isinstance(targets, np.ndarray):
        return _search_many_numpy_impl(data, targets)
    else:
        list_data = data if isinstance(data, list) else list(data)
        list_targets = targets if isinstance(targets, list) else list(targets)
        return _search_many_impl(list_data, list_targets)


__all__ = ["sort", "sort_inplace", "search", "search_many", "load_global_config", "Rational"]
