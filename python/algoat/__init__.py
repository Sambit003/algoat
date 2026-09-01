"""Algoat: High-performance C++20 and Python algorithm library.

Features smart dynamic fallback dispatch based on data profiles,
domain-specific non-comparative sorting for Float16, boolean, and complex arrays,
and zero-copy NumPy interoperability.
"""

from . import _algoat_impl
from ._algoat_impl import sort_inplace, load_global_config, Rational
import numpy as np
from typing import Union, List, Any, Optional

def sort(data: Union[np.ndarray, List[Any]]) -> Union[np.ndarray, List[Any]]:
    """Sort an array or list using smart dynamic algorithm dispatch.

    If given a NumPy ndarray, operates in-place with zero-copy C++ execution:
    - `np.float16`: Dispatches to O(N) bit-flipping counting/radix sort.
    - `np.bool_`: Dispatches to branchless O(N) counting pass and memset.
    - `np.complex64` / `np.complex128`: Dispatches to 2D Morton Z-order curve spatial radix sort.
    - Other numeric types (float32, float64, int32, int64, etc.): Dispatches to dynamic C++ Dispatcher.

    If given a standard Python list, returns a newly allocated sorted list without mutating the original.

    Args:
        data: A NumPy 1D array or a Python list of comparable elements.

    Returns:
        The sorted NumPy array (in-place) or a new sorted Python list.
    """
    if isinstance(data, np.ndarray):
        if data.dtype == np.float16:
            # We need to view the array as uint16_t to pass it to the C++ extension
            view_arr = data.view(np.uint16)
            _algoat_impl.sort_numpy_f16(view_arr)
            return data
        elif data.dtype == np.bool_:
            view_arr = data.view(np.uint8)
            _algoat_impl.sort_numpy_bool(view_arr)
            return data
        elif data.dtype == np.complex64:
            view_arr = data.view(np.complex64)
            _algoat_impl.sort_numpy_c64(view_arr)
            return data
        elif data.dtype == np.complex128:
            view_arr = data.view(np.complex128)
            _algoat_impl.sort_numpy_c128(view_arr)
            return data
        else:
            _algoat_impl.sort_numpy(data)
            return data
    else:
        return _algoat_impl.sort(data)


_search_impl = _algoat_impl.search
_search_numpy_impl = _algoat_impl.search_numpy
_search_many_impl = _algoat_impl.search_many
_search_many_numpy_impl = _algoat_impl.search_many_numpy


def search(data: Union[np.ndarray, List[Any]], target: Any) -> Optional[int]:
    """Search for a target value within a sorted array or list using branchless bisection.

    If given a NumPy ndarray, performs zero-copy branchless C++ search directly on contiguous memory.
    If given a standard Python list, searches using fast branchless traversal.

    Args:
        data: A sorted NumPy 1D array or sorted Python list.
        target: The value to locate.

    Returns:
        Index of the matching element if found, or None.
    """
    if type(data) is list:
        return _search_impl(data, target)
    elif isinstance(data, np.ndarray):
        return _search_numpy_impl(data, target)
    return _search_impl(data, target)


def search_many(
    data: Union[np.ndarray, List[Any]], targets: Union[np.ndarray, List[Any]]
) -> List[Optional[int]]:
    """Batch search for multiple target values with amortized FFI overhead.

    Args:
        data: A sorted array or list to search within.
        targets: An array or list of target values to locate.

    Returns:
        A list of matching indices (or None for targets not found).
    """
    if type(data) is list and type(targets) is list:
        return _search_many_impl(data, targets)
    elif isinstance(data, np.ndarray) and isinstance(targets, np.ndarray):
        return _search_many_numpy_impl(data, targets)
    else:
        list_data = data if isinstance(data, list) else list(data)
        list_targets = targets if isinstance(targets, list) else list(targets)
        return _search_many_impl(list_data, list_targets)



__all__ = ["sort", "sort_inplace", "search", "search_many", "load_global_config", "Rational"]

