package algoat

/*
#cgo CFLAGS: -I${SRCDIR}/../../include
#cgo CXXFLAGS: -std=c++20 -I${SRCDIR}/../../include
#cgo LDFLAGS: -L${SRCDIR}/../../build/src -L${SRCDIR}/../../build -L${SRCDIR}/../../build/release/src -L${SRCDIR}/../../build/debug/src -lalgoat -latomic -lstdc++ -lm -lpthread
#include <algoat/c_api/algoat_c.h>
*/
import "C"

import (
	"reflect"
	"runtime"
	"slices"
	"unsafe"

	"golang.org/x/exp/constraints"
)

// Number represents any ordered integer or floating-point number.
type Number interface {
	constraints.Integer | constraints.Float
}

// dispatchSort calls the appropriate C-ABI sorting entry point based on element reflection Kind.
// Raw pointer is passed directly without memory copying or allocation.
func dispatchSort(ptr unsafe.Pointer, n int, kind reflect.Kind, parallel int) bool {
	cLen := C.size_t(n)
	cPar := C.int(parallel)

	switch kind {
	case reflect.Int64:
		C.algoat_sort_int64((*C.int64_t)(ptr), cLen, cPar)
	case reflect.Uint64:
		C.algoat_sort_uint64((*C.uint64_t)(ptr), cLen, cPar)
	case reflect.Int:
		if unsafe.Sizeof(int(0)) == 8 {
			C.algoat_sort_int64((*C.int64_t)(ptr), cLen, cPar)
		} else {
			C.algoat_sort_int32((*C.int32_t)(ptr), cLen, cPar)
		}
	case reflect.Uint, reflect.Uintptr:
		if unsafe.Sizeof(uint(0)) == 8 {
			C.algoat_sort_uint64((*C.uint64_t)(ptr), cLen, cPar)
		} else {
			C.algoat_sort_uint32((*C.uint32_t)(ptr), cLen, cPar)
		}
	case reflect.Int32:
		C.algoat_sort_int32((*C.int32_t)(ptr), cLen, cPar)
	case reflect.Uint32:
		C.algoat_sort_uint32((*C.uint32_t)(ptr), cLen, cPar)
	case reflect.Float64:
		C.algoat_sort_float64((*C.double)(ptr), cLen, cPar)
	case reflect.Float32:
		C.algoat_sort_float32((*C.float)(ptr), cLen, cPar)
	default:
		return false
	}
	return true
}

// dispatchRadixSort calls the appropriate C-ABI radix sorting entry point.
func dispatchRadixSort(ptr unsafe.Pointer, n int, kind reflect.Kind) bool {
	cLen := C.size_t(n)

	switch kind {
	case reflect.Int64:
		C.algoat_radix_sort_int64((*C.int64_t)(ptr), cLen)
	case reflect.Uint64:
		C.algoat_radix_sort_uint64((*C.uint64_t)(ptr), cLen)
	case reflect.Int:
		if unsafe.Sizeof(int(0)) == 8 {
			C.algoat_radix_sort_int64((*C.int64_t)(ptr), cLen)
		} else {
			C.algoat_radix_sort_int32((*C.int32_t)(ptr), cLen)
		}
	case reflect.Uint, reflect.Uintptr:
		if unsafe.Sizeof(uint(0)) == 8 {
			C.algoat_radix_sort_uint64((*C.uint64_t)(ptr), cLen)
		} else {
			C.algoat_radix_sort_uint32((*C.uint32_t)(ptr), cLen)
		}
	case reflect.Int32:
		C.algoat_radix_sort_int32((*C.int32_t)(ptr), cLen)
	case reflect.Uint32:
		C.algoat_radix_sort_uint32((*C.uint32_t)(ptr), cLen)
	default:
		return false
	}
	return true
}

// dispatchSearch calls the appropriate C-ABI branchless lower-bound search entry point.
func dispatchSearch[T Number](ptr unsafe.Pointer, n int, kind reflect.Kind, target T) (int, bool) {
	cLen := C.size_t(n)

	switch kind {
	case reflect.Int64:
		return int(C.algoat_search_int64((*C.int64_t)(ptr), cLen, C.int64_t(int64(target)))), true
	case reflect.Uint64:
		return int(C.algoat_search_uint64((*C.uint64_t)(ptr), cLen, C.uint64_t(uint64(target)))), true
	case reflect.Int:
		if unsafe.Sizeof(int(0)) == 8 {
			return int(C.algoat_search_int64((*C.int64_t)(ptr), cLen, C.int64_t(int(target)))), true
		}
		return int(C.algoat_search_int32((*C.int32_t)(ptr), cLen, C.int32_t(int(target)))), true
	case reflect.Uint, reflect.Uintptr:
		if unsafe.Sizeof(uint(0)) == 8 {
			return int(C.algoat_search_uint64((*C.uint64_t)(ptr), cLen, C.uint64_t(uint(target)))), true
		}
		return int(C.algoat_search_uint32((*C.uint32_t)(ptr), cLen, C.uint32_t(uint(target)))), true
	case reflect.Int32:
		return int(C.algoat_search_int32((*C.int32_t)(ptr), cLen, C.int32_t(int32(target)))), true
	case reflect.Uint32:
		return int(C.algoat_search_uint32((*C.uint32_t)(ptr), cLen, C.uint32_t(uint32(target)))), true
	case reflect.Float64:
		return int(C.algoat_search_float64((*C.double)(ptr), cLen, C.double(float64(target)))), true
	case reflect.Float32:
		return int(C.algoat_search_float32((*C.float)(ptr), cLen, C.float(float32(target)))), true
	default:
		return 0, false
	}
}

// dispatchTopK calls the appropriate C-ABI top-k selection entry point.
func dispatchTopK(ptr unsafe.Pointer, n int, kind reflect.Kind, k int) bool {
	cLen := C.size_t(n)
	cK := C.size_t(k)

	switch kind {
	case reflect.Int64:
		C.algoat_topk_int64((*C.int64_t)(ptr), cLen, cK)
	case reflect.Uint64:
		C.algoat_topk_uint64((*C.uint64_t)(ptr), cLen, cK)
	case reflect.Int:
		if unsafe.Sizeof(int(0)) == 8 {
			C.algoat_topk_int64((*C.int64_t)(ptr), cLen, cK)
		} else {
			C.algoat_topk_int32((*C.int32_t)(ptr), cLen, cK)
		}
	case reflect.Uint, reflect.Uintptr:
		if unsafe.Sizeof(uint(0)) == 8 {
			C.algoat_topk_uint64((*C.uint64_t)(ptr), cLen, cK)
		} else {
			C.algoat_topk_uint32((*C.uint32_t)(ptr), cLen, cK)
		}
	case reflect.Int32:
		C.algoat_topk_int32((*C.int32_t)(ptr), cLen, cK)
	case reflect.Uint32:
		C.algoat_topk_uint32((*C.uint32_t)(ptr), cLen, cK)
	case reflect.Float64:
		C.algoat_topk_float64((*C.double)(ptr), cLen, cK)
	case reflect.Float32:
		C.algoat_topk_float32((*C.float)(ptr), cLen, cK)
	default:
		return false
	}
	return true
}

// Sort sorts the given slice in-place in ascending order using Algoat's
// multi-threaded parallel sorting engine (IPS⁴o) for large slices, or PDQSort
// for sequential execution. The operation is zero-copy and allocation-free.
func Sort[T Number](slice []T) {
	n := len(slice)
	if n <= 1 {
		return
	}
	if n == 2 {
		if slice[0] > slice[1] {
			slice[0], slice[1] = slice[1], slice[0]
		}
		return
	}
	if cap(slice) == 0 {
		return
	}

	var p runtime.Pinner
	ptr := unsafe.Pointer(unsafe.SliceData(slice))
	p.Pin(ptr)
	defer p.Unpin()

	kind := reflect.TypeOf(slice).Elem().Kind()
	if !dispatchSort(ptr, n, kind, 1) {
		slices.Sort(slice)
	}
}

// SortSequential sorts the given slice in-place in ascending order using Algoat's
// single-threaded pattern-defeating quicksort (PDQSort) engine.
func SortSequential[T Number](slice []T) {
	n := len(slice)
	if n <= 1 {
		return
	}
	if n == 2 {
		if slice[0] > slice[1] {
			slice[0], slice[1] = slice[1], slice[0]
		}
		return
	}
	if cap(slice) == 0 {
		return
	}

	var p runtime.Pinner
	ptr := unsafe.Pointer(unsafe.SliceData(slice))
	p.Pin(ptr)
	defer p.Unpin()

	kind := reflect.TypeOf(slice).Elem().Kind()
	if !dispatchSort(ptr, n, kind, 0) {
		slices.Sort(slice)
	}
}

// SortRadix sorts an integer slice in linear time using Algoat's Least Significant
// Digit (LSD) Radix Sort engine.
func SortRadix[T constraints.Integer](slice []T) {
	n := len(slice)
	if n <= 1 {
		return
	}
	if n == 2 {
		if slice[0] > slice[1] {
			slice[0], slice[1] = slice[1], slice[0]
		}
		return
	}
	if cap(slice) == 0 {
		return
	}

	var p runtime.Pinner
	ptr := unsafe.Pointer(unsafe.SliceData(slice))
	p.Pin(ptr)
	defer p.Unpin()

	kind := reflect.TypeOf(slice).Elem().Kind()
	if !dispatchRadixSort(ptr, n, kind) {
		slices.Sort(slice)
	}
}

// Search performs a branchless lower-bound binary search for target in slice.
// It returns the index of the first element not less than target, or len(slice)
// if all elements are less than target. The slice must be sorted in ascending order.
func Search[T Number](slice []T, target T) int {
	n := len(slice)
	if n == 0 {
		return 0
	}
	if cap(slice) == 0 {
		idx, _ := slices.BinarySearch(slice, target)
		return idx
	}

	var p runtime.Pinner
	ptr := unsafe.Pointer(unsafe.SliceData(slice))
	p.Pin(ptr)
	defer p.Unpin()

	kind := reflect.TypeOf(slice).Elem().Kind()
	if idx, ok := dispatchSearch(ptr, n, kind, target); ok {
		return idx
	}

	idx, _ := slices.BinarySearch(slice, target)
	return idx
}

// TopK rearranges slice so that the prefix subrange [0, k) contains the k
// smallest elements in strictly sorted ascending order. It returns the zero-copy
// subslice slice[:k].
func TopK[T Number](slice []T, k int) []T {
	if k <= 0 {
		return slice[:0]
	}
	n := len(slice)
	if k >= n {
		k = n
	}
	if n <= 1 {
		return slice[:k]
	}
	if cap(slice) == 0 {
		slices.Sort(slice)
		return slice[:k]
	}

	var p runtime.Pinner
	ptr := unsafe.Pointer(unsafe.SliceData(slice))
	p.Pin(ptr)
	defer p.Unpin()

	kind := reflect.TypeOf(slice).Elem().Kind()
	if !dispatchTopK(ptr, n, kind, k) {
		slices.Sort(slice)
	}

	return slice[:k]
}
