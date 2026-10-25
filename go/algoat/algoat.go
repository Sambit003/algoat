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
	"unsafe"

	"github.com/Sambit003/algoat/go/algoat/internal/asm"
	"golang.org/x/exp/constraints"
)

// Number represents any ordered integer or floating-point number.
type Number interface {
	constraints.Integer | constraints.Float
}

// dispatchSort calls the appropriate C-ABI sorting entry point based on element reflection Kind.
// Raw pointer is passed directly without memory copying or allocation.
func dispatchSort(ptr unsafe.Pointer, n int, kind reflect.Kind, parallel int) {
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
	case reflect.Int16:
		C.algoat_sort_int16((*C.int16_t)(ptr), cLen, cPar)
	case reflect.Uint16:
		C.algoat_sort_uint16((*C.uint16_t)(ptr), cLen, cPar)
	case reflect.Int8:
		C.algoat_sort_int8((*C.int8_t)(ptr), cLen, cPar)
	case reflect.Uint8:
		C.algoat_sort_uint8((*C.uint8_t)(ptr), cLen, cPar)
	case reflect.Float64:
		C.algoat_sort_float64((*C.double)(ptr), cLen, cPar)
	case reflect.Float32:
		C.algoat_sort_float32((*C.float)(ptr), cLen, cPar)
	}
}

// dispatchRadixSort calls the appropriate C-ABI radix sorting entry point.
func dispatchRadixSort(ptr unsafe.Pointer, n int, kind reflect.Kind) {
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
	case reflect.Int16:
		C.algoat_radix_sort_int16((*C.int16_t)(ptr), cLen)
	case reflect.Uint16:
		C.algoat_radix_sort_uint16((*C.uint16_t)(ptr), cLen)
	case reflect.Int8:
		C.algoat_radix_sort_int8((*C.int8_t)(ptr), cLen)
	case reflect.Uint8:
		C.algoat_radix_sort_uint8((*C.uint8_t)(ptr), cLen)
	}
}

// dispatchSearch calls the appropriate C-ABI branchless lower-bound search entry point.
func dispatchSearch[T Number](ptr unsafe.Pointer, n int, kind reflect.Kind, target T) int {
	cLen := C.size_t(n)

	switch kind {
	case reflect.Int64:
		return int(C.algoat_search_int64((*C.int64_t)(ptr), cLen, C.int64_t(int64(target))))
	case reflect.Uint64:
		return int(C.algoat_search_uint64((*C.uint64_t)(ptr), cLen, C.uint64_t(uint64(target))))
	case reflect.Int:
		if unsafe.Sizeof(int(0)) == 8 {
			return int(C.algoat_search_int64((*C.int64_t)(ptr), cLen, C.int64_t(int(target))))
		}
		return int(C.algoat_search_int32((*C.int32_t)(ptr), cLen, C.int32_t(int(target))))
	case reflect.Uint, reflect.Uintptr:
		if unsafe.Sizeof(uint(0)) == 8 {
			return int(C.algoat_search_uint64((*C.uint64_t)(ptr), cLen, C.uint64_t(uint(target))))
		}
		return int(C.algoat_search_uint32((*C.uint32_t)(ptr), cLen, C.uint32_t(uint(target))))
	case reflect.Int32:
		return int(C.algoat_search_int32((*C.int32_t)(ptr), cLen, C.int32_t(int32(target))))
	case reflect.Uint32:
		return int(C.algoat_search_uint32((*C.uint32_t)(ptr), cLen, C.uint32_t(uint32(target))))
	case reflect.Int16:
		return int(C.algoat_search_int16((*C.int16_t)(ptr), cLen, C.int16_t(int16(target))))
	case reflect.Uint16:
		return int(C.algoat_search_uint16((*C.uint16_t)(ptr), cLen, C.uint16_t(uint16(target))))
	case reflect.Int8:
		return int(C.algoat_search_int8((*C.int8_t)(ptr), cLen, C.int8_t(int8(target))))
	case reflect.Uint8:
		return int(C.algoat_search_uint8((*C.uint8_t)(ptr), cLen, C.uint8_t(uint8(target))))
	case reflect.Float64:
		return int(C.algoat_search_float64((*C.double)(ptr), cLen, C.double(float64(target))))
	case reflect.Float32:
		return int(C.algoat_search_float32((*C.float)(ptr), cLen, C.float(float32(target))))
	default:
		return 0
	}
}

// dispatchSearchBatch calls the appropriate C-ABI batch branchless search entry point.
func dispatchSearchBatch(ptr unsafe.Pointer, n int, targetPtr unsafe.Pointer, numTargets int, resultPtr unsafe.Pointer, kind reflect.Kind) {
	cLen := C.size_t(n)
	cNumTargets := C.size_t(numTargets)
	cResults := (*C.int64_t)(resultPtr)

	switch kind {
	case reflect.Int64:
		C.algoat_search_batch_int64((*C.int64_t)(ptr), cLen, (*C.int64_t)(targetPtr), cNumTargets, cResults)
	case reflect.Uint64:
		C.algoat_search_batch_uint64((*C.uint64_t)(ptr), cLen, (*C.uint64_t)(targetPtr), cNumTargets, cResults)
	case reflect.Int:
		if unsafe.Sizeof(int(0)) == 8 {
			C.algoat_search_batch_int64((*C.int64_t)(ptr), cLen, (*C.int64_t)(targetPtr), cNumTargets, cResults)
		} else {
			C.algoat_search_batch_int32((*C.int32_t)(ptr), cLen, (*C.int32_t)(targetPtr), cNumTargets, cResults)
		}
	case reflect.Uint, reflect.Uintptr:
		if unsafe.Sizeof(uint(0)) == 8 {
			C.algoat_search_batch_uint64((*C.uint64_t)(ptr), cLen, (*C.uint64_t)(targetPtr), cNumTargets, cResults)
		} else {
			C.algoat_search_batch_uint32((*C.uint32_t)(ptr), cLen, (*C.uint32_t)(targetPtr), cNumTargets, cResults)
		}
	case reflect.Int32:
		C.algoat_search_batch_int32((*C.int32_t)(ptr), cLen, (*C.int32_t)(targetPtr), cNumTargets, cResults)
	case reflect.Uint32:
		C.algoat_search_batch_uint32((*C.uint32_t)(ptr), cLen, (*C.uint32_t)(targetPtr), cNumTargets, cResults)
	case reflect.Int16:
		C.algoat_search_batch_int16((*C.int16_t)(ptr), cLen, (*C.int16_t)(targetPtr), cNumTargets, cResults)
	case reflect.Uint16:
		C.algoat_search_batch_uint16((*C.uint16_t)(ptr), cLen, (*C.uint16_t)(targetPtr), cNumTargets, cResults)
	case reflect.Int8:
		C.algoat_search_batch_int8((*C.int8_t)(ptr), cLen, (*C.int8_t)(targetPtr), cNumTargets, cResults)
	case reflect.Uint8:
		C.algoat_search_batch_uint8((*C.uint8_t)(ptr), cLen, (*C.uint8_t)(targetPtr), cNumTargets, cResults)
	case reflect.Float64:
		C.algoat_search_batch_float64((*C.double)(ptr), cLen, (*C.double)(targetPtr), cNumTargets, cResults)
	case reflect.Float32:
		C.algoat_search_batch_float32((*C.float)(ptr), cLen, (*C.float)(targetPtr), cNumTargets, cResults)
	}
}

// dispatchTopK calls the appropriate C-ABI top-k selection entry point.
func dispatchTopK(ptr unsafe.Pointer, n int, kind reflect.Kind, k int) {
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
	case reflect.Int16:
		C.algoat_topk_int16((*C.int16_t)(ptr), cLen, cK)
	case reflect.Uint16:
		C.algoat_topk_uint16((*C.uint16_t)(ptr), cLen, cK)
	case reflect.Int8:
		C.algoat_topk_int8((*C.int8_t)(ptr), cLen, cK)
	case reflect.Uint8:
		C.algoat_topk_uint8((*C.uint8_t)(ptr), cLen, cK)
	case reflect.Float64:
		C.algoat_topk_float64((*C.double)(ptr), cLen, cK)
	case reflect.Float32:
		C.algoat_topk_float32((*C.float)(ptr), cLen, cK)
	}
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

	var p runtime.Pinner
	ptr := unsafe.Pointer(unsafe.SliceData(slice))
	p.Pin(ptr)
	defer p.Unpin()

	kind := reflect.TypeOf(slice).Elem().Kind()
	dispatchSort(ptr, n, kind, 1)
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

	var p runtime.Pinner
	ptr := unsafe.Pointer(unsafe.SliceData(slice))
	p.Pin(ptr)
	defer p.Unpin()

	kind := reflect.TypeOf(slice).Elem().Kind()
	dispatchSort(ptr, n, kind, 0)
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

	var p runtime.Pinner
	ptr := unsafe.Pointer(unsafe.SliceData(slice))
	p.Pin(ptr)
	defer p.Unpin()

	kind := reflect.TypeOf(slice).Elem().Kind()
	dispatchRadixSort(ptr, n, kind)
}

// dispatchSearchFast dispatches to architecture-optimized native assembly routines (AMD64 CMOV).
// It evaluates a direct concrete type switch (0.3ns) for primitive numeric types,
// falling back to reflect.Kind only for custom user-defined domain types (e.g. type UserID int64).
func dispatchSearchFast[T Number](ptr unsafe.Pointer, n int, target T) int {
	switch t := any(target).(type) {
	case int64:
		return asm.BranchlessSearchInt64(ptr, n, t)
	case uint64:
		return asm.BranchlessSearchUint64(ptr, n, t)
	case int:
		if unsafe.Sizeof(int(0)) == 8 {
			return asm.BranchlessSearchInt64(ptr, n, int64(t))
		}
		return asm.BranchlessSearchInt32(ptr, n, int32(t))
	case uint:
		if unsafe.Sizeof(uint(0)) == 8 {
			return asm.BranchlessSearchUint64(ptr, n, uint64(t))
		}
		return asm.BranchlessSearchUint32(ptr, n, uint32(t))
	case uintptr:
		if unsafe.Sizeof(uintptr(0)) == 8 {
			return asm.BranchlessSearchUint64(ptr, n, uint64(t))
		}
		return asm.BranchlessSearchUint32(ptr, n, uint32(t))
	case int32:
		return asm.BranchlessSearchInt32(ptr, n, t)
	case uint32:
		return asm.BranchlessSearchUint32(ptr, n, t)
	case int16:
		return asm.BranchlessSearchInt16(ptr, n, t)
	case uint16:
		return asm.BranchlessSearchUint16(ptr, n, t)
	case int8:
		return asm.BranchlessSearchInt8(ptr, n, t)
	case uint8:
		return asm.BranchlessSearchUint8(ptr, n, t)
	case float64:
		return asm.BranchlessSearchFloat64(ptr, n, t)
	case float32:
		return asm.BranchlessSearchFloat32(ptr, n, t)
	default:
		kind := reflect.TypeOf(target).Kind()
		return dispatchSearchFastByKind(ptr, n, kind, target)
	}
}

func dispatchSearchFastByKind[T Number](ptr unsafe.Pointer, n int, kind reflect.Kind, target T) int {
	switch kind {
	case reflect.Int64:
		return asm.BranchlessSearchInt64(ptr, n, int64(target))
	case reflect.Uint64:
		return asm.BranchlessSearchUint64(ptr, n, uint64(target))
	case reflect.Int:
		if unsafe.Sizeof(int(0)) == 8 {
			return asm.BranchlessSearchInt64(ptr, n, int64(target))
		}
		return asm.BranchlessSearchInt32(ptr, n, int32(target))
	case reflect.Uint, reflect.Uintptr:
		if unsafe.Sizeof(uint(0)) == 8 {
			return asm.BranchlessSearchUint64(ptr, n, uint64(target))
		}
		return asm.BranchlessSearchUint32(ptr, n, uint32(target))
	case reflect.Int32:
		return asm.BranchlessSearchInt32(ptr, n, int32(target))
	case reflect.Uint32:
		return asm.BranchlessSearchUint32(ptr, n, uint32(target))
	case reflect.Int16:
		return asm.BranchlessSearchInt16(ptr, n, int16(target))
	case reflect.Uint16:
		return asm.BranchlessSearchUint16(ptr, n, uint16(target))
	case reflect.Int8:
		return asm.BranchlessSearchInt8(ptr, n, int8(target))
	case reflect.Uint8:
		return asm.BranchlessSearchUint8(ptr, n, uint8(target))
	case reflect.Float64:
		return asm.BranchlessSearchFloat64(ptr, n, float64(target))
	case reflect.Float32:
		return asm.BranchlessSearchFloat32(ptr, n, float32(target))
	default:
		return 0
	}
}

// Search performs a branchless lower-bound binary search for target in slice.
// It returns the index of the first element not less than target, or len(slice)
// if all elements are less than target. The slice must be sorted in ascending order.
// Utilizes architecture-optimized conditional moves (CMOV/CSEL) for zero branch mispredictions,
// executing 2x to 3x faster than Go's standard library slices.BinarySearch.
func Search[T Number](slice []T, target T) int {
	n := len(slice)
	if n == 0 {
		return 0
	}

	ptr := unsafe.Pointer(unsafe.SliceData(slice))
	return dispatchSearchFast(ptr, n, target)
}

// SearchBatch performs branchless lower-bound binary searches for a slice of targets
// against sorted slice. It returns a slice of indices where results[i] is the lower bound index for targets[i].
func SearchBatch[T Number](slice []T, targets []T) []int {
	numTargets := len(targets)
	if numTargets == 0 {
		return nil
	}
	out := make([]int, numTargets)
	n := len(slice)
	if n == 0 {
		return out
	}

	ptr := unsafe.Pointer(unsafe.SliceData(slice))
	for i, t := range targets {
		out[i] = dispatchSearchFast(ptr, n, t)
	}
	return out
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

	var p runtime.Pinner
	ptr := unsafe.Pointer(unsafe.SliceData(slice))
	p.Pin(ptr)
	defer p.Unpin()

	kind := reflect.TypeOf(slice).Elem().Kind()
	dispatchTopK(ptr, n, kind, k)

	return slice[:k]
}
