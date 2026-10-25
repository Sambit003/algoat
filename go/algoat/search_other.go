//go:build !amd64

package algoat

import (
	"reflect"
	"runtime"
	"unsafe"
)

// searchSingle routes to Algoat's C-ABI C++ branchless binary search engine via Cgo on non-AMD64 architectures.
func searchSingle[T Number](ptr unsafe.Pointer, n int, target T) int {
	var p runtime.Pinner
	p.Pin(ptr)
	defer p.Unpin()

	kind := reflect.TypeOf(target).Kind()
	return dispatchSearch(ptr, n, kind, target)
}

// searchBatch routes to Algoat's C-ABI C++ batch branchless search engine via Cgo on non-AMD64 architectures.
func searchBatch[T Number](slice []T, targets []T) []int {
	numTargets := len(targets)
	if numTargets == 0 {
		return nil
	}
	n := len(slice)
	if n == 0 {
		return make([]int, numTargets)
	}

	var p runtime.Pinner
	ptr := unsafe.Pointer(unsafe.SliceData(slice))
	p.Pin(ptr)
	defer p.Unpin()

	targetPtr := unsafe.Pointer(unsafe.SliceData(targets))
	p.Pin(targetPtr)

	kind := reflect.TypeOf(slice).Elem().Kind()

	if unsafe.Sizeof(int(0)) == 8 {
		out := make([]int, numTargets)
		resultPtr := unsafe.Pointer(unsafe.SliceData(out))
		p.Pin(resultPtr)
		dispatchSearchBatch(ptr, n, targetPtr, numTargets, resultPtr, kind)
		return out
	}

	raw := make([]int64, numTargets)
	resultPtr := unsafe.Pointer(unsafe.SliceData(raw))
	p.Pin(resultPtr)
	dispatchSearchBatch(ptr, n, targetPtr, numTargets, resultPtr, kind)

	out := make([]int, numTargets)
	for i, r := range raw {
		out[i] = int(r)
	}
	return out
}
