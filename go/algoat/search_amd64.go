//go:build amd64

package algoat

import (
	"reflect"
	"unsafe"

	"github.com/Sambit003/algoat/go/algoat/internal/asm"
)

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

func searchSingle[T Number](ptr unsafe.Pointer, n int, target T) int {
	return dispatchSearchFast(ptr, n, target)
}

func searchBatch[T Number](slice []T, targets []T) []int {
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
