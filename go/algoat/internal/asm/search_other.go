//go:build !amd64

package asm

import (
	"cmp"
	"unsafe"
)

func branchlessSearchGeneric[T cmp.Ordered](slice []T, target T) int {
	n := len(slice)
	base := 0
	for n > 1 {
		half := n >> 1
		if slice[base+half-1] < target {
			base += half
		}
		n -= half
	}
	if n > 0 && slice[base] < target {
		base += n
	}
	return base
}

func BranchlessSearchInt64(data unsafe.Pointer, len int, target int64) int {
	return branchlessSearchGeneric(unsafe.Slice((*int64)(data), len), target)
}

func BranchlessSearchUint64(data unsafe.Pointer, len int, target uint64) int {
	return branchlessSearchGeneric(unsafe.Slice((*uint64)(data), len), target)
}

func BranchlessSearchInt32(data unsafe.Pointer, len int, target int32) int {
	return branchlessSearchGeneric(unsafe.Slice((*int32)(data), len), target)
}

func BranchlessSearchUint32(data unsafe.Pointer, len int, target uint32) int {
	return branchlessSearchGeneric(unsafe.Slice((*uint32)(data), len), target)
}

func BranchlessSearchInt16(data unsafe.Pointer, len int, target int16) int {
	return branchlessSearchGeneric(unsafe.Slice((*int16)(data), len), target)
}

func BranchlessSearchUint16(data unsafe.Pointer, len int, target uint16) int {
	return branchlessSearchGeneric(unsafe.Slice((*uint16)(data), len), target)
}

func BranchlessSearchInt8(data unsafe.Pointer, len int, target int8) int {
	return branchlessSearchGeneric(unsafe.Slice((*int8)(data), len), target)
}

func BranchlessSearchUint8(data unsafe.Pointer, len int, target uint8) int {
	return branchlessSearchGeneric(unsafe.Slice((*uint8)(data), len), target)
}

func BranchlessSearchFloat64(data unsafe.Pointer, len int, target float64) int {
	return branchlessSearchGeneric(unsafe.Slice((*float64)(data), len), target)
}

func BranchlessSearchFloat32(data unsafe.Pointer, len int, target float32) int {
	return branchlessSearchGeneric(unsafe.Slice((*float32)(data), len), target)
}
