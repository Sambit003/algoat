//go:build amd64

package asm

import "unsafe"

//go:noescape
func BranchlessSearchInt64(data unsafe.Pointer, len int, target int64) int

//go:noescape
func BranchlessSearchUint64(data unsafe.Pointer, len int, target uint64) int

//go:noescape
func BranchlessSearchInt32(data unsafe.Pointer, len int, target int32) int

//go:noescape
func BranchlessSearchUint32(data unsafe.Pointer, len int, target uint32) int

//go:noescape
func BranchlessSearchInt16(data unsafe.Pointer, len int, target int16) int

//go:noescape
func BranchlessSearchUint16(data unsafe.Pointer, len int, target uint16) int

//go:noescape
func BranchlessSearchInt8(data unsafe.Pointer, len int, target int8) int

//go:noescape
func BranchlessSearchUint8(data unsafe.Pointer, len int, target uint8) int

//go:noescape
func BranchlessSearchFloat64(data unsafe.Pointer, len int, target float64) int

//go:noescape
func BranchlessSearchFloat32(data unsafe.Pointer, len int, target float32) int
