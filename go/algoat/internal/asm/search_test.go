package asm

import (
	"slices"
	"testing"
	"unsafe"
)

func TestAllAsmTypes(t *testing.T) {
	// int64
	d64 := []int64{-50, -20, 0, 10, 30, 70, 100}
	for _, target := range []int64{-100, -50, -30, 0, 5, 10, 50, 70, 100, 150} {
		exp, _ := slices.BinarySearch(d64, target)
		act := BranchlessSearchInt64(unsafe.Pointer(unsafe.SliceData(d64)), len(d64), target)
		if exp != act {
			t.Fatalf("int64 target %d: exp %d, act %d", target, exp, act)
		}
	}

	// uint64
	u64 := []uint64{0, 10, 30, 70, 100, 500, 1000}
	for _, target := range []uint64{0, 5, 10, 50, 70, 100, 150, 2000} {
		exp, _ := slices.BinarySearch(u64, target)
		act := BranchlessSearchUint64(unsafe.Pointer(unsafe.SliceData(u64)), len(u64), target)
		if exp != act {
			t.Fatalf("uint64 target %d: exp %d, act %d", target, exp, act)
		}
	}

	// int32
	d32 := []int32{-50, -20, 0, 10, 30, 70, 100}
	for _, target := range []int32{-100, -50, -30, 0, 5, 10, 50, 70, 100, 150} {
		exp, _ := slices.BinarySearch(d32, target)
		act := BranchlessSearchInt32(unsafe.Pointer(unsafe.SliceData(d32)), len(d32), target)
		if exp != act {
			t.Fatalf("int32 target %d: exp %d, act %d", target, exp, act)
		}
	}

	// uint32
	u32 := []uint32{0, 10, 30, 70, 100, 500, 1000}
	for _, target := range []uint32{0, 5, 10, 50, 70, 100, 150, 2000} {
		exp, _ := slices.BinarySearch(u32, target)
		act := BranchlessSearchUint32(unsafe.Pointer(unsafe.SliceData(u32)), len(u32), target)
		if exp != act {
			t.Fatalf("uint32 target %d: exp %d, act %d", target, exp, act)
		}
	}

	// int16
	d16 := []int16{-500, -20, 0, 10, 30, 700, 1000}
	for _, target := range []int16{-1000, -500, -30, 0, 5, 10, 50, 700, 1000, 1500} {
		exp, _ := slices.BinarySearch(d16, target)
		act := BranchlessSearchInt16(unsafe.Pointer(unsafe.SliceData(d16)), len(d16), target)
		if exp != act {
			t.Fatalf("int16 target %d: exp %d, act %d", target, exp, act)
		}
	}

	// uint16
	u16 := []uint16{0, 10, 30, 700, 1000, 5000, 10000}
	for _, target := range []uint16{0, 5, 10, 50, 700, 1000, 1500, 20000} {
		exp, _ := slices.BinarySearch(u16, target)
		act := BranchlessSearchUint16(unsafe.Pointer(unsafe.SliceData(u16)), len(u16), target)
		if exp != act {
			t.Fatalf("uint16 target %d: exp %d, act %d", target, exp, act)
		}
	}

	// int8
	d8 := []int8{-50, -20, 0, 10, 30, 70, 100}
	for _, target := range []int8{-100, -50, -30, 0, 5, 10, 50, 70, 100, 120} {
		exp, _ := slices.BinarySearch(d8, target)
		act := BranchlessSearchInt8(unsafe.Pointer(unsafe.SliceData(d8)), len(d8), target)
		if exp != act {
			t.Fatalf("int8 target %d: exp %d, act %d", target, exp, act)
		}
	}

	// uint8
	u8 := []uint8{0, 10, 30, 70, 100, 150, 200}
	for _, target := range []uint8{0, 5, 10, 50, 70, 100, 120, 250} {
		exp, _ := slices.BinarySearch(u8, target)
		act := BranchlessSearchUint8(unsafe.Pointer(unsafe.SliceData(u8)), len(u8), target)
		if exp != act {
			t.Fatalf("uint8 target %d: exp %d, act %d", target, exp, act)
		}
	}

	// float64
	f64 := []float64{-10.5, 0.0, 1.25, 3.5, 100.0}
	for _, target := range []float64{-20.0, -10.5, -5.0, 0.0, 1.25, 2.0, 3.5, 50.0, 100.0, 200.0} {
		exp, _ := slices.BinarySearch(f64, target)
		act := BranchlessSearchFloat64(unsafe.Pointer(unsafe.SliceData(f64)), len(f64), target)
		if exp != act {
			t.Fatalf("float64 target %f: exp %d, act %d", target, exp, act)
		}
	}

	// float32
	f32 := []float32{-10.5, 0.0, 1.25, 3.5, 100.0}
	for _, target := range []float32{-20.0, -10.5, -5.0, 0.0, 1.25, 2.0, 3.5, 50.0, 100.0, 200.0} {
		exp, _ := slices.BinarySearch(f32, target)
		act := BranchlessSearchFloat32(unsafe.Pointer(unsafe.SliceData(f32)), len(f32), target)
		if exp != act {
			t.Fatalf("float32 target %f: exp %d, act %d", target, exp, act)
		}
	}
}
