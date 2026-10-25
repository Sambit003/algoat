package algoat_test

import (
	"math/rand"
	"slices"
	"testing"
	"unsafe"

	"github.com/Sambit003/algoat/go/algoat"
)

type UserID int64
type Temperature float64
type Score int32

func TestSortCustomDerivedTypes(t *testing.T) {
	users := []UserID{100, 5, 200, 3, 50, 1}
	algoat.Sort(users)
	if !slices.IsSorted(users) {
		t.Fatalf("custom type UserID failed to sort: %v", users)
	}

	temps := []Temperature{98.6, -10.5, 32.0, 212.0, 0.0}
	algoat.Sort(temps)
	if !slices.IsSorted(temps) {
		t.Fatalf("custom type Temperature failed to sort: %v", temps)
	}

	scores := []Score{500, 100, 300, 200}
	algoat.Sort(scores)
	if !slices.IsSorted(scores) {
		t.Fatalf("custom type Score failed to sort: %v", scores)
	}
}

func TestSortRadix(t *testing.T) {
	n := 50_000
	data := make([]int64, n)
	rng := rand.New(rand.NewSource(12345))
	for i := range data {
		data[i] = rng.Int63()
	}

	algoat.SortRadix(data)
	if !slices.IsSorted(data) {
		t.Fatalf("SortRadix failed to sort int64 slice")
	}

	data32 := make([]int32, 1000)
	for i := range data32 {
		data32[i] = int32(rng.Intn(10000) - 5000)
	}
	algoat.SortRadix(data32)
	if !slices.IsSorted(data32) {
		t.Fatalf("SortRadix failed to sort int32 slice")
	}
}

func TestSortInt64(t *testing.T) {
	n := 100_000
	data := make([]int64, n)
	rng := rand.New(rand.NewSource(42))
	for i := range data {
		data[i] = rng.Int63()
	}

	origPtr := unsafe.SliceData(data)
	algoat.Sort(data)
	newPtr := unsafe.SliceData(data)

	if origPtr != newPtr {
		t.Fatalf("expected in-place sort with zero copy: original ptr %p != new ptr %p", origPtr, newPtr)
	}

	if !slices.IsSorted(data) {
		t.Fatalf("expected slice to be sorted in ascending order")
	}
}

func TestSortInt32(t *testing.T) {
	data := []int32{42, -10, 0, 999, -5000, 3, 7, 2, 99, -1}
	algoat.Sort(data)
	if !slices.IsSorted(data) {
		t.Fatalf("expected slice to be sorted: got %v", data)
	}
}

func TestSortInt(t *testing.T) {
	data := []int{100, -20, 50, 0, 10, -5, 30}
	algoat.Sort(data)
	if !slices.IsSorted(data) {
		t.Fatalf("expected slice to be sorted: got %v", data)
	}
}

func TestSortUint64(t *testing.T) {
	data := []uint64{1000, 20, 500, 0, 10, 5, 3000000000}
	algoat.Sort(data)
	if !slices.IsSorted(data) {
		t.Fatalf("expected slice to be sorted: got %v", data)
	}
}

func TestSortFloat64(t *testing.T) {
	data := []float64{3.14159, -0.5, 2.71828, 0.0, -100.25, 1e5, -1e5}
	algoat.Sort(data)
	if !slices.IsSorted(data) {
		t.Fatalf("expected float64 slice to be sorted: got %v", data)
	}
}

func TestSortFloat32(t *testing.T) {
	data := []float32{3.14, -0.5, 2.71, 0.0, -100.2, 50.0}
	algoat.Sort(data)
	if !slices.IsSorted(data) {
		t.Fatalf("expected float32 slice to be sorted: got %v", data)
	}
}

func TestSortPatterns(t *testing.T) {
	t.Run("AlreadySorted", func(t *testing.T) {
		data := []int64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
		algoat.Sort(data)
		if !slices.IsSorted(data) {
			t.Fatalf("failed on already sorted data")
		}
	})

	t.Run("ReverseSorted", func(t *testing.T) {
		data := []int64{10, 9, 8, 7, 6, 5, 4, 3, 2, 1}
		algoat.Sort(data)
		if !slices.IsSorted(data) {
			t.Fatalf("failed on reverse sorted data")
		}
	})

	t.Run("AllEqual", func(t *testing.T) {
		data := []int64{7, 7, 7, 7, 7, 7, 7}
		algoat.Sort(data)
		if !slices.IsSorted(data) {
			t.Fatalf("failed on all equal data")
		}
	})

	t.Run("PipeOrgan", func(t *testing.T) {
		data := []int64{1, 3, 5, 7, 9, 8, 6, 4, 2, 0}
		algoat.Sort(data)
		if !slices.IsSorted(data) {
			t.Fatalf("failed on pipe-organ data")
		}
	})
}

func TestSortEdgeCases(t *testing.T) {
	var nilSlice []int64
	algoat.Sort(nilSlice) // must not panic

	emptySlice := []int64{}
	algoat.Sort(emptySlice) // must not panic

	singleElement := []int64{42}
	algoat.Sort(singleElement)
	if singleElement[0] != 42 {
		t.Fatalf("single element modified: %v", singleElement)
	}

	twoElements := []int64{2, 1}
	algoat.Sort(twoElements)
	if twoElements[0] != 1 || twoElements[1] != 2 {
		t.Fatalf("two elements failed to sort: %v", twoElements)
	}
}

func TestSearch(t *testing.T) {
	data := []int64{10, 20, 30, 40, 50, 60, 70, 80}

	if idx := algoat.Search(data, int64(30)); idx != 2 {
		t.Fatalf("expected Search for 30 to return 2, got %d", idx)
	}

	if idx := algoat.Search(data, int64(35)); idx != 3 {
		t.Fatalf("expected Search for 35 to return 3, got %d", idx)
	}

	if idx := algoat.Search(data, int64(5)); idx != 0 {
		t.Fatalf("expected Search for 5 to return 0, got %d", idx)
	}

	if idx := algoat.Search(data, int64(100)); idx != 8 {
		t.Fatalf("expected Search for 100 to return 8, got %d", idx)
	}

	// Floats
	floats := []float64{-10.5, 0.0, 1.25, 3.5, 100.0}
	if idx := algoat.Search(floats, 1.25); idx != 2 {
		t.Fatalf("expected Search for 1.25 to return 2, got %d", idx)
	}
	if idx := algoat.Search(floats, -20.0); idx != 0 {
		t.Fatalf("expected Search for -20.0 to return 0, got %d", idx)
	}
	if idx := algoat.Search(floats, 200.0); idx != 5 {
		t.Fatalf("expected Search for 200.0 to return 5, got %d", idx)
	}

	// Empty
	var empty []int64
	if idx := algoat.Search(empty, int64(10)); idx != 0 {
		t.Fatalf("expected Search in empty to return 0, got %d", idx)
	}
}

func TestTopK(t *testing.T) {
	data := []int64{90, 10, 50, 20, 80, 30, 70, 40, 60}
	origPtr := unsafe.SliceData(data)

	top3 := algoat.TopK(data, 3)

	if len(top3) != 3 {
		t.Fatalf("expected top3 length 3, got %d", len(top3))
	}

	if unsafe.SliceData(top3) != origPtr {
		t.Fatalf("expected TopK to return zero-copy subslice")
	}

	expected := []int64{10, 20, 30}
	for i := range expected {
		if top3[i] != expected[i] {
			t.Fatalf("at index %d: expected %d, got %d", i, expected[i], top3[i])
		}
	}

	// Edge cases
	emptyTop := algoat.TopK(data, 0)
	if len(emptyTop) != 0 {
		t.Fatalf("expected 0 elements for k=0, got %d", len(emptyTop))
	}

	allTop := algoat.TopK(data, 20)
	if len(allTop) != len(data) {
		t.Fatalf("expected %d elements for k >= len, got %d", len(data), len(allTop))
	}
	if !slices.IsSorted(allTop) {
		t.Fatalf("expected allTop to be fully sorted")
	}
}

func TestZeroAllocationsAndNoLeak(t *testing.T) {
	// Preallocate buffer to test that Sort, Search, and TopK perform zero heap allocations
	buf := make([]int64, 1000)
	for i := range buf {
		buf[i] = int64(1000 - i)
	}

	allocsSort := testing.AllocsPerRun(100, func() {
		algoat.Sort(buf)
	})
	if allocsSort != 0 {
		t.Fatalf("expected 0 heap allocations for Sort, got %f", allocsSort)
	}

	allocsSearch := testing.AllocsPerRun(100, func() {
		_ = algoat.Search(buf, int64(500))
	})
	if allocsSearch != 0 {
		t.Fatalf("expected 0 heap allocations for Search, got %f", allocsSearch)
	}

	allocsTopK := testing.AllocsPerRun(100, func() {
		_ = algoat.TopK(buf, 10)
	})
	if allocsTopK != 0 {
		t.Fatalf("expected 0 heap allocations for TopK, got %f", allocsTopK)
	}
}
