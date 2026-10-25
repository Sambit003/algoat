package algoat_test

import (
	"bytes"
	"encoding/binary"
	"slices"
	"testing"

	"github.com/Sambit003/algoat/go/algoat"
)

func FuzzSort(f *testing.F) {
	// Seed corpus with various patterns
	seeds := [][]byte{
		{},
		{1, 2, 3, 4, 5, 6, 7, 8},
		{8, 7, 6, 5, 4, 3, 2, 1},
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{255, 128, 64, 32, 16, 8, 4, 2, 1, 0, 100, 200, 150, 75, 25, 5},
	}
	for _, s := range seeds {
		f.Add(s)
	}

	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) < 8 {
			return
		}
		// Convert bytes to []int64
		numInts := len(data) / 8
		ints := make([]int64, numInts)
		reader := bytes.NewReader(data[:numInts*8])
		if err := binary.Read(reader, binary.LittleEndian, &ints); err != nil {
			return
		}

		oracle := slices.Clone(ints)
		slices.Sort(oracle)

		algoat.Sort(ints)

		if !slices.IsSorted(ints) {
			t.Fatalf("slice not sorted")
		}

		if !slices.Equal(ints, oracle) {
			t.Fatalf("algoat.Sort does not match slices.Sort oracle: got %v, want %v", ints, oracle)
		}
	})
}

func FuzzSearch(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, int64(5))
	f.Add([]byte{10, 20, 30, 40, 50, 60, 70, 80}, int64(25))
	f.Add([]byte{}, int64(0))

	f.Fuzz(func(t *testing.T, data []byte, target int64) {
		numInts := len(data) / 8
		if numInts == 0 {
			return
		}
		ints := make([]int64, numInts)
		reader := bytes.NewReader(data[:numInts*8])
		if err := binary.Read(reader, binary.LittleEndian, &ints); err != nil {
			return
		}

		// Pre-sort for binary search
		slices.Sort(ints)

		gotIdx := algoat.Search(ints, target)
		oracleIdx, _ := slices.BinarySearch(ints, target)

		if gotIdx != oracleIdx {
			t.Fatalf("algoat.Search(%v, %d) = %d; oracle BinarySearch = %d", ints, target, gotIdx, oracleIdx)
		}
	})
}

func FuzzTopK(f *testing.F) {
	f.Add([]byte{5, 2, 8, 1, 9, 3, 7, 4, 6, 0}, 3)
	f.Add([]byte{10, 20, 30, 40, 50}, 0)
	f.Add([]byte{10, 20, 30, 40, 50}, 10)

	f.Fuzz(func(t *testing.T, data []byte, k int) {
		numInts := len(data) / 8
		if numInts == 0 {
			return
		}
		ints := make([]int64, numInts)
		reader := bytes.NewReader(data[:numInts*8])
		if err := binary.Read(reader, binary.LittleEndian, &ints); err != nil {
			return
		}

		top := algoat.TopK(ints, k)

		expectedK := k
		if expectedK < 0 {
			expectedK = 0
		} else if expectedK > len(ints) {
			expectedK = len(ints)
		}

		if len(top) != expectedK {
			t.Fatalf("expected len(top) == %d, got %d", expectedK, len(top))
		}

		if !slices.IsSorted(top) {
			t.Fatalf("top-k prefix is not sorted")
		}

		if expectedK > 0 && expectedK < len(ints) {
			maxInTop := top[len(top)-1]
			for i := expectedK; i < len(ints); i++ {
				if ints[i] < maxInTop {
					t.Fatalf("element outside top-k at %d (%d) is less than max in top (%d)", i, ints[i], maxInTop)
				}
			}
		}
	})
}
