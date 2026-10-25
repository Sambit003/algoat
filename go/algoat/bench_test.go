package algoat_test

import (
	"fmt"
	"math/rand"
	"slices"
	"testing"

	"github.com/Sambit003/algoat/go/algoat"
)

func generateRandomInt64(n int, seed int64) []int64 {
	rng := rand.New(rand.NewSource(seed))
	data := make([]int64, n)
	for i := range data {
		data[i] = rng.Int63()
	}
	return data
}

func BenchmarkSort(b *testing.B) {
	sizes := []int{100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000}

	for _, n := range sizes {
		baseData := generateRandomInt64(n, 12345)

		b.Run(fmt.Sprintf("Algoat_Parallel/N=%d", n), func(b *testing.B) {
			buf := make([]int64, n)
			b.ResetTimer()
			b.SetBytes(int64(n * 8))
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				copy(buf, baseData)
				b.StartTimer()
				algoat.Sort(buf)
			}
		})

		b.Run(fmt.Sprintf("Algoat_Radix/N=%d", n), func(b *testing.B) {
			buf := make([]int64, n)
			b.ResetTimer()
			b.SetBytes(int64(n * 8))
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				copy(buf, baseData)
				b.StartTimer()
				algoat.SortRadix(buf)
			}
		})

		b.Run(fmt.Sprintf("Algoat_Sequential/N=%d", n), func(b *testing.B) {
			buf := make([]int64, n)
			b.ResetTimer()
			b.SetBytes(int64(n * 8))
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				copy(buf, baseData)
				b.StartTimer()
				algoat.SortSequential(buf)
			}
		})

		b.Run(fmt.Sprintf("Go_slices_Sort/N=%d", n), func(b *testing.B) {
			buf := make([]int64, n)
			b.ResetTimer()
			b.SetBytes(int64(n * 8))
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				copy(buf, baseData)
				b.StartTimer()
				slices.Sort(buf)
			}
		})
	}
}

func BenchmarkSearch(b *testing.B) {
	sizes := []int{1_000, 10_000, 100_000, 1_000_000}

	for _, n := range sizes {
		data := generateRandomInt64(n, 54321)
		slices.Sort(data)

		rng := rand.New(rand.NewSource(12345))
		targets := make([]int64, 1024)
		for i := range targets {
			targets[i] = data[rng.Intn(n)]
		}

		b.Run(fmt.Sprintf("Algoat_Branchless_Single/N=%d", n), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = algoat.Search(data, targets[i&1023])
			}
		})

		b.Run(fmt.Sprintf("Go_BinarySearch_Single/N=%d", n), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = slices.BinarySearch(data, targets[i&1023])
			}
		})
	}
}

func BenchmarkSearchBatch(b *testing.B) {
	sizes := []int{10_000, 100_000, 1_000_000}
	numQueries := 10_000

	for _, n := range sizes {
		data := generateRandomInt64(n, 54321)
		slices.Sort(data)

		rng := rand.New(rand.NewSource(99999))
		queries := make([]int64, numQueries)
		for i := range queries {
			queries[i] = data[rng.Intn(n)]
		}

		b.Run(fmt.Sprintf("Algoat_SearchBatch_10k/N=%d", n), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = algoat.SearchBatch(data, queries)
			}
		})

		b.Run(fmt.Sprintf("Go_slices_BinarySearch_10k/N=%d", n), func(b *testing.B) {
			out := make([]int, numQueries)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for j, q := range queries {
					idx, _ := slices.BinarySearch(data, q)
					out[j] = idx
				}
			}
		})
	}
}

func BenchmarkTopK(b *testing.B) {
	n := 100_000
	baseData := generateRandomInt64(n, 9999)
	k := 50

	b.Run("Algoat_TopK_50_of_100k", func(b *testing.B) {
		buf := make([]int64, n)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			b.StopTimer()
			copy(buf, baseData)
			b.StartTimer()
			_ = algoat.TopK(buf, k)
		}
	})

	b.Run("Go_FullSort_Top50", func(b *testing.B) {
		buf := make([]int64, n)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			b.StopTimer()
			copy(buf, baseData)
			b.StartTimer()
			slices.Sort(buf)
			_ = buf[:k]
		}
	})
}
