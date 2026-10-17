#include "algoat/strings/multikey_quicksort.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

TEST(MultiKeyQuicksortTest, EmptyAndSingleElement) {
    std::vector<std::string> empty_vec;
    algoat::strings::multikey_quicksort(std::span<std::string>{empty_vec});
    EXPECT_TRUE(empty_vec.empty());

    std::vector<std::string> single_vec = {"hello"};
    algoat::strings::multikey_quicksort(std::span<std::string>{single_vec});
    EXPECT_EQ(single_vec.size(), 1);
    EXPECT_EQ(single_vec[0], "hello");

    std::vector<std::string_view> single_sv = {"world"};
    algoat::strings::multikey_quicksort(std::span<std::string_view>{single_sv});
    EXPECT_EQ(single_sv[0], "world");
}

TEST(MultiKeyQuicksortTest, BasicStringSort) {
    std::vector<std::string> words = {"banana", "apple",      "cherry",  "date",
                                      "fig",    "elderberry", "apricot", "blueberry"};
    auto expected = words;
    std::sort(expected.begin(), expected.end());

    algoat::strings::multikey_quicksort(std::span<std::string>{words});
    EXPECT_EQ(words, expected);
}

TEST(MultiKeyQuicksortTest, StringViewSpan) {
    std::vector<std::string_view> views = {"https://example.com/b",   "https://example.com/a",
                                           "http://example.com",      "https://example.org",
                                           "ftp://files.example.com", "https://example.com/a/b"};
    auto expected = views;
    std::sort(expected.begin(), expected.end());

    algoat::strings::multikey_quicksort(std::span<std::string_view>{views});
    EXPECT_EQ(views, expected);
}

TEST(MultiKeyQuicksortTest, IdenticalAndPrefixStrings) {
    std::vector<std::string> data = {"abc", "ab", "a", "abcd", "abcde", "ab", "a", "abc", ""};
    auto expected = data;
    std::sort(expected.begin(), expected.end());

    algoat::strings::multikey_quicksort(std::span<std::string>{data});
    EXPECT_EQ(data, expected);
}

TEST(MultiKeyQuicksortTest, EmbeddedNullAndNonAscii) {
    std::vector<std::string> data = {
        std::string("a\0b", 3),
        std::string("a", 1),
        std::string("a\0a", 3),
        std::string("a\0c", 3),
        "zoo",
        "\xC3\xA9", // é
        "\xC3\xA0", // à
        "\xFF\xFE"  // High bytes
    };
    auto expected = data;
    std::sort(expected.begin(), expected.end());

    algoat::strings::multikey_quicksort(std::span<std::string>{data});
    EXPECT_EQ(data, expected);
}

TEST(MultiKeyQuicksortTest, LargeRandomStrings) {
    std::mt19937_64 rng(42);
    std::vector<std::string> data;
    data.reserve(5000);
    const std::string charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

    for (int i = 0; i < 5000; ++i) {
        int len = rng() % 30;
        std::string s;
        s.reserve(len);
        for (int j = 0; j < len; ++j) {
            s.push_back(charset[rng() % charset.size()]);
        }
        data.push_back(std::move(s));
    }

    auto expected = data;
    std::sort(expected.begin(), expected.end());

    algoat::strings::multikey_quicksort(std::span<std::string>{data});
    EXPECT_EQ(data, expected);
}

TEST(MultiKeyQuicksortTest, RangeOverload) {
    std::vector<std::string> words = {"zebra", "yak", "xenon", "walrus"};
    auto expected = words;
    std::sort(expected.begin(), expected.end());

    algoat::strings::multikey_quicksort(words);
    EXPECT_EQ(words, expected);
}

TEST(MultiKeyQuicksortTest, CArrayRange) {
    std::string arr[4] = {"zebra", "yak", "xenon", "walrus"};
    algoat::strings::multikey_quicksort(arr);
    EXPECT_EQ(arr[0], "walrus");
    EXPECT_EQ(arr[1], "xenon");
    EXPECT_EQ(arr[2], "yak");
    EXPECT_EQ(arr[3], "zebra");
}

TEST(MultiKeyQuicksortTest, DeepIdenticalPrefixStackSafety) {
    // 100 strings with 50,000 identical prefix bytes - must not exhaust call stack
    const std::string prefix(50000, 'x');
    std::vector<std::string> data;
    data.reserve(100);
    for (int i = 0; i < 100; ++i) {
        data.push_back(prefix + std::to_string(99 - i));
    }
    auto expected = data;
    std::sort(expected.begin(), expected.end());

    algoat::strings::multikey_quicksort(std::span<std::string>{data});
    EXPECT_EQ(data, expected);
}

} // namespace

#ifndef __has_feature
#define __has_feature(x) 0
#endif

#if !defined(__SANITIZE_ADDRESS__) && (!defined(__has_feature) || !__has_feature(address_sanitizer))
static std::atomic<std::size_t> g_active_allocations{0};
static std::atomic<bool> g_track_allocations{false};

void* operator new(std::size_t size) {
    if (g_track_allocations.load(std::memory_order_relaxed)) {
        g_active_allocations.fetch_add(size, std::memory_order_relaxed);
    }
    void* p = std::malloc(size);
    if (!p) {
        throw std::bad_alloc();
    }
    return p;
}

void operator delete(void* p) noexcept {
    std::free(p);
}

void operator delete(void* p, std::size_t) noexcept {
    std::free(p);
}

TEST(MultiKeyQuicksortTest, ZeroHeapAllocations) {
    std::vector<std::string_view> data = {
        "apple", "banana", "cherry", "date",      "elderberry", "fig",   "grape",  "honeydew",
        "kiwi",  "lemon",  "mango",  "nectarine", "orange",     "peach", "quince", "raspberry"};

    g_track_allocations.store(true, std::memory_order_relaxed);
    g_active_allocations.store(0, std::memory_order_relaxed);

    algoat::strings::multikey_quicksort(std::span<std::string_view>{data});

    std::size_t allocs = g_active_allocations.load(std::memory_order_relaxed);
    g_track_allocations.store(false, std::memory_order_relaxed);

    EXPECT_EQ(allocs, 0);
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}
#endif
