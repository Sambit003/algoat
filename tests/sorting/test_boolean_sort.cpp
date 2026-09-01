#include "algoat/sorting/boolean_sort.hpp"

#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <span>
#include <vector>

template <typename T>
void verify_boolean_sort(std::vector<T> data) {
    auto original = data;

    algoat::sorting::sort_boolean(std::span{data});

    bool found_nonzero = false;

    for (const T value : data) {
        if (value == 0) {
            EXPECT_FALSE(found_nonzero);
        } else {
            found_nonzero = true;
        }
    }

    std::sort(original.begin(), original.end());
    std::sort(data.begin(), data.end());

    EXPECT_EQ(data, original);
}

TEST(IntegralInputTest, HandlesMixedValues) {
    std::vector<std::uint8_t> data = {5, 0, 3, 2, 0, 100, 7};

    verify_boolean_sort(data);
}

TEST(IntegralInputTest, HandlesAllZeroValues) {
    std::vector<std::uint8_t> data = {0, 0, 0, 0};

    verify_boolean_sort(data);
}

TEST(IntegralInputTest, HandlesAllNonzeroValues) {
    std::vector<std::uint8_t> data = {5, 2, 3, 100};

    verify_boolean_sort(data);
}

TEST(IntegralInputTest, HandlesSingleElement) {
    std::vector<std::uint8_t> data = {7};

    verify_boolean_sort(data);
}

TEST(IntegralInputTest, PreservesDuplicates) {
    std::vector<std::uint8_t> data = {2, 0, 5, 0, 2, 3, 5};

    verify_boolean_sort(data);
}

TEST(BooleanSortTest, SortsBooleanValues) {
    bool data[] = {true, false, true, false};
    std::span<bool> view{data};

    algoat::sorting::sort_boolean(view);

    EXPECT_FALSE(data[0]);
    EXPECT_FALSE(data[1]);
    EXPECT_TRUE(data[2]);
    EXPECT_TRUE(data[3]);
}

TEST(BooleanSortTest, HandlesAllFalseValues) {
    bool data[] = {false, false, false, false};
    std::span<bool> view{data};

    algoat::sorting::sort_boolean(view);

    for (bool value : data) {
        EXPECT_FALSE(value);
    }
}

TEST(BooleanSortTest, HandlesAllTrueValues) {
    bool data[] = {true, true, true, true};
    std::span<bool> view{data};

    algoat::sorting::sort_boolean(view);

    for (bool value : data) {
        EXPECT_TRUE(value);
    }
}

TEST(BooleanSortTest, HandlesSingleElement) {
    bool data[] = {true};
    std::span<bool> view{data};

    algoat::sorting::sort_boolean(view);

    EXPECT_TRUE(data[0]);
}

TEST(BooleanSortTest, HandlesEmptyInput) {
    std::span<bool> view{};

    EXPECT_NO_THROW(
        algoat::sorting::sort_boolean(view)
    );
}

TEST(BooleanSortTest, HandlesLargeAllFalseInput) {
    auto values = std::make_unique<bool[]>(100'000);
    std::fill_n(values.get(), 100'000, false);
    std::span<bool> view{values.get(), 100'000};
    algoat::sorting::sort_boolean(view);

    EXPECT_TRUE(std::all_of(values.get(), values.get() + 100'000, [](bool value) {
        return !value;
    }));
}

TEST(BooleanSortTest, HandlesLargeAllTrueInput) {
    auto values = std::make_unique<bool[]>(100'000);
    std::fill_n(values.get(), 100'000, true);
    std::span<bool> view{values.get(), 100'000};
    algoat::sorting::sort_boolean(view);

    EXPECT_TRUE(std::all_of(values.get(), values.get() + 100'000, [](bool value) {
        return value;
    }));
}

TEST(BooleanSortTest, HandlesLargeAlternatingInput) {
    auto values = std::make_unique<bool[]>(100'000);
    for (std::size_t i = 0; i < 100'000; ++i) {
        values[i] = (i % 2) != 0;
    }

    algoat::sorting::sort_boolean(std::span<bool>{values.get(), 100'000});

    EXPECT_TRUE(std::all_of(values.get(), values.get() + 50'000,
                            [](bool value) { return !value; }));
    EXPECT_TRUE(std::all_of(values.get() + 50'000, values.get() + 100'000,
                            [](bool value) { return value; }));
}

TEST(BooleanSortTest, HandlesBoundarySizes) {
    for (std::size_t size = 0; size <= 3; ++size) {
        auto values = std::make_unique<bool[]>(size == 0 ? 1 : size);
        for (std::size_t i = 0; i < size; ++i) {
            values[i] = ((size - i) % 2) != 0;
        }

        algoat::sorting::sort_boolean(std::span<bool>{values.get(), size});

        EXPECT_TRUE(std::is_sorted(values.get(), values.get() + size));
    }
}




