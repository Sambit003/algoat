#include "algoat/sorting/bitonicsort.hpp"
#include "algoat/sorting/bubblesort.hpp"
#include "algoat/sorting/combsort.hpp"
#include "algoat/sorting/cyclesort.hpp"
#include "algoat/sorting/gnomesort.hpp"
#include "algoat/sorting/selectionsort.hpp"
#include "algoat/sorting/shellsort.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <utility>
#include <vector>

using namespace algoat::sorting;

template <typename Algo> class ClassicalSortTest : public ::testing::Test {
protected:
    Algo algo;
    void verify_sort(std::vector<int>& data) {
        auto expected = data;
        std::sort(expected.begin(), expected.end());
        algo.sort(std::span{data});
        EXPECT_EQ(data, expected);
    }
};

using SortAlgos =
    ::testing::Types<SelectionSort, BubbleSort, ShellSort, CombSort, GnomeSort, CycleSort>;
TYPED_TEST_SUITE(ClassicalSortTest, SortAlgos);

TYPED_TEST(ClassicalSortTest, RandomData) {
    std::vector<int> data = {5, 3, 8, 1, 9, 2, 7};
    this->verify_sort(data);
}

TYPED_TEST(ClassicalSortTest, EmptyInput) {
    std::vector<int> data;
    this->verify_sort(data);
}

TYPED_TEST(ClassicalSortTest, SingleElement) {
    std::vector<int> data = {42};
    this->verify_sort(data);
}

TYPED_TEST(ClassicalSortTest, AllDuplicates) {
    std::vector<int> data = {7, 7, 7, 7, 7};
    this->verify_sort(data);
}

TYPED_TEST(ClassicalSortTest, AlreadySorted) {
    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7};
    this->verify_sort(data);
}

TYPED_TEST(ClassicalSortTest, ReverseSorted) {
    std::vector<int> data = {7, 6, 5, 4, 3, 2, 1};
    this->verify_sort(data);
}

TYPED_TEST(ClassicalSortTest, LargeRandom) {
    std::vector<int> data(1000);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(1, 10000);
    for (int& x : data) {
        x = dist(gen);
    }
    this->verify_sort(data);
}

class BitonicSortTest : public ::testing::Test {
protected:
    BitonicSort algo;
    void verify_sort(std::vector<int>& data) {
        auto expected = data;
        std::sort(expected.begin(), expected.end());
        algo.sort(std::span{data});
        EXPECT_EQ(data, expected);
    }
};

TEST_F(BitonicSortTest, RandomDataPowerOfTwo) {
    std::vector<int> data = {5, 3, 8, 1, 9, 2, 7, 4}; // Size 8
    verify_sort(data);
}

TEST_F(BitonicSortTest, EmptyInput) {
    std::vector<int> data;
    verify_sort(data);
}

TEST_F(BitonicSortTest, LargeRandomPowerOfTwo) {
    std::vector<int> data(1024);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(1, 10000);
    for (int& x : data) {
        x = dist(gen);
    }
    verify_sort(data);
}

TEST_F(BitonicSortTest, ThrowsOnNonPowerOfTwo) {
    std::vector<int> data = {1, 2, 3};
    EXPECT_THROW(algo.sort(std::span{data}), std::invalid_argument);
}

class MoveTracker {
public:
    enum class State { Valid, MovedFrom, Destructed };

private:
    std::unique_ptr<int> value_;
    State state_{State::Valid};
    int id_{-1};

public:
    explicit MoveTracker(int val, int id = -1)
        : value_(std::make_unique<int>(val)), state_(State::Valid), id_(id) {}

    ~MoveTracker() {
        if (state_ == State::Destructed) {
            ADD_FAILURE() << "Double destruction detected on MoveTracker id=" << id_;
        }
        state_ = State::Destructed;
        value_.reset();
    }

    MoveTracker(const MoveTracker&) = delete;
    MoveTracker& operator=(const MoveTracker&) = delete;

    MoveTracker(MoveTracker&& other) noexcept {
        if (other.state_ == State::MovedFrom) {
            ADD_FAILURE() << "Use-After-Move: Move constructor called on moved-from object id="
                          << other.id_;
        }
        value_ = std::move(other.value_);
        id_ = other.id_;
        state_ = State::Valid;
        other.state_ = State::MovedFrom;
        other.id_ = -1000 - other.id_;
    }

    MoveTracker& operator=(MoveTracker&& other) noexcept {
        if (this == &other)
            return *this;
        if (other.state_ == State::MovedFrom) {
            ADD_FAILURE() << "Use-After-Move: Move assignment from moved-from object id="
                          << other.id_;
        }
        value_ = std::move(other.value_);
        id_ = other.id_;
        state_ = State::Valid;
        other.state_ = State::MovedFrom;
        other.id_ = -1000 - other.id_;
        return *this;
    }

    bool operator<(const MoveTracker& other) const {
        if (state_ != State::Valid) {
            ADD_FAILURE() << "Use-After-Move: LHS in operator< is in invalid state";
        }
        if (other.state_ != State::Valid) {
            ADD_FAILURE() << "Use-After-Move: RHS in operator< is in invalid state";
        }
        return *value_ < *other.value_;
    }

    bool operator==(const MoveTracker& other) const {
        if (state_ != State::Valid || other.state_ != State::Valid) {
            ADD_FAILURE() << "Use-After-Move: operator== on invalid state";
        }
        return *value_ == *other.value_;
    }

    bool operator!=(const MoveTracker& other) const {
        return !(*this == other);
    }
    bool operator>(const MoveTracker& other) const {
        return other < *this;
    }
    bool operator<=(const MoveTracker& other) const {
        return !(other < *this);
    }
    bool operator>=(const MoveTracker& other) const {
        return !(*this < other);
    }

    int value() const {
        if (state_ != State::Valid)
            ADD_FAILURE() << "Read from moved-from object";
        return *value_;
    }
    bool is_valid() const {
        return state_ == State::Valid;
    }
};

TEST(CycleSortMemoryTest, HandlesMoveOnlyObjectsWithoutUseAfterMove) {
    std::vector<int> raw = {5, 2, 8, 1, 9, 3, 7, 4, 6, 0};
    std::vector<MoveTracker> data;
    for (size_t i = 0; i < raw.size(); ++i) {
        data.emplace_back(raw[i], static_cast<int>(i));
    }

    algoat::sorting::CycleSort{}.sort(std::span{data});

    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_TRUE(data[i].is_valid()) << "Element at index " << i << " is moved-from!";
        if (i > 0) {
            EXPECT_LE(data[i - 1].value(), data[i].value());
        }
    }
}
