#include "algoat/core/registry.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace algoat::core;

struct MockAlgo {
    std::string name;
};

using MockRegistry = Registry<MockAlgo>;

TEST(RegistryTest, RegisterAndCreate) {
    MockRegistry registry;
    registry.register_algo("algo1", []() { return MockAlgo{"algo1_instance"}; });

    ASSERT_TRUE(registry.has("algo1"));
    auto instance = registry.create("algo1");
    EXPECT_EQ(instance.name, "algo1_instance");
}

TEST(RegistryTest, DuplicateRegistrationThrows) {
    MockRegistry registry;
    registry.register_algo("algo1", []() { return MockAlgo{"first"}; });

    EXPECT_THROW(registry.register_algo("algo1", []() { return MockAlgo{"second"}; }),
                 std::runtime_error);
}

TEST(RegistryTest, UnknownNameThrows) {
    MockRegistry registry;
    EXPECT_THROW(registry.create("unknown"), std::runtime_error);
}

TEST(RegistryTest, ListRegistered) {
    MockRegistry registry;
    registry.register_algo("algo2", []() { return MockAlgo{"a2"}; });
    registry.register_algo("algo1", []() { return MockAlgo{"a1"}; });

    auto registered = registry.list_registered();
    EXPECT_EQ(registered.size(), 2);

    // std::unordered_map order is not guaranteed, so sort before checking
    std::sort(registered.begin(), registered.end());
    EXPECT_EQ(registered[0], "algo1");
    EXPECT_EQ(registered[1], "algo2");
}

TEST(RegistryTest, GetReturnsOptional) {
    MockRegistry registry;
    registry.register_algo("algo1", []() { return MockAlgo{"algo1_instance"}; });

    auto factory_opt = registry.get("algo1");
    ASSERT_TRUE(factory_opt.has_value());
    EXPECT_EQ((*factory_opt)().name, "algo1_instance");

    EXPECT_FALSE(registry.get("unknown").has_value());
}

TEST(RegistryTest, TransparentStringViewLookup) {
    MockRegistry registry;
    registry.register_algo("my_algo", []() { return MockAlgo{"val"}; });

    std::string_view sv = "my_algo";
    EXPECT_TRUE(registry.has(sv));
    EXPECT_TRUE(registry.get(sv).has_value());
    EXPECT_EQ(registry.create(sv).name, "val");
}

TEST(RegistryTest, ExtremeEdgeCases) {
    MockRegistry registry;
    // Empty string registration
    registry.register_algorithm("", []() { return MockAlgo{"empty"}; });
    EXPECT_TRUE(registry.has(""));
    EXPECT_EQ(registry.create("").name, "empty");

    // Null characters in name
    std::string bad_name("a\0b", 3);
    registry.register_algorithm(bad_name, []() { return MockAlgo{"null_char"}; });
    EXPECT_TRUE(registry.has(bad_name));

    // Very long name
    std::string long_name(10000, 'x');
    registry.register_algorithm(long_name, []() { return MockAlgo{"long"}; });
    EXPECT_TRUE(registry.has(long_name));
}
