#include "algoat/core/registry.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace algoat::core;

struct MockAlgo {
    std::string name;
};

using MockVariant = std::variant<MockAlgo>;
using MockRegistry = Registry<MockVariant>;

TEST(RegistryTest, RegisterAndCreate) {
    MockRegistry registry;
    registry.register_algorithm("algo1", []() -> std::any { return MockAlgo{"algo1_instance"}; });

    ASSERT_TRUE(registry.has("algo1"));
    auto instance = registry.create("algo1");
    EXPECT_EQ(std::get<MockAlgo>(instance).name, "algo1_instance");
}

TEST(RegistryTest, DuplicateRegistrationThrows) {
    MockRegistry registry;
    registry.register_algorithm("algo1", []() -> std::any { return MockAlgo{"first"}; });

    EXPECT_THROW(
        registry.register_algorithm("algo1", []() -> std::any { return MockAlgo{"second"}; }),
        std::runtime_error);
}

TEST(RegistryTest, UnknownNameThrows) {
    MockRegistry registry;
    EXPECT_THROW(registry.create("unknown"), std::runtime_error);
}

TEST(RegistryTest, ListRegistered) {
    MockRegistry registry;
    registry.register_algorithm("algo2", []() -> std::any { return MockAlgo{"a2"}; });
    registry.register_algorithm("algo1", []() -> std::any { return MockAlgo{"a1"}; });

    auto registered = registry.list_registered();
    EXPECT_EQ(registered.size(), 2);

    // std::unordered_map order is not guaranteed, so sort before checking
    std::sort(registered.begin(), registered.end());
    EXPECT_EQ(registered[0], "algo1");
    EXPECT_EQ(registered[1], "algo2");
}

TEST(RegistryTest, GetReturnsOptional) {
    MockRegistry registry;
    registry.register_algorithm("algo1", []() -> std::any { return MockAlgo{"algo1_instance"}; });

    auto factory_opt = registry.get("algo1");
    ASSERT_TRUE(factory_opt.has_value());
    auto result = any_to_variant<MockVariant>((*factory_opt)());
    EXPECT_EQ(std::get<MockAlgo>(result).name, "algo1_instance");

    EXPECT_FALSE(registry.get("unknown").has_value());
}

TEST(RegistryTest, TransparentStringViewLookup) {
    MockRegistry registry;
    registry.register_algorithm("my_algo", []() -> std::any { return MockAlgo{"val"}; });

    std::string_view sv = "my_algo";
    EXPECT_TRUE(registry.has(sv));
    auto sv_factory = registry.get(sv);
    ASSERT_TRUE(sv_factory.has_value());
    auto sv_result = any_to_variant<MockVariant>((*sv_factory)());
    EXPECT_EQ(std::get<MockAlgo>(registry.create(sv)).name, "val");
}

TEST(RegistryTest, ExtremeEdgeCases) {
    MockRegistry registry;
    // Empty string registration
    registry.register_algorithm("", []() -> std::any { return MockAlgo{"empty"}; });
    EXPECT_TRUE(registry.has(""));
    EXPECT_EQ(std::get<MockAlgo>(registry.create("")).name, "empty");

    // Null characters in name
    std::string bad_name("a\0b", 3);
    registry.register_algorithm(bad_name, []() -> std::any { return MockAlgo{"null_char"}; });
    EXPECT_TRUE(registry.has(bad_name));

    // Very long name
    std::string long_name(10000, 'x');
    registry.register_algorithm(long_name, []() -> std::any { return MockAlgo{"long"}; });
    EXPECT_TRUE(registry.has(long_name));
}
