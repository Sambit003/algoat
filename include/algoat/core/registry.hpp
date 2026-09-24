/**
 * @file registry.hpp
 * @brief Type-safe runtime algorithm registry and factory pattern.
 *
 * Implements an extensible registry that stores algorithm factory functions keyed
 * by string identifiers, returning a @c std::variant of algorithm instances to avoid
 * virtual table overhead while maintaining runtime polymorphism.
 */

#pragma once

#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace algoat::core {

struct StringHash {
    using transparent_key_equal = std::equal_to<>;
    using is_transparent = void;

    size_t operator()(std::string_view txt) const {
        return std::hash<std::string_view>{}(txt);
    }
    size_t operator()(const std::string& txt) const {
        return std::hash<std::string_view>{}(txt);
    }
    size_t operator()(const char* txt) const {
        return std::hash<std::string_view>{}(txt);
    }
};

/**
 * @class Registry
 * @brief Dynamic factory registry mapping algorithm names to instance creators.
 *
 * Uses <tt>std::function<AlgoVariant()></tt> to construct algorithm objects held within a
 * @c std::variant. This enables dynamic string-based selection with zero virtual function
 * dispatch overhead via @c std::visit.
 *
 * @tparam AlgoVariant A @c std::variant containing all supported algorithm types.
 */
template <typename AlgoVariant> class Registry {
public:
    /// Type alias for algorithm factory callables.
    using FactoryFn = std::function<AlgoVariant()>;

    static Registry& global() {
        static Registry instance;
        return instance;
    }

    void register_algorithm(std::string_view name, FactoryFn factory) {
        if (factories_.contains(name)) {
            throw std::runtime_error("Algorithm already registered: " + std::string(name));
        }
        factories_.emplace(std::string(name), std::move(factory));
    }

    // Retained for backwards compatibility if needed
    void register_algo(std::string_view name, FactoryFn factory) {
        register_algorithm(name, std::move(factory));
    }

    [[nodiscard]] std::optional<FactoryFn> get(std::string_view name) const noexcept {
        auto it = factories_.find(name);
        if (it != factories_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    AlgoVariant create(std::string_view name) const {
        auto factory = get(name);
        if (!factory) {
            throw std::runtime_error("Algorithm not found in registry: " + std::string(name));
        }
        return (*factory)();
    }

    bool has(std::string_view name) const noexcept {
        return factories_.contains(name);
    }

    std::vector<std::string> list_registered() const {
        std::vector<std::string> names;
        names.reserve(factories_.size());
        for (const auto& [name, _] : factories_) {
            names.push_back(name);
        }
        return names;
    }

private:
    std::unordered_map<std::string, FactoryFn, StringHash, std::equal_to<>>
        factories_; ///< Stored algorithm constructors.
};

} // namespace algoat::core

#define ALGOAT_CONCAT_IMPL(x, y) x##y
#define ALGOAT_CONCAT(x, y) ALGOAT_CONCAT_IMPL(x, y)

#define ALGOAT_REGISTER_ALGORITHM_IMPL(VariantType, Name, AlgoType, Counter)                       \
    inline const auto ALGOAT_CONCAT(registrar_, Counter) = []() {                                  \
        ::algoat::core::Registry<VariantType>::global().register_algorithm(                        \
            Name, []() -> VariantType { return AlgoType{}; });                                     \
        return 0;                                                                                  \
    }();

#define ALGOAT_REGISTER_ALGORITHM(VariantType, Name, AlgoType)                                     \
    ALGOAT_REGISTER_ALGORITHM_IMPL(VariantType, Name, AlgoType, __COUNTER__)
