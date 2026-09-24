/**
 * @file registry.hpp
 * @brief Type-safe runtime algorithm registry and factory pattern.
 *
 * Implements an extensible registry that stores algorithm factory functions keyed
 * by string identifiers, returning a @c std::variant of algorithm instances to avoid
 * virtual table overhead while maintaining runtime polymorphism.
 */

#pragma once

#include <any>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace algoat::core {

struct StringHash {
    using transparent_key_equal = std::equal_to<>;
    using is_transparent = void;

    size_t operator()(std::string_view txt) const {
        return std::hash<std::string_view>{}(txt);
    }
};

template <typename Variant, typename... Types>
Variant any_to_variant_impl(const std::any& a, std::variant<Types...>*) {
    if (auto* ptr = std::any_cast<Variant>(&a)) {
        return *ptr;
    }

    Variant result;
    bool found = (... || [&]() {
        if (auto* ptr = std::any_cast<Types>(&a)) {
            result = *ptr;
            return true;
        }
        return false;
    }());
    if (!found)
        throw std::bad_any_cast();
    return result;
}

template <typename Variant> Variant any_to_variant(const std::any& a) {
    return any_to_variant_impl<Variant>(a, static_cast<Variant*>(nullptr));
}

class BaseRegistry {
public:
    using AnyFactoryFn = std::any (*)();

    static BaseRegistry& global(std::string_view domain) {
        static std::unordered_map<std::string, BaseRegistry> instances;
        return instances[std::string(domain)];
    }

    void register_algorithm(std::string_view name, AnyFactoryFn factory) {
        if (factories_.contains(name)) {
            throw std::runtime_error("Algorithm already registered: " + std::string(name));
        }
        factories_.emplace(std::string(name), factory);
    }

    [[nodiscard]] std::optional<AnyFactoryFn> get(std::string_view name) const noexcept {
        auto it = factories_.find(name);
        if (it != factories_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    /**
     * @brief Checks whether an algorithm is registered under the given name.
     *
     *
     * @param name Name to query.
     * @return @c true if registered, @c false otherwise.
     */
    bool has(std::string_view name) const noexcept {
        return factories_.contains(name);
    }

    /**
     * @brief Returns a list of all registered algorithm names.
     *
     * @return <tt>std::vector<std::string></tt> List of registered algorithm identifiers.
     */
    std::vector<std::string> list_registered() const {
        std::vector<std::string> names;
        names.reserve(factories_.size());
        for (const auto& [name, _] : factories_) {
            names.push_back(name);
        }
        return names;
    }

protected:
    std::unordered_map<std::string, AnyFactoryFn, StringHash, std::equal_to<>> factories_;
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
    static Registry global(std::string_view domain) {
        return Registry(domain);
    }

    Registry(std::string_view domain = "default") : base_(BaseRegistry::global(domain)) {}

    /// Forwards stateless factory function pointers to the underlying BaseRegistry.
    void register_algorithm(std::string_view name, BaseRegistry::AnyFactoryFn factory) {
        base_.register_algorithm(name, factory);
    }

    AlgoVariant create(std::string_view name) const {
        auto factory = base_.get(name);
        if (!factory) {
            throw std::runtime_error("Algorithm not found in registry: " + std::string(name));
        }
        return any_to_variant<AlgoVariant>((*factory)());
    }

    [[nodiscard]] std::optional<BaseRegistry::AnyFactoryFn>
    get(std::string_view name) const noexcept {
        return base_.get(name);
    }

    bool has(std::string_view name) const noexcept {
        return base_.has(name);
    }
    std::vector<std::string> list_registered() const {
        return base_.list_registered();
    }

private:
    BaseRegistry& base_;
};

} // namespace algoat::core

#define ALGOAT_CONCAT_IMPL(x, y) x##y
#define ALGOAT_CONCAT(x, y) ALGOAT_CONCAT_IMPL(x, y)

#define ALGOAT_REGISTER_ALGORITHM_IMPL(Domain, Name, AlgoType, Counter)                            \
    inline const auto ALGOAT_CONCAT(registrar_, Counter) = []() {                                  \
        ::algoat::core::BaseRegistry::global(Domain).register_algorithm(                           \
            Name, []() -> std::any { return AlgoType{}; });                                        \
        return 0;                                                                                  \
    }();

#define ALGOAT_REGISTER_ALGORITHM(Domain, Name, AlgoType)                                          \
    ALGOAT_REGISTER_ALGORITHM_IMPL(Domain, Name, AlgoType, __COUNTER__)
