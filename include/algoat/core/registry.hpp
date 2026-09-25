/**
 * @file registry.hpp
 * @brief Type-safe runtime algorithm registry and factory pattern.
 *
 * Implements an extensible registry that stores algorithm factory functions keyed
 * by string identifiers, returning a @c std::variant of algorithm instances to avoid
 * virtual table overhead while maintaining runtime polymorphism.
 */

#pragma once

#include <algorithm>
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

inline size_t get_next_type_id() noexcept {
    static size_t id = 0;
    return id++;
}

template <typename T> size_t get_type_id() noexcept {
    static size_t id = get_next_type_id();
    return id;
}

template <typename Variant> struct VariantCaster;

template <typename... Types> struct VariantCaster<std::variant<Types...>> {
    using Variant = std::variant<Types...>;
    using CastFn = Variant (*)(const std::any&);

    static const std::vector<CastFn>& get_casters() {
        static std::vector<CastFn> casters = []() {
            std::vector<CastFn> arr;
            size_t max_id = std::max({get_type_id<Types>()...});
            arr.resize(max_id + 1, nullptr);
            (..., (arr[get_type_id<Types>()] = [](const std::any& a) -> Variant {
                 return *std::any_cast<Types>(&a);
             }));
            return arr;
        }();
        return casters;
    }
};

class BaseRegistry {
public:
    using AnyFactoryFn = std::any (*)();

    struct RegistryEntry {
        size_t type_id;
        AnyFactoryFn factory;
    };

    static BaseRegistry& global(std::string_view domain) {
        static std::unordered_map<std::string, BaseRegistry, StringHash, std::equal_to<>> instances;
        auto it = instances.find(domain);
        if (it == instances.end()) {
            it = instances.emplace(std::string(domain), BaseRegistry{}).first;
        }
        return it->second;
    }

    void register_algorithm(std::string_view name, size_t type_id, AnyFactoryFn factory) {
        if (factories_.contains(name)) {
            throw std::runtime_error("Algorithm already registered: " + std::string(name));
        }
        factories_.emplace(std::string(name), RegistryEntry{type_id, factory});
    }

    [[nodiscard]] std::optional<RegistryEntry> get(std::string_view name) const noexcept {
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
    std::unordered_map<std::string, RegistryEntry, StringHash, std::equal_to<>> factories_;
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
    struct TypedFactory {
        size_t type_id;
        BaseRegistry::AnyFactoryFn raw_factory;

        AlgoVariant operator()() const {
            const auto& casters = VariantCaster<AlgoVariant>::get_casters();
            if (type_id >= casters.size() || !casters[type_id]) {
                throw std::bad_any_cast();
            }
            return casters[type_id](raw_factory());
        }
    };

    static Registry global(std::string_view domain) {
        return Registry(domain);
    }

    Registry(std::string_view domain = "default") : base_(BaseRegistry::global(domain)) {}

    void register_algorithm(std::string_view name, size_t type_id,
                            BaseRegistry::AnyFactoryFn factory) {
        base_.register_algorithm(name, type_id, factory);
    }

    AlgoVariant create(std::string_view name) const {
        auto factory_opt = get(name);
        if (!factory_opt) {
            throw std::runtime_error("Algorithm not found in registry: " + std::string(name));
        }
        return (*factory_opt)();
    }

    [[nodiscard]] std::optional<TypedFactory> get(std::string_view name) const noexcept {
        auto entry_opt = base_.get(name);
        if (!entry_opt) {
            return std::nullopt;
        }
        return TypedFactory{entry_opt->type_id, entry_opt->factory};
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
            Name, ::algoat::core::get_type_id<AlgoType>(),                                         \
            []() -> std::any { return AlgoType{}; });                                              \
        return 0;                                                                                  \
    }();

#define ALGOAT_REGISTER_ALGORITHM(Domain, Name, AlgoType)                                          \
    ALGOAT_REGISTER_ALGORITHM_IMPL(Domain, Name, AlgoType, __COUNTER__)
