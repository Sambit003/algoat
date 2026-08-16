/**
 * @file registry.hpp
 * @brief Type-safe runtime algorithm registry and factory pattern.
 * 
 * Implements an extensible registry that stores algorithm factory functions keyed
 * by string identifiers, returning a @c std::variant of algorithm instances to avoid
 * virtual table overhead while maintaining runtime polymorphism.
 */

#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <functional>
#include <vector>
#include <stdexcept>

namespace algoat::core {

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
template<typename AlgoVariant>
class Registry {
public:
    /// Type alias for algorithm factory callables.
    using FactoryFn = std::function<AlgoVariant()>;

    /**
     * @brief Registers an algorithm factory under a unique string name.
     * 
 *
 * @param name Unique identifier for the algorithm (e.g., "quicksort").
 *
 * @param factory Callable that constructs and returns the algorithm variant.
     * @throws std::runtime_error If an algorithm with the given name is already registered.
     */
    void register_algo(std::string_view name, FactoryFn factory) {
        std::string name_str{name};
        if (factories_.contains(name_str)) {
            throw std::runtime_error("Algorithm already registered: " + name_str);
        }
        factories_[name_str] = std::move(factory);
    }

    /**
     * @brief Creates an algorithm variant instance by name.
     * 
 *
 * @param name Name of the algorithm to instantiate.
     * @return @c AlgoVariant The constructed algorithm variant.
     * @throws std::runtime_error If the algorithm name is not registered.
     */
    AlgoVariant create(std::string_view name) const {
        std::string name_str{name};
        auto it = factories_.find(name_str);
        if (it == factories_.end()) {
            throw std::runtime_error("Algorithm not found in registry: " + name_str);
        }
        return it->second();
    }

    /**
     * @brief Checks whether an algorithm is registered under the given name.
     * 
 *
 * @param name Name to query.
     * @return @c true if registered, @c false otherwise.
     */
    bool has(std::string_view name) const {
        return factories_.contains(std::string{name});
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

private:
    std::unordered_map<std::string, FactoryFn> factories_; ///< Stored algorithm constructors.
};

} // namespace algoat::core
