/**
 * @file policy.hpp
 * @brief Declarative compile-time policy tag chaining and structural NTTP descriptors.
 *
 * Provides C++20 structural class descriptors (PolicyDescriptor) and policy tags
 * composable via operator| into PolicyChain instances with compile-time mutual
 * exclusivity and invariant verification.
 */

#pragma once

#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace algoat::meta {

/**
 * @struct PolicyDescriptor
 * @brief C++20 Structural aggregate descriptor holding normalized compile-time policies.
 *
 * Meets ISO/IEC 14882:2020 [temp.param]p7 requirements for structural class NTTPs:
 * - Literal class with all public and non-mutable non-static data members.
 * - Aggregate status preserved (enables designated initializer syntax).
 * - Defaulted operator== for equivalence and constexpr evaluation.
 */
struct PolicyDescriptor {
    bool stable = false;        ///< Require stable sort order for equivalent elements.
    bool in_place = false;      ///< Require strict O(1) auxiliary space / in-place execution.
    bool nearly_sorted = false; ///< Hint that input data is partially sorted.
    bool force_radix = false;   ///< Force non-comparative Radix Sort kernel.

    /**
     * @brief Structural equivalence comparator required by C++20 NTTP specifications.
     */
    constexpr bool operator==(const PolicyDescriptor&) const = default;
};

static_assert(std::is_aggregate_v<PolicyDescriptor>,
              "PolicyDescriptor must remain a structural aggregate for designated initializers.");

// ============================================================================
// Policy Facets (Orthogonal Category Discriminators)
// ============================================================================

/**
 * @struct stability_facet
 * @brief Category tag identifying policies that dictate element stability guarantees.
 */
struct stability_facet {};

/**
 * @struct memory_facet
 * @brief Category tag identifying policies that dictate auxiliary memory bounds.
 */
struct memory_facet {};

/**
 * @struct order_facet
 * @brief Category tag identifying policies that provide distribution or presorted hints.
 */
struct order_facet {};

/**
 * @struct kernel_facet
 * @brief Category tag identifying policies that explicitly select algorithm families.
 */
struct kernel_facet {};

// ============================================================================
// Policy Tag Concept
// ============================================================================

/**
 * @concept IsPolicyTag
 * @brief Satisfied by types representing individual declarative policy tags.
 */
template <typename T>
concept IsPolicyTag = requires {
    typename std::remove_cvref_t<T>::facet;
    { std::remove_cvref_t<T>::apply(std::declval<PolicyDescriptor&>()) };
};

// ============================================================================
// Declarative Policy Tags & Singletons
// ============================================================================
namespace policy {

/**
 * @struct stable_t
 * @brief Requires strict stability (relative order of equivalent keys is preserved).
 */
struct stable_t {
    using facet = stability_facet;

    /**
     * @brief Applies stability policy to descriptor.
     * @param d Target descriptor to mutate.
     */
    static constexpr void apply(PolicyDescriptor& d) noexcept {
        d.stable = true;
    }

    /**
     * @brief Implicit conversion to structural PolicyDescriptor.
     */
    [[nodiscard]] constexpr operator PolicyDescriptor() const noexcept {
        return PolicyDescriptor{.stable = true};
    }
};

/**
 * @struct in_place_t
 * @brief Enforces strict O(1) auxiliary memory footprint during sorting.
 */
struct in_place_t {
    using facet = memory_facet;

    /**
     * @brief Applies in-place policy to descriptor.
     * @param d Target descriptor to mutate.
     */
    static constexpr void apply(PolicyDescriptor& d) noexcept {
        d.in_place = true;
    }

    /**
     * @brief Implicit conversion to structural PolicyDescriptor.
     */
    [[nodiscard]] constexpr operator PolicyDescriptor() const noexcept {
        return PolicyDescriptor{.in_place = true};
    }
};

/**
 * @struct nearly_sorted_t
 * @brief Hints that the data is partially or nearly sorted (routes to adaptive TimSort).
 */
struct nearly_sorted_t {
    using facet = order_facet;

    /**
     * @brief Applies nearly-sorted policy to descriptor.
     * @param d Target descriptor to mutate.
     */
    static constexpr void apply(PolicyDescriptor& d) noexcept {
        d.nearly_sorted = true;
    }

    /**
     * @brief Implicit conversion to structural PolicyDescriptor.
     */
    [[nodiscard]] constexpr operator PolicyDescriptor() const noexcept {
        return PolicyDescriptor{.nearly_sorted = true};
    }
};

/**
 * @struct force_radix_t
 * @brief Forces linear-time Radix Sort kernel (requires integral element types).
 */
struct force_radix_t {
    using facet = kernel_facet;

    /**
     * @brief Applies force-radix policy to descriptor.
     * @param d Target descriptor to mutate.
     */
    static constexpr void apply(PolicyDescriptor& d) noexcept {
        d.force_radix = true;
    }

    /**
     * @brief Implicit conversion to structural PolicyDescriptor.
     */
    [[nodiscard]] constexpr operator PolicyDescriptor() const noexcept {
        return PolicyDescriptor{.force_radix = true};
    }
};

inline constexpr stable_t stable{};
inline constexpr in_place_t in_place{};
inline constexpr nearly_sorted_t nearly_sorted{};
inline constexpr force_radix_t force_radix{};

} // namespace policy

// ============================================================================
// PolicyChain Container & Validation
// ============================================================================

/**
 * @struct PolicyChain
 * @brief Type-level heterogeneous container representing a composed chain of policy tags.
 *
 * Verifies compile-time mutual exclusivity and duplicate tags using fold expressions.
 * Provides zero-overhead static conversion to PolicyDescriptor.
 *
 * @tparam Tags Parameter pack of policy tag types.
 */
template <typename... Tags> struct PolicyChain {
    /**
     * @brief Checks if a specific tag is present in the chain.
     */
    template <typename T> static constexpr bool has = (std::is_same_v<T, Tags> || ...);

    /**
     * @brief Counts the occurrences of tags belonging to a given facet.
     */
    template <typename Facet>
    static constexpr std::size_t count_facet =
        (0 + ... + (std::is_same_v<typename Tags::facet, Facet> ? 1 : 0));

    /**
     * @brief Counts the occurrences of a specific tag type in the chain.
     */
    template <typename Tag>
    static constexpr std::size_t count_tag = (0 + ... + (std::is_same_v<Tag, Tags> ? 1 : 0));

    // Compile-time mutual exclusivity and duplicate checking
    static_assert(((count_tag<Tags> <= 1) && ...),
                  "PolicyChain error: Duplicate policy tags specified in chain.");

    static_assert(count_facet<stability_facet> <= 1,
                  "PolicyChain error: Conflicting stability policies specified.");

    static_assert(count_facet<memory_facet> <= 1,
                  "PolicyChain error: Conflicting memory policies specified.");

    static_assert(count_facet<order_facet> <= 1,
                  "PolicyChain error: Conflicting ordering policies specified.");

    static_assert(count_facet<kernel_facet> <= 1,
                  "PolicyChain error: Conflicting algorithm kernel overrides specified.");

    // Semantic cross-facet exclusivity checks
    static_assert(!(has<policy::force_radix_t> && has<policy::stable_t> && has<policy::in_place_t>),
                  "PolicyChain error: Stable radix sort requires an auxiliary buffer and cannot be "
                  "combined with in_place.");

    static_assert(!(has<policy::nearly_sorted_t> && has<policy::force_radix_t>),
                  "PolicyChain error: nearly_sorted and force_radix policies are mutually "
                  "exclusive.");

    /**
     * @brief Normalizes all tags in the chain into a structural PolicyDescriptor at compile time.
     * @return Evaluated PolicyDescriptor.
     */
    [[nodiscard]] static constexpr PolicyDescriptor to_descriptor() noexcept {
        PolicyDescriptor desc{};
        (Tags::apply(desc), ...);
        return desc;
    }

    /**
     * @brief Implicit conversion operator enabling direct usage as a structural NTTP.
     */
    [[nodiscard]] constexpr operator PolicyDescriptor() const noexcept {
        return to_descriptor();
    }
};

/**
 * @struct is_policy_chain
 * @brief Type trait to identify PolicyChain specializations.
 */
template <typename T> struct is_policy_chain : std::false_type {};

template <typename... Tags> struct is_policy_chain<PolicyChain<Tags...>> : std::true_type {};

/**
 * @concept IsPolicyChain
 * @brief Concept satisfied only by PolicyChain types.
 */
template <typename T>
concept IsPolicyChain = is_policy_chain<std::remove_cvref_t<T>>::value;

// ============================================================================
// operator| Composition Overloads
// ============================================================================

namespace policy {

/**
 * @brief Composes two policy tags into a PolicyChain.
 * @param t1 First policy tag.
 * @param t2 Second policy tag.
 * @return PolicyChain combining both tags.
 */
template <IsPolicyTag T1, IsPolicyTag T2> constexpr auto operator|(T1 t1, T2 t2) noexcept {
    (void)t1;
    (void)t2;
    return PolicyChain<std::remove_cvref_t<T1>, std::remove_cvref_t<T2>>{};
}

/**
 * @brief Appends a policy tag to an existing PolicyChain.
 * @param chain Existing PolicyChain.
 * @param tag Policy tag to append.
 * @return Expanded PolicyChain.
 */
template <IsPolicyChain Chain, IsPolicyTag Tag>
constexpr auto operator|(Chain chain, Tag tag) noexcept {
    (void)chain;
    (void)tag;
    return []<typename... Tags>(PolicyChain<Tags...>) {
        return PolicyChain<Tags..., std::remove_cvref_t<Tag>>{};
    }(Chain{});
}

/**
 * @brief Prepends a policy tag to an existing PolicyChain.
 * @param tag Policy tag to prepend.
 * @param chain Existing PolicyChain.
 * @return Expanded PolicyChain.
 */
template <IsPolicyTag Tag, IsPolicyChain Chain>
constexpr auto operator|(Tag tag, Chain chain) noexcept {
    (void)tag;
    (void)chain;
    return []<typename... Tags>(PolicyChain<Tags...>) {
        return PolicyChain<std::remove_cvref_t<Tag>, Tags...>{};
    }(Chain{});
}

/**
 * @brief Concatenates two PolicyChain instances.
 * @param c1 First PolicyChain.
 * @param c2 Second PolicyChain.
 * @return Combined PolicyChain.
 */
template <IsPolicyChain C1, IsPolicyChain C2> constexpr auto operator|(C1 c1, C2 c2) noexcept {
    (void)c1;
    (void)c2;
    return []<typename... T1, typename... T2>(PolicyChain<T1...>, PolicyChain<T2...>) {
        return PolicyChain<T1..., T2...>{};
    }(C1{}, C2{});
}

} // namespace policy

} // namespace algoat::meta
