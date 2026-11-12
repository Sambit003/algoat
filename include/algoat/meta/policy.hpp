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

/**
 * @struct SearchPolicyDescriptor
 * @brief C++20 Structural aggregate descriptor holding compile-time search policies.
 *
 * Meets ISO/IEC 14882:2020 [temp.param]p7 requirements for structural class NTTPs.
 */
struct SearchPolicyDescriptor {
    bool branchless = false;   ///< Force branchless CMOV binary search
    bool prefetch = true;      ///< Enable software prefetching
    bool eytzinger = false;    ///< Use Eytzinger breadth-first layout
    bool static_btree = false; ///< Use 64-byte static B-tree layout
    bool galloping = false;    ///< Use galloping exponential search

    /**
     * @brief Structural equivalence comparator required by C++20 NTTP specifications.
     */
    constexpr bool operator==(const SearchPolicyDescriptor&) const = default;
};

static_assert(std::is_aggregate_v<SearchPolicyDescriptor>,
              "SearchPolicyDescriptor must remain a structural aggregate for designated "
              "initializers.");

/**
 * @struct SelectPolicyDescriptor
 * @brief C++20 Structural aggregate descriptor holding compile-time selection policies.
 *
 * Meets ISO/IEC 14882:2020 [temp.param]p7 requirements for structural class NTTPs.
 */
struct SelectPolicyDescriptor {
    bool in_place = true;          ///< Require in-place partitioning
    bool guaranteed_linear = true; ///< Enforce O(N) worst-case time (Introselect)
    bool floyd_rivest = false;     ///< Use Floyd-Rivest Algorithm 489 sampling
    bool branchless = false;       ///< Use BlockQuicksort branchless partitioning

    /**
     * @brief Structural equivalence comparator required by C++20 NTTP specifications.
     */
    constexpr bool operator==(const SelectPolicyDescriptor&) const = default;
};

static_assert(std::is_aggregate_v<SelectPolicyDescriptor>,
              "SelectPolicyDescriptor must remain a structural aggregate for designated "
              "initializers.");

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
 * @brief Category tag identifying policies that explicitly select sorting algorithm families.
 */
struct kernel_facet {};

/**
 * @struct prefetch_facet
 * @brief Category tag identifying policies that configure software memory prefetching.
 */
struct prefetch_facet {};

/**
 * @struct branching_facet
 * @brief Category tag identifying policies that enforce branchless execution paths.
 */
struct branching_facet {};

/**
 * @struct search_layout_facet
 * @brief Category tag identifying mutually exclusive search layouts or access patterns.
 */
struct search_layout_facet {};

/**
 * @struct selection_guarantee_facet
 * @brief Category tag identifying selection worst-case complexity guarantees.
 */
struct selection_guarantee_facet {};

/**
 * @struct selection_kernel_facet
 * @brief Category tag identifying explicit selection algorithm family overrides.
 */
struct selection_kernel_facet {};

// ============================================================================
// Policy Tag Concept
// ============================================================================

/**
 * @concept TagApplicableTo
 * @brief Satisfied when policy tag T can mutate structural descriptor type Descriptor.
 */
template <typename T, typename Descriptor>
concept TagApplicableTo = requires(Descriptor& d) { std::remove_cvref_t<T>::apply(d); };

/**
 * @concept IsPolicyTag
 * @brief Satisfied by types representing individual declarative policy tags for sorting,
 * searching, or selection.
 */
template <typename T>
concept IsPolicyTag =
    requires { typename std::remove_cvref_t<T>::facet; } &&
    (TagApplicableTo<T, PolicyDescriptor> || TagApplicableTo<T, SearchPolicyDescriptor> ||
     TagApplicableTo<T, SelectPolicyDescriptor>);

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
 * @brief Enforces strict O(1) auxiliary memory footprint during sorting and selection.
 */
struct in_place_t {
    using facet = memory_facet;

    /**
     * @brief Applies in-place policy to sorting descriptor.
     * @param d Target sorting descriptor to mutate.
     */
    static constexpr void apply(PolicyDescriptor& d) noexcept {
        d.in_place = true;
    }

    /**
     * @brief Applies in-place policy to selection descriptor.
     * @param d Target selection descriptor to mutate.
     */
    static constexpr void apply(SelectPolicyDescriptor& d) noexcept {
        d.in_place = true;
    }

    /**
     * @brief Implicit conversion to structural PolicyDescriptor.
     */
    [[nodiscard]] constexpr operator PolicyDescriptor() const noexcept {
        return PolicyDescriptor{.in_place = true};
    }

    /**
     * @brief Implicit conversion to structural SelectPolicyDescriptor.
     */
    [[nodiscard]] constexpr operator SelectPolicyDescriptor() const noexcept {
        return SelectPolicyDescriptor{.in_place = true};
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

/**
 * @struct branchless_t
 * @brief Enforces branchless CMOV searching or BlockQuicksort branchless partitioning.
 */
struct branchless_t {
    using facet = branching_facet;

    /**
     * @brief Applies branchless policy to search descriptor.
     * @param d Target search descriptor to mutate.
     */
    static constexpr void apply(SearchPolicyDescriptor& d) noexcept {
        d.branchless = true;
    }

    /**
     * @brief Applies branchless policy to selection descriptor.
     * @param d Target selection descriptor to mutate.
     */
    static constexpr void apply(SelectPolicyDescriptor& d) noexcept {
        d.branchless = true;
    }

    /**
     * @brief Implicit conversion to structural SearchPolicyDescriptor.
     */
    [[nodiscard]] constexpr operator SearchPolicyDescriptor() const noexcept {
        return SearchPolicyDescriptor{.branchless = true};
    }

    /**
     * @brief Implicit conversion to structural SelectPolicyDescriptor.
     */
    [[nodiscard]] constexpr operator SelectPolicyDescriptor() const noexcept {
        return SelectPolicyDescriptor{.branchless = true};
    }
};

/**
 * @struct prefetch_t
 * @brief Enables software memory prefetching during search traversal.
 */
struct prefetch_t {
    using facet = prefetch_facet;

    /**
     * @brief Applies prefetch policy to search descriptor.
     * @param d Target search descriptor to mutate.
     */
    static constexpr void apply(SearchPolicyDescriptor& d) noexcept {
        d.prefetch = true;
    }

    /**
     * @brief Implicit conversion to structural SearchPolicyDescriptor.
     */
    [[nodiscard]] constexpr operator SearchPolicyDescriptor() const noexcept {
        return SearchPolicyDescriptor{.prefetch = true};
    }
};

/**
 * @struct no_prefetch_t
 * @brief Disables software memory prefetching during search traversal.
 */
struct no_prefetch_t {
    using facet = prefetch_facet;

    /**
     * @brief Disables prefetch policy in search descriptor.
     * @param d Target search descriptor to mutate.
     */
    static constexpr void apply(SearchPolicyDescriptor& d) noexcept {
        d.prefetch = false;
    }

    /**
     * @brief Implicit conversion to structural SearchPolicyDescriptor.
     */
    [[nodiscard]] constexpr operator SearchPolicyDescriptor() const noexcept {
        return SearchPolicyDescriptor{.prefetch = false};
    }
};

/**
 * @struct eytzinger_t
 * @brief Selects the cache-oblivious Eytzinger breadth-first search layout.
 */
struct eytzinger_t {
    using facet = search_layout_facet;

    /**
     * @brief Applies Eytzinger layout policy to search descriptor.
     * @param d Target search descriptor to mutate.
     */
    static constexpr void apply(SearchPolicyDescriptor& d) noexcept {
        d.eytzinger = true;
    }

    /**
     * @brief Implicit conversion to structural SearchPolicyDescriptor.
     */
    [[nodiscard]] constexpr operator SearchPolicyDescriptor() const noexcept {
        return SearchPolicyDescriptor{.eytzinger = true};
    }
};

/**
 * @struct static_btree_t
 * @brief Selects the 64-byte cache-line aligned Static B-Tree search layout.
 */
struct static_btree_t {
    using facet = search_layout_facet;

    /**
     * @brief Applies Static B-Tree layout policy to search descriptor.
     * @param d Target search descriptor to mutate.
     */
    static constexpr void apply(SearchPolicyDescriptor& d) noexcept {
        d.static_btree = true;
    }

    /**
     * @brief Implicit conversion to structural SearchPolicyDescriptor.
     */
    [[nodiscard]] constexpr operator SearchPolicyDescriptor() const noexcept {
        return SearchPolicyDescriptor{.static_btree = true};
    }
};

/**
 * @struct galloping_t
 * @brief Selects Bentley & Yao galloping exponential search for low-rank targets.
 */
struct galloping_t {
    using facet = search_layout_facet;

    /**
     * @brief Applies galloping policy to search descriptor.
     * @param d Target search descriptor to mutate.
     */
    static constexpr void apply(SearchPolicyDescriptor& d) noexcept {
        d.galloping = true;
    }

    /**
     * @brief Implicit conversion to structural SearchPolicyDescriptor.
     */
    [[nodiscard]] constexpr operator SearchPolicyDescriptor() const noexcept {
        return SearchPolicyDescriptor{.galloping = true};
    }
};

/**
 * @struct introselect_t
 * @brief Enforces Musser's Introselect with guaranteed O(N) worst-case time via BFPRT fallback.
 */
struct introselect_t {
    using facet = selection_guarantee_facet;

    /**
     * @brief Applies introselect guarantee to selection descriptor.
     * @param d Target selection descriptor to mutate.
     */
    static constexpr void apply(SelectPolicyDescriptor& d) noexcept {
        d.guaranteed_linear = true;
    }

    /**
     * @brief Implicit conversion to structural SelectPolicyDescriptor.
     */
    [[nodiscard]] constexpr operator SelectPolicyDescriptor() const noexcept {
        return SelectPolicyDescriptor{.guaranteed_linear = true};
    }
};

/**
 * @typedef guaranteed_linear_t
 * @brief Alias policy enforcing strict O(N) worst-case time complexity for selection.
 */
using guaranteed_linear_t = introselect_t;

/**
 * @struct floyd_rivest_t
 * @brief Selects Floyd-Rivest Algorithm 489 sublinear sampling selection.
 */
struct floyd_rivest_t {
    using facet = selection_kernel_facet;

    /**
     * @brief Applies Floyd-Rivest policy to selection descriptor.
     * @param d Target selection descriptor to mutate.
     */
    static constexpr void apply(SelectPolicyDescriptor& d) noexcept {
        d.guaranteed_linear = false;
        d.floyd_rivest = true;
    }

    /**
     * @brief Implicit conversion to structural SelectPolicyDescriptor.
     */
    [[nodiscard]] constexpr operator SelectPolicyDescriptor() const noexcept {
        return SelectPolicyDescriptor{.guaranteed_linear = false, .floyd_rivest = true};
    }
};

inline constexpr stable_t stable{};
inline constexpr in_place_t in_place{};
inline constexpr nearly_sorted_t nearly_sorted{};
inline constexpr force_radix_t force_radix{};
inline constexpr branchless_t branchless{};
inline constexpr prefetch_t prefetch{};
inline constexpr no_prefetch_t no_prefetch{};
inline constexpr eytzinger_t eytzinger{};
inline constexpr static_btree_t static_btree{};
inline constexpr galloping_t galloping{};
inline constexpr introselect_t introselect{};
inline constexpr guaranteed_linear_t guaranteed_linear{};
inline constexpr floyd_rivest_t floyd_rivest{};

} // namespace policy

// ============================================================================
// PolicyChain Container & Validation
// ============================================================================

/**
 * @struct PolicyChain
 * @brief Type-level heterogeneous container representing a composed chain of policy tags.
 *
 * Verifies compile-time mutual exclusivity and duplicate tags using fold expressions.
 * Provides zero-overhead static conversion to PolicyDescriptor, SearchPolicyDescriptor,
 * and SelectPolicyDescriptor.
 *
 * @tparam Tags Parameter pack of policy tag types.
 */
template <typename... Tags> struct PolicyChain {
    /**
     * @brief Checks if a specific tag is present in the chain.
     */
    template <typename T> static constexpr bool has = (std::is_same_v<T, Tags> || ...);

    /**
     * @brief Checks whether every tag in the chain applies to Descriptor.
     */
    template <typename Descriptor>
    static constexpr bool applicable_to = (TagApplicableTo<Tags, Descriptor> && ...);

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

    static_assert(applicable_to<PolicyDescriptor> || applicable_to<SearchPolicyDescriptor> ||
                      applicable_to<SelectPolicyDescriptor>,
                  "PolicyChain error: Cannot mix incompatible domain policy tags across sorting, "
                  "searching, and selection.");

    static_assert(count_facet<stability_facet> <= 1,
                  "PolicyChain error: Conflicting stability policies specified.");

    static_assert(count_facet<memory_facet> <= 1,
                  "PolicyChain error: Conflicting memory policies specified.");

    static_assert(count_facet<order_facet> <= 1,
                  "PolicyChain error: Conflicting ordering policies specified.");

    static_assert(count_facet<kernel_facet> <= 1,
                  "PolicyChain error: Conflicting algorithm kernel overrides specified.");

    static_assert(count_facet<prefetch_facet> <= 1,
                  "PolicyChain error: Conflicting prefetch policies specified.");

    static_assert(count_facet<branching_facet> <= 1,
                  "PolicyChain error: Conflicting branching policies specified.");

    // Explicit check before search_layout_facet so eytzinger | static_btree yields a clear message
    static_assert(!(has<policy::eytzinger_t> && has<policy::static_btree_t>),
                  "PolicyChain error: eytzinger and static_btree search layouts are mutually "
                  "exclusive.");

    static_assert(count_facet<search_layout_facet> <= 1,
                  "PolicyChain error: Conflicting search layout/strategy policies specified.");

    static_assert(count_facet<selection_guarantee_facet> <= 1,
                  "PolicyChain error: Conflicting selection guarantee policies specified.");

    static_assert(count_facet<selection_kernel_facet> <= 1,
                  "PolicyChain error: Conflicting selection kernel overrides specified.");

    // Semantic cross-facet exclusivity checks
    static_assert(!(has<policy::force_radix_t> && has<policy::stable_t> && has<policy::in_place_t>),
                  "PolicyChain error: Stable radix sort requires an auxiliary buffer and cannot be "
                  "combined with in_place.");

    static_assert(!(has<policy::nearly_sorted_t> && has<policy::force_radix_t>),
                  "PolicyChain error: nearly_sorted and force_radix policies are mutually "
                  "exclusive.");

    static_assert(!(has<policy::floyd_rivest_t> && has<policy::branchless_t>),
                  "PolicyChain error: floyd_rivest and branchless BlockQuicksort policies are "
                  "mutually exclusive.");

    static_assert(!(has<policy::floyd_rivest_t> && has<policy::introselect_t>),
                  "PolicyChain error: floyd_rivest and introselect policies are mutually "
                  "exclusive.");

    static_assert(!(has<policy::branchless_t> && has<policy::static_btree_t>),
                  "PolicyChain error: branchless and static_btree policies are mutually "
                  "exclusive.");

    static_assert(!(has<policy::branchless_t> && has<policy::galloping_t>),
                  "PolicyChain error: branchless and galloping policies are mutually exclusive.");

    /**
     * @brief Normalizes all tags in the chain into a descriptor of type Descriptor at compile time.
     * @tparam Descriptor Target structural descriptor type.
     * @return Evaluated descriptor.
     */
    template <typename Descriptor>
    [[nodiscard]] static constexpr Descriptor to_descriptor_as() noexcept {
        static_assert(
            applicable_to<Descriptor>,
            "PolicyChain error: Chain contains tags incompatible with target descriptor.");
        Descriptor desc{};
        if constexpr (applicable_to<Descriptor>) {
            (Tags::apply(desc), ...);
        }
        return desc;
    }

    /**
     * @brief Normalizes all tags in the chain into a structural PolicyDescriptor at compile time.
     * @return Evaluated PolicyDescriptor.
     */
    [[nodiscard]] static constexpr PolicyDescriptor to_descriptor() noexcept {
        return to_descriptor_as<PolicyDescriptor>();
    }

    /**
     * @brief Normalizes all tags in the chain into a structural SearchPolicyDescriptor at compile
     * time.
     * @return Evaluated SearchPolicyDescriptor.
     */
    [[nodiscard]] static constexpr SearchPolicyDescriptor to_search_descriptor() noexcept {
        return to_descriptor_as<SearchPolicyDescriptor>();
    }

    /**
     * @brief Normalizes all tags in the chain into a structural SelectPolicyDescriptor at compile
     * time.
     * @return Evaluated SelectPolicyDescriptor.
     */
    [[nodiscard]] static constexpr SelectPolicyDescriptor to_select_descriptor() noexcept {
        return to_descriptor_as<SelectPolicyDescriptor>();
    }

    /**
     * @brief Implicit conversion operator enabling direct usage as a sorting NTTP.
     */
    [[nodiscard]] constexpr operator PolicyDescriptor() const noexcept
        requires(applicable_to<PolicyDescriptor>)
    {
        return to_descriptor();
    }

    /**
     * @brief Implicit conversion operator enabling direct usage as a searching NTTP.
     */
    [[nodiscard]] constexpr operator SearchPolicyDescriptor() const noexcept
        requires(applicable_to<SearchPolicyDescriptor>)
    {
        return to_search_descriptor();
    }

    /**
     * @brief Implicit conversion operator enabling direct usage as a selection NTTP.
     */
    [[nodiscard]] constexpr operator SelectPolicyDescriptor() const noexcept
        requires(applicable_to<SelectPolicyDescriptor>)
    {
        return to_select_descriptor();
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
