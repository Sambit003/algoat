/**
 * @file aho_corasick.hpp
 * @brief High-performance Aho-Corasick multi-pattern automaton with Double-Array Trie (DAT)
 * and AVX2 root-level character dispatch.
 *
 * Implements:
 * - Aho & Corasick (CACM 1975) multi-pattern matching DFA with failure and output links.
 * - Jun-ichi Aoe (IEEE 1989) & Susumu Yata (2007) Double-Array Trie (BASE & CHECK).
 * - Wojciech Muła & Daniel Lemire (2021) SIMD root-bypass character skipping.
 */

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#endif
#include <optional>
#include <queue>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace algoat::dp {

/// \brief A match found by the automaton.
struct AcMatch {
    size_t end_pos;    ///< End position (exclusive) in text
    size_t pattern_id; ///< Index of the matched keyword (from add_keyword order)
    size_t length;     ///< Length of the matched keyword
};

/// \brief Aho-Corasick multi-pattern string search engine.
/// \tparam CharT Character type (uint8_t for binary, char for ASCII text)
template <typename CharT = char> class AhoCorasick {
    static_assert(sizeof(CharT) == 1,
                  "AhoCorasick requires 1-byte character types (char, uint8_t, int8_t, std::byte)");

public:
    AhoCorasick() = default;

    AhoCorasick(const AhoCorasick&) = default;
    AhoCorasick(AhoCorasick&&) noexcept = default;
    AhoCorasick& operator=(const AhoCorasick&) = default;
    AhoCorasick& operator=(AhoCorasick&&) noexcept = default;
    ~AhoCorasick() = default;

    /// \brief Adds a keyword to the automaton. Must be called before build().
    /// \param keyword The pattern to match.
    /// \param id Optional external identifier; defaults to sequential insertion index.
    void add_keyword(std::basic_string_view<CharT> keyword,
                     std::optional<size_t> id = std::nullopt) {
        size_t pattern_id = id.value_or(num_keywords_);
        raw_keywords_.push_back(
            RawKeyword{std::basic_string<CharT>(keyword), pattern_id, keyword.size()});
        keyword_lengths_.push_back(keyword.size());
        ++num_keywords_;
    }

    /// \brief Finalizes the DFA: constructs failure links and Double-Array Trie layout.
    /// \details BFS-based failure link construction: O(sum |keywords|).
    ///          Double-Array construction uses the DARTS algorithm.
    void build() {
        build_double_array();
        build_failure_links();
        init_root_filter();
    }

    /// \brief Searches text and returns all non-overlapping matches.
    [[nodiscard]] std::vector<AcMatch> find_all(std::basic_string_view<CharT> text) const {
        std::vector<AcMatch> matches;
        if (num_keywords_ == 0 || text.empty()) {
            return matches;
        }

        const size_t n = text.size();
        size_t pos = 0;
        int32_t state = 0;
        size_t last_end = 0;
        const int32_t* const base = base_.data();
        const int32_t* const check = check_.data();
        const int32_t* const fail = fail_.data();
        const int32_t* const output = output_.data();
        const int32_t* const root_trans = root_transitions_.data();
        const auto* const out_table = output_table_.data();

        while (pos < n) {
            // SIMD root bypass when in state 0
            if (state == 0 && can_simd_skip() && pos + 32 <= n) {
                size_t skipped = simd_skip_root(text.data() + pos, n - pos);
                pos += skipped;
                if (pos >= n) {
                    break;
                }
            }

            unsigned char c = static_cast<unsigned char>(text[pos]);
            state = step_state(state, c, base, check, fail, root_trans);
            ++pos;

            if (state != 0 && output[state] != -1) {
                int32_t out_idx = output[state];
                const OutputEntry* best_entry = nullptr;

                while (out_idx != -1) {
                    const auto& entry = out_table[out_idx];
                    size_t start = pos - entry.pattern_length;
                    if (start >= last_end) {
                        if (best_entry == nullptr ||
                            entry.pattern_length > best_entry->pattern_length) {
                            best_entry = &entry;
                        }
                    }
                    out_idx = entry.next_output;
                }

                if (best_entry != nullptr) {
                    matches.push_back(
                        AcMatch{pos, best_entry->pattern_id, best_entry->pattern_length});
                    last_end = pos;
                    state = 0; // Reset state for standard leftmost non-overlapping search
                }
            }
        }

        return matches;
    }

    /// \brief Searches text and returns the first match found.
    [[nodiscard]] std::optional<AcMatch> find_first(std::basic_string_view<CharT> text) const {
        if (num_keywords_ == 0 || text.empty()) {
            return std::nullopt;
        }

        const size_t n = text.size();
        size_t pos = 0;
        int32_t state = 0;
        const int32_t* const base = base_.data();
        const int32_t* const check = check_.data();
        const int32_t* const fail = fail_.data();
        const int32_t* const output = output_.data();
        const int32_t* const root_trans = root_transitions_.data();
        const auto* const out_table = output_table_.data();

        while (pos < n) {
            if (state == 0 && can_simd_skip() && pos + 32 <= n) {
                size_t skipped = simd_skip_root(text.data() + pos, n - pos);
                pos += skipped;
                if (pos >= n) {
                    break;
                }
            }

            unsigned char c = static_cast<unsigned char>(text[pos]);
            state = step_state(state, c, base, check, fail, root_trans);
            ++pos;

            if (state != 0 && output[state] != -1) {
                int32_t out_idx = output[state];
                const auto& entry = out_table[out_idx];
                return AcMatch{pos, entry.pattern_id, entry.pattern_length};
            }
        }

        return std::nullopt;
    }

    /// \brief Streams matches to a callback (avoids vector allocation).
    void find_all_callback(std::basic_string_view<CharT> text,
                           std::function<bool(AcMatch)> callback) const {
        if (num_keywords_ == 0 || text.empty()) {
            return;
        }

        const size_t n = text.size();
        size_t pos = 0;
        int32_t state = 0;
        const int32_t* const base = base_.data();
        const int32_t* const check = check_.data();
        const int32_t* const fail = fail_.data();
        const int32_t* const output = output_.data();
        const int32_t* const root_trans = root_transitions_.data();
        const auto* const out_table = output_table_.data();

        while (pos < n) {
            if (state == 0 && can_simd_skip() && pos + 32 <= n) {
                size_t skipped = simd_skip_root(text.data() + pos, n - pos);
                pos += skipped;
                if (pos >= n) {
                    break;
                }
            }

            unsigned char c = static_cast<unsigned char>(text[pos]);
            state = step_state(state, c, base, check, fail, root_trans);
            ++pos;

            if (state != 0 && output[state] != -1) {
                int32_t out_idx = output[state];
                while (out_idx != -1) {
                    const auto& entry = out_table[out_idx];
                    if (!callback(AcMatch{pos, entry.pattern_id, entry.pattern_length})) {
                        return; // Early stop requested by callback
                    }
                    out_idx = entry.next_output;
                }
            }
        }
    }

    [[nodiscard]] size_t num_keywords() const noexcept {
        return num_keywords_;
    }

    [[nodiscard]] size_t num_states() const noexcept {
        return base_.size();
    }

    [[nodiscard]] size_t memory_usage_bytes() const noexcept {
        size_t total = sizeof(*this);
        total += base_.capacity() * sizeof(int32_t);
        total += check_.capacity() * sizeof(int32_t);
        total += fail_.capacity() * sizeof(int32_t);
        total += output_.capacity() * sizeof(int32_t);
        total += output_table_.capacity() * sizeof(OutputEntry);
        total += keyword_lengths_.capacity() * sizeof(size_t);
        total += root_start_chars_.capacity() * sizeof(unsigned char);
        return total;
    }

private:
    static constexpr size_t kAlphabetSize = 256;

    // Double-Array Trie storage
    std::vector<int32_t> base_;   ///< BASE[state]: offset for child lookup
    std::vector<int32_t> check_;  ///< CHECK[state]: parent state for validity
    std::vector<int32_t> fail_;   ///< Failure function (Aho-Corasick fail links)
    std::vector<int32_t> output_; ///< Output link index (-1 = no output)

    // Output table: maps state → list of pattern IDs matching at this state
    struct OutputEntry {
        size_t pattern_id;
        size_t pattern_length;
        int32_t next_output; ///< Linked list of outputs
    };
    std::vector<OutputEntry> output_table_;

    // Keyword metadata
    std::vector<size_t> keyword_lengths_;
    size_t num_keywords_{0};

    // Temporary storage for keywords prior to build()
    struct RawKeyword {
        std::basic_string<CharT> pattern;
        size_t id;
        size_t length;
    };
    std::vector<RawKeyword> raw_keywords_;

    // Root-level filter acceleration
    std::array<int32_t, 256> root_transitions_{};
    std::vector<unsigned char> root_start_chars_;
    size_t root_char_count_{0};

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) &&           \
    defined(__AVX2__)
    // Truffle 256-bit character class lookup tables (Muła & Lemire 2021)
    // truffle_lut_hc_: 16-byte nibble LUT for low-half ASCII characters (0 <= c < 128, bit 7 == 0)
    alignas(32) std::array<uint8_t, 32> truffle_lut_hc_{};
    // truffle_lut_hs_: 16-byte nibble LUT for high-half binary characters (128 <= c < 256, bit 7 ==
    // 1)
    alignas(32) std::array<uint8_t, 32> truffle_lut_hs_{};
#endif

    template <typename BasePtr, typename CheckPtr, typename FailPtr, typename RootPtr>
    [[nodiscard]] static inline int32_t step_state(int32_t state, unsigned char c, BasePtr base,
                                                   CheckPtr check, FailPtr fail,
                                                   RootPtr root_trans) noexcept {
        if (state == 0) {
            return root_trans[c];
        }
        int32_t next = base[state] + static_cast<int32_t>(c);
        if (check[next] == state) {
            return next;
        }
        state = fail[state];
        while (state > 0) {
            next = base[state] + static_cast<int32_t>(c);
            if (check[next] == state) {
                return next;
            }
            state = fail[state];
        }
        return root_trans[c];
    }

    // Intermediate trie node representation used only during build()
    struct IntermediateNode {
        std::vector<std::pair<unsigned char, int32_t>> children;
        std::vector<size_t> pattern_ids;
    };

    void build_double_array() {
        if (raw_keywords_.empty()) {
            base_.assign(1, 0);
            check_.assign(1, -1);
            fail_.assign(1, 0);
            output_.assign(1, -1);
            return;
        }

        // 1. Build intermediate trie from raw keywords
        std::vector<IntermediateNode> trie;
        trie.emplace_back(); // Root at node 0

        for (size_t pid = 0; pid < raw_keywords_.size(); ++pid) {
            const auto& kw_info = raw_keywords_[pid];
            int32_t curr = 0;
            for (CharT ch : kw_info.pattern) {
                unsigned char c = static_cast<unsigned char>(ch);
                auto it = std::find_if(trie[curr].children.begin(), trie[curr].children.end(),
                                       [c](const auto& pair) { return pair.first == c; });
                if (it != trie[curr].children.end()) {
                    curr = it->second;
                } else {
                    int32_t next_id = static_cast<int32_t>(trie.size());
                    trie.emplace_back();
                    trie[curr].children.emplace_back(c, next_id);
                    // Maintain sorted order by character code for deterministic packing
                    std::sort(trie[curr].children.begin(), trie[curr].children.end(),
                              [](const auto& a, const auto& b) { return a.first < b.first; });
                    curr = next_id;
                }
            }
            trie[curr].pattern_ids.push_back(pid);
        }

        // 2. DARTS greedy packing into flat BASE and CHECK arrays
        std::vector<int32_t> trie_to_dat(trie.size(), -1);
        trie_to_dat[0] = 0;

        base_.assign(1, 0);
        check_.assign(1, -1);

        std::vector<bool> used_base;
        size_t next_check_pos = 1;
        size_t first_free_base = 1;

        std::queue<int32_t> q;
        q.push(0);

        while (!q.empty()) {
            int32_t u = q.front();
            q.pop();

            int32_t s = trie_to_dat[u];
            const auto& children = trie[u].children;

            if (children.empty()) {
                base_[s] = 0;
                continue;
            }

            unsigned char first_code = children[0].first;
            size_t pos = std::max({static_cast<size_t>(first_code) + 1, next_check_pos,
                                   first_free_base + first_code});
            int32_t b = -1;

            while (true) {
                while (pos < check_.size() && check_[pos] != -1) {
                    ++pos;
                }
                int32_t cand_b = static_cast<int32_t>(pos - first_code);
                if (cand_b <= 0) {
                    ++pos;
                    continue;
                }
                if (static_cast<size_t>(cand_b) < first_free_base) {
                    pos = first_free_base + first_code;
                    continue;
                }
                if (cand_b < static_cast<int32_t>(used_base.size()) && used_base[cand_b]) {
                    ++pos;
                    continue;
                }

                bool conflict = false;
                for (const auto& [code, child_u] : children) {
                    size_t target = static_cast<size_t>(cand_b + code);
                    if (target < check_.size() && check_[target] != -1) {
                        conflict = true;
                        break;
                    }
                }

                if (!conflict) {
                    b = cand_b;
                    break;
                }
                ++pos;
            }

            if (static_cast<size_t>(b) >= used_base.size()) {
                used_base.resize(static_cast<size_t>(b) + 1, false);
            }
            used_base[static_cast<size_t>(b)] = true;
            while (first_free_base < used_base.size() && used_base[first_free_base]) {
                ++first_free_base;
            }

            if (static_cast<size_t>(s) >= base_.size()) {
                base_.resize(static_cast<size_t>(s) + 1, 0);
            }
            base_[s] = b;

            for (const auto& [code, child_u] : children) {
                size_t target = static_cast<size_t>(b + code);
                if (target >= check_.size()) {
                    check_.resize(target + 1, -1);
                    base_.resize(target + 1, 0);
                }
                check_[target] = s;
                trie_to_dat[child_u] = static_cast<int32_t>(target);
                q.push(child_u);
            }

            while (next_check_pos < check_.size() && check_[next_check_pos] != -1) {
                ++next_check_pos;
            }
        }

        size_t total_states = std::max(base_.size(), check_.size());
        size_t max_req = 256;
        for (int32_t b : base_) {
            if (b > 0) {
                max_req = std::max(max_req, static_cast<size_t>(b) + 256);
            }
        }
        total_states = std::max(total_states, max_req);
        base_.resize(total_states, 0);
        check_.resize(total_states, -1);
        fail_.assign(total_states, 0);
        output_.assign(total_states, -1);

        // Populate direct pattern termination output entries
        for (size_t u = 0; u < trie.size(); ++u) {
            int32_t s = trie_to_dat[u];
            if (s != -1 && !trie[u].pattern_ids.empty()) {
                for (size_t raw_idx : trie[u].pattern_ids) {
                    const auto& kw = raw_keywords_[raw_idx];
                    int32_t entry_idx = static_cast<int32_t>(output_table_.size());
                    output_table_.push_back(OutputEntry{kw.id, kw.length, output_[s]});
                    output_[s] = entry_idx;
                }
            }
        }

        // Free temporary raw keyword memory
        raw_keywords_.clear();
        raw_keywords_.shrink_to_fit();
    }

    void build_failure_links() {
        if (base_.empty()) {
            return;
        }

        std::queue<int32_t> q;
        fail_[0] = 0;

        // BFS depth 1: direct children of root (state 0)
        for (size_t c = 0; c < kAlphabetSize; ++c) {
            unsigned char uc = static_cast<unsigned char>(c);
            if (is_valid_state(0, uc)) {
                int32_t child = base_[0] + static_cast<int32_t>(uc);
                fail_[child] = 0;
                // Append output link chain from root (state 0) if root has outputs
                int32_t f_out = output_[0];
                if (f_out != -1) {
                    if (output_[child] == -1) {
                        output_[child] = f_out;
                    } else {
                        int32_t cur = output_[child];
                        while (output_table_[cur].next_output != -1) {
                            cur = output_table_[cur].next_output;
                        }
                        output_table_[cur].next_output = f_out;
                    }
                }
                q.push(child);
            }
        }

        // BFS depth >= 2
        while (!q.empty()) {
            int32_t r = q.front();
            q.pop();

            for (size_t c = 0; c < kAlphabetSize; ++c) {
                unsigned char uc = static_cast<unsigned char>(c);
                if (is_valid_state(r, uc)) {
                    int32_t u = base_[r] + static_cast<int32_t>(uc);
                    q.push(u);

                    int32_t state = fail_[r];
                    while (state != 0 && !is_valid_state(state, uc)) {
                        state = fail_[state];
                    }

                    if (is_valid_state(state, uc)) {
                        fail_[u] = base_[state] + static_cast<int32_t>(uc);
                    } else {
                        fail_[u] = 0;
                    }

                    // Append output link chain from fail_[u] to u
                    int32_t f_out = output_[fail_[u]];
                    if (f_out != -1) {
                        if (output_[u] == -1) {
                            output_[u] = f_out;
                        } else {
                            int32_t cur = output_[u];
                            while (output_table_[cur].next_output != -1) {
                                cur = output_table_[cur].next_output;
                            }
                            output_table_[cur].next_output = f_out;
                        }
                    }
                }
            }
        }

        // Shrink vectors to reclaim excess dynamic allocation
        base_.shrink_to_fit();
        check_.shrink_to_fit();
        fail_.shrink_to_fit();
        output_.shrink_to_fit();
        output_table_.shrink_to_fit();
        keyword_lengths_.shrink_to_fit();
    }

    void init_root_filter() {
        root_transitions_.fill(0);
        root_start_chars_.clear();
        root_char_count_ = 0;

        for (size_t c = 0; c < kAlphabetSize; ++c) {
            unsigned char uc = static_cast<unsigned char>(c);
            if (is_valid_state(0, uc)) {
                root_transitions_[c] = base_[0] + static_cast<int32_t>(uc);
                root_start_chars_.push_back(uc);
                ++root_char_count_;
            }
        }
        root_start_chars_.shrink_to_fit();

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) &&           \
    defined(__AVX2__)
        std::array<uint8_t, 16> lut_hc{};
        std::array<uint8_t, 16> lut_hs{};
        for (unsigned char uc : root_start_chars_) {
            uint8_t lo = uc & 0x0F;
            uint8_t hi = (uc >> 4) & 0x07;
            uint8_t bit = static_cast<uint8_t>(1u << hi);
            if (uc < 128) {
                lut_hc[lo] |= bit;
            } else {
                lut_hs[lo] |= bit;
            }
        }
        for (size_t i = 0; i < 16; ++i) {
            truffle_lut_hc_[i] = lut_hc[i];
            truffle_lut_hc_[i + 16] = lut_hc[i];
            truffle_lut_hs_[i] = lut_hs[i];
            truffle_lut_hs_[i + 16] = lut_hs[i];
        }
#endif
    }

    [[nodiscard]] bool can_simd_skip() const noexcept {
        return root_char_count_ > 0 && root_char_count_ < 240;
    }

    [[nodiscard]] size_t simd_skip_root(const CharT* ptr, size_t remaining) const noexcept {
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) &&           \
    defined(__AVX2__)
        size_t offset = 0;
        const size_t num_chars = root_char_count_;

        if (num_chars == 1) {
            __m256i t0 = _mm256_set1_epi8(static_cast<char>(root_start_chars_[0]));
            while (offset + 32 <= remaining) {
                __m256i chunk = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(reinterpret_cast<const char*>(ptr) + offset));
                __m256i eq = _mm256_cmpeq_epi8(chunk, t0);
                uint32_t mask = static_cast<uint32_t>(_mm256_movemask_epi8(eq));
                if (mask != 0) {
                    return offset + static_cast<size_t>(std::countr_zero(mask));
                }
                offset += 32;
            }
        } else if (num_chars == 2) {
            __m256i t0 = _mm256_set1_epi8(static_cast<char>(root_start_chars_[0]));
            __m256i t1 = _mm256_set1_epi8(static_cast<char>(root_start_chars_[1]));
            while (offset + 32 <= remaining) {
                __m256i chunk = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(reinterpret_cast<const char*>(ptr) + offset));
                __m256i eq0 = _mm256_cmpeq_epi8(chunk, t0);
                __m256i eq1 = _mm256_cmpeq_epi8(chunk, t1);
                __m256i hit = _mm256_or_si256(eq0, eq1);
                uint32_t mask = static_cast<uint32_t>(_mm256_movemask_epi8(hit));
                if (mask != 0) {
                    return offset + static_cast<size_t>(std::countr_zero(mask));
                }
                offset += 32;
            }
        } else if (num_chars <= 4) {
            __m256i t0 = _mm256_set1_epi8(static_cast<char>(root_start_chars_[0]));
            __m256i t1 = _mm256_set1_epi8(static_cast<char>(root_start_chars_[1]));
            __m256i t2 = _mm256_set1_epi8(static_cast<char>(root_start_chars_[2]));
            __m256i t3 =
                (num_chars > 3) ? _mm256_set1_epi8(static_cast<char>(root_start_chars_[3])) : t0;
            while (offset + 32 <= remaining) {
                __m256i chunk = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(reinterpret_cast<const char*>(ptr) + offset));
                __m256i eq0 = _mm256_cmpeq_epi8(chunk, t0);
                __m256i eq1 = _mm256_cmpeq_epi8(chunk, t1);
                __m256i eq2 = _mm256_cmpeq_epi8(chunk, t2);
                __m256i eq3 = _mm256_cmpeq_epi8(chunk, t3);
                __m256i hit = _mm256_or_si256(_mm256_or_si256(eq0, eq1), _mm256_or_si256(eq2, eq3));
                uint32_t mask = static_cast<uint32_t>(_mm256_movemask_epi8(hit));
                if (mask != 0) {
                    return offset + static_cast<size_t>(std::countr_zero(mask));
                }
                offset += 32;
            }
        } else {
            // Truffle 256-bit character class test across 32 bytes
            __m256i lut_hc =
                _mm256_load_si256(reinterpret_cast<const __m256i*>(truffle_lut_hc_.data()));
            __m256i lut_hs =
                _mm256_load_si256(reinterpret_cast<const __m256i*>(truffle_lut_hs_.data()));
            __m256i highconst = _mm256_set1_epi8(static_cast<char>(0x80));
            __m256i shuf_mask_hi = _mm256_set1_epi64x(0x8040201008040201ULL);

            while (offset + 32 <= remaining) {
                __m256i chunk = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(reinterpret_cast<const char*>(ptr) + offset));
                __m256i shuf1 = _mm256_shuffle_epi8(lut_hc, chunk);
                __m256i t1 = _mm256_xor_si256(chunk, highconst);
                __m256i shuf2 = _mm256_shuffle_epi8(lut_hs, t1);
                __m256i lo_bits = _mm256_or_si256(shuf1, shuf2);

                __m256i t2 = _mm256_and_si256(_mm256_srli_epi16(chunk, 4), _mm256_set1_epi8(0x07));
                __m256i hi_bit = _mm256_shuffle_epi8(shuf_mask_hi, t2);

                __m256i hit = _mm256_and_si256(lo_bits, hi_bit);
                __m256i eq0 = _mm256_cmpeq_epi8(hit, _mm256_setzero_si256());
                uint32_t mask = ~static_cast<uint32_t>(_mm256_movemask_epi8(eq0));
                if (mask != 0) {
                    return offset + static_cast<size_t>(std::countr_zero(mask));
                }
                offset += 32;
            }
        }
        return offset;
#else
        size_t offset = 0;
        while (offset < remaining) {
            unsigned char c = static_cast<unsigned char>(ptr[offset]);
            if (root_transitions_[c] != 0) {
                return offset;
            }
            ++offset;
        }
        return offset;
#endif
    }

    [[nodiscard]] inline int32_t transition(int32_t state, unsigned char c) const noexcept {
        return step_state(state, c, base_.data(), check_.data(), fail_.data(),
                          root_transitions_.data());
    }

    [[nodiscard]] bool is_valid_state(int32_t state, unsigned char c) const noexcept {
        if (state < 0 || static_cast<size_t>(state) >= base_.size()) {
            return false;
        }
        int32_t next = base_[state] + static_cast<int32_t>(c);
        if (next < 0 || static_cast<size_t>(next) >= check_.size()) {
            return false;
        }
        return check_[next] == state;
    }
};

extern template class AhoCorasick<char>;
extern template class AhoCorasick<uint8_t>;

} // namespace algoat::dp
