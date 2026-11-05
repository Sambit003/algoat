from algoat import dp


def test_levenshtein_distance_basics():
    assert dp.levenshtein_distance("", "") == 0
    assert dp.levenshtein_distance("", "test") == 4
    assert dp.levenshtein_distance("test", "") == 4
    assert dp.levenshtein_distance("kitten", "sitting") == 3
    assert dp.levenshtein_distance("flaw", "lawn") == 2
    assert dp.levenshtein_distance("saturday", "sunday") == 3


def test_levenshtein_within_k():
    assert dp.levenshtein_within_k("kitten", "sitting", 1) == 2
    assert dp.levenshtein_within_k("kitten", "sitting", 2) == 3
    assert dp.levenshtein_within_k("kitten", "sitting", 3) == 3
    assert dp.levenshtein_within_k("kitten", "sitting", 5) == 3
    assert dp.levenshtein_within_k("abc", "abcdefg", 2) == 3
    assert dp.levenshtein_within_k("abc", "abc", 0) == 0
    assert dp.levenshtein_within_k("abc", "abd", 0) == 1


def test_levenshtein_align():
    alignment = dp.levenshtein_align("kitten", "sitting")
    assert alignment.distance == 3
    ops = alignment.operations

    # Verify that operations transform pattern to text
    i, j, cost = 0, 0, 0
    pattern = "kitten"
    text = "sitting"
    reconstructed = []

    for op in ops:
        if op == dp.EditOp.Match:
            assert pattern[i] == text[j]
            reconstructed.append(pattern[i])
            i += 1
            j += 1
        elif op == dp.EditOp.Substitute:
            assert pattern[i] != text[j]
            reconstructed.append(text[j])
            i += 1
            j += 1
            cost += 1
        elif op == dp.EditOp.Insert:
            reconstructed.append(text[j])
            j += 1
            cost += 1
        elif op == dp.EditOp.Delete:
            i += 1
            cost += 1

    assert i == len(pattern)
    assert j == len(text)
    assert "".join(reconstructed) == text
    assert cost == 3


def test_lcs_length():
    assert dp.lcs_length("", "") == 0
    assert dp.lcs_length("abc", "") == 0
    assert dp.lcs_length("AGGTAB", "GXTXAYB") == 4
    assert dp.lcs_length("ABCBDAB", "BDCAB") == 4
    assert dp.lcs_length("abcdef", "abcdef") == 6


def test_levenshtein_batch():
    pattern = "kitten"
    texts = ["sitting", "kitten", "mitten", "kit", "kitchen"]
    expected = [dp.levenshtein_distance(pattern, t) for t in texts]

    batch_res = dp.levenshtein_batch(pattern, texts)
    assert batch_res == expected
