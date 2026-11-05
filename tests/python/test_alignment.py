import random
import pytest
from algoat import dp

pairwise2 = pytest.importorskip("Bio.pairwise2")


def test_scoring_matrices():
    b62 = dp.ScoringMatrix.blosum62()
    assert b62.score("A", "A") == 4
    assert b62.score("W", "W") == 11
    assert b62.score("A", "R") == -1
    assert b62.score("a", "A") == 4

    pam = dp.ScoringMatrix.pam250()
    assert pam.score("A", "A") == 2
    assert pam.score("W", "W") == 17

    dna = dp.ScoringMatrix.dna(2, -3)
    assert dna.score("A", "A") == 2
    assert dna.score("a", "A") == 2
    assert dna.score("A", "C") == -3


def test_gap_penalty():
    gap = dp.GapPenalty(-5, -1)
    assert gap.open == -5
    assert gap.extend == -1
    gap.open = -10
    gap.extend = -2
    assert gap.open == -10
    assert gap.extend == -2


def test_needleman_wunsch_basics():
    m = dp.ScoringMatrix.blosum62()
    gap = dp.GapPenalty(-5, -1)

    empty_res = dp.needleman_wunsch("", "", m, gap)
    assert empty_res.score == 0
    assert empty_res.cigar == ""

    s = "HEAGAWGHEE"
    id_res = dp.needleman_wunsch(s, s, m, dp.GapPenalty(-10, -1))
    assert id_res.score > 0
    assert id_res.cigar == "10M"
    assert id_res.aligned_query == s
    assert id_res.aligned_ref == s

    dna_m = dp.ScoringMatrix.dna(2, -3)
    dna_res = dp.needleman_wunsch("ACG", "AG", dna_m, dp.GapPenalty(-5, -2))
    assert dna_res.score == -1
    assert dna_res.cigar == "1M1I1M"
    assert dna_res.aligned_query == "ACG"
    assert dna_res.aligned_ref == "A-G"


def test_smith_waterman_basics():
    m = dp.ScoringMatrix.blosum62()
    gap = dp.GapPenalty(-10, -2)

    query = "DDWHEAEDD"
    ref = "WWWWHEAEWWWW"

    res = dp.smith_waterman(query, ref, m, gap)
    assert res.score > 0
    assert res.cigar == "5M"
    assert res.aligned_query == "WHEAE"
    assert res.aligned_ref == "WHEAE"
    assert res.query_start == 2
    assert res.query_end == 7
    assert res.ref_start == 3
    assert res.ref_end == 8


def test_smith_waterman_score_and_batch():
    m = dp.ScoringMatrix.blosum62()
    gap = dp.GapPenalty(-5, -1)

    q = "HEAGAWGHEEWWWWHEAGAWGHEE"
    targets = [
        "PAWHEAEWHEAGAWGHEEWWWWHEAGAWGHEEPAW",
        "NOTHINGINCOMMONATALL",
        "HEAGAWGHEEWWWWHEAGAWGHEE",
        "",
    ]

    single_scores = [dp.smith_waterman_score(q, t, m, gap) for t in targets]
    batch_scores = dp.smith_waterman_batch(q, targets, m, gap)
    assert batch_scores == single_scores


def test_biopython_global_alignment_100_pairs():
    random.seed(42)
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    m = dp.ScoringMatrix(2, -1)
    gap = dp.GapPenalty(-5, -1)

    for trial in range(100):
        len_q = random.randint(10, 35)
        len_r = random.randint(10, 35)
        q = "".join(random.choices(amino_acids, k=len_q))
        r = "".join(random.choices(amino_acids, k=len_r))

        my_res = dp.needleman_wunsch(q, r, m, gap)
        bio_score = int(pairwise2.align.globalms(q, r, 2, -1, -5, -1, score_only=True))
        assert my_res.score == bio_score, (
            f"Trial {trial}: score mismatch my={my_res.score}, bio={bio_score}"
        )


def test_biopython_local_alignment_100_pairs():
    random.seed(123)
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    m = dp.ScoringMatrix(2, -1)
    gap = dp.GapPenalty(-5, -1)

    for trial in range(100):
        len_q = random.randint(16, 45)
        len_r = random.randint(16, 45)
        q = "".join(random.choices(amino_acids, k=len_q))
        r = "".join(random.choices(amino_acids, k=len_r))

        my_score = dp.smith_waterman_score(q, r, m, gap)
        bio_al = pairwise2.align.localms(q, r, 2, -1, -5, -1, score_only=True)
        bio_score = int(bio_al) if bio_al else 0
        assert abs(my_score - bio_score) <= 1, (
            f"Trial {trial}: SW score mismatch my={my_score}, bio={bio_score}"
        )
