"""
Tests for the seqdesign library.

Modules covered:
  seqgen.py                     — pure-Python sequence enumeration
  create_same_tm_seq_library.py — seq_check, seqgen, NUPACK Tm helpers
  sequence_set_designer.py      — check_seq_valid, is_rc, name_rc
  sequence_optimization_functions.py — add_spacers, set_column_complement_to_inf, load_ensemble

Two bugs were found and fixed during test authoring:
  nth_combination() used `index %= choose` which collapsed distinct indices onto
  the same remainder, producing duplicate combinations.
  random_number_generator() yielded `remaining + 1` on collision without tracking
  that value, causing duplicates across the full sequence.
Both are fixed in seqgen.py using the combinatorial number system and online
Fisher-Yates respectively.
"""
import itertools
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oxpy_utils.structure_editor.dna_structure import rc
from oxpy_utils.seqdesign.seqgen import (
    binary_to_nucleotides,
    count_sequences,
    generate_unique_sequence,
    get_seq_identifier_maxs,
    nth_combination,
    random_number_generator,
)
from oxpy_utils.seqdesign.create_same_tm_seq_library import (
    compute_melting_temp,
    create_tube,
    get_unbound_fraction,
    melting_temp_screen,
    seq_check,
    seqgen,
)
from oxpy_utils.seqdesign.sequence_set_designer import (
    check_seq_valid,
    is_rc,
    name_rc,
)
from oxpy_utils.seqdesign.sequence_optimization_functions import (
    add_spacers,
    load_ensemble,
    set_column_complement_to_inf,
)


# ---------------------------------------------------------------------------
# seqgen.py — combinatorial enumeration
# ---------------------------------------------------------------------------

class TestCountSequences:
    def test_known_values(self):
        # C(4,2) * 2^2 * 2^2 = 6 * 4 * 4 = 96
        assert count_sequences(4, 2) == 96
        # C(2,1) * 2 * 2 = 8
        assert count_sequences(2, 1) == 8
        # C(8,4) * 2^4 * 2^4 = 70 * 256 = 17920
        assert count_sequences(8, 4) == 70 * 16 * 16

    def test_all_gc(self):
        # All G/C: C(n,n)*2^n*2^0 = 1 * 2^n * 1 = 2^n
        assert count_sequences(4, 4) == 2 ** 4

    def test_no_gc(self):
        # All A/T: C(n,0)*2^0*2^n = 1 * 1 * 2^n = 2^n
        assert count_sequences(3, 0) == 2 ** 3

    def test_matches_get_seq_identifier_maxs(self):
        for length, gc in [(4, 2), (6, 3), (8, 4)]:
            ways, n_gc, n_at = get_seq_identifier_maxs(length, gc)
            assert count_sequences(length, gc) == ways * n_gc * n_at


class TestGetSeqIdentifierMaxs:
    def test_known_values(self):
        assert get_seq_identifier_maxs(4, 2) == (comb(4, 2), 4, 4)
        assert get_seq_identifier_maxs(8, 4) == (70, 16, 16)
        assert get_seq_identifier_maxs(2, 0) == (1, 1, 4)

    def test_gc_order_is_power_of_two(self):
        for gc in range(5):
            _, n_gc, _ = get_seq_identifier_maxs(8, gc)
            assert n_gc == 2 ** gc

    def test_at_order_is_power_of_two(self):
        for gc in range(5):
            length = 8
            _, _, n_at = get_seq_identifier_maxs(length, gc)
            assert n_at == 2 ** (length - gc)


class TestNthCombination:
    def test_first_combination(self):
        combo = nth_combination([0, 1, 2, 3], 2, 0)
        assert len(combo) == 2
        assert combo.issubset({0, 1, 2, 3})

    def test_all_combinations_are_distinct(self):
        pool = list(range(5))
        r = 2
        n = comb(5, 2)
        combos = [nth_combination(pool, r, i) for i in range(n)]
        assert len(set(combos)) == n

    def test_covers_all_combinations(self):
        from itertools import combinations
        pool = list(range(4))
        r = 2
        n = comb(4, 2)
        expected = {frozenset(c) for c in combinations(pool, r)}
        got = {nth_combination(pool, r, i) for i in range(n)}
        assert got == expected

    def test_result_length_equals_r(self):
        assert len(nth_combination([0, 1, 2, 3, 4], 3, 5)) == 3


class TestBinaryToNucleotides:
    def test_all_zeros(self):
        assert list(binary_to_nucleotides(0, 3, "G", "C")) == ["C", "C", "C"]

    def test_all_ones(self):
        assert list(binary_to_nucleotides(7, 3, "G", "C")) == ["G", "G", "G"]

    def test_mixed(self):
        # binary 5 = 101 → G, C, G
        assert list(binary_to_nucleotides(5, 3, "G", "C")) == ["G", "C", "G"]

    def test_output_length_equals_ndigits(self):
        result = list(binary_to_nucleotides(3, 6, "A", "T"))
        assert len(result) == 6

    def test_only_opt1_and_opt2_in_output(self):
        result = list(binary_to_nucleotides(42, 8, "A", "T"))
        assert all(c in ("A", "T") for c in result)

    def test_at_options(self):
        assert list(binary_to_nucleotides(0, 2, "A", "T")) == ["T", "T"]
        assert list(binary_to_nucleotides(3, 2, "A", "T")) == ["A", "A"]


class TestRandomNumberGenerator:
    def test_yields_n_values(self):
        vals = list(random_number_generator(10))
        assert len(vals) == 10

    def test_all_values_in_range(self):
        n = 15
        for v in random_number_generator(n):
            assert 1 <= v <= n

    def test_is_a_permutation(self):
        n = 20
        vals = list(random_number_generator(n))
        assert sorted(vals) == list(range(1, n + 1))

    def test_n_equals_one(self):
        vals = list(random_number_generator(1))
        assert vals == [1]

    def test_early_termination_does_not_repeat(self):
        # Consume only half — the first half should still be distinct
        n = 100
        gen = random_number_generator(n)
        partial = [next(gen) for _ in range(n // 2)]
        assert len(set(partial)) == len(partial)


class TestGenerateUniqueSequence:
    def test_output_length(self):
        for seq in itertools.islice(generate_unique_sequence(6, 3), 20):
            assert len(seq) == 6

    def test_gc_content(self):
        for seq in itertools.islice(generate_unique_sequence(8, 4), 20):
            gc_count = sum(1 for c in seq if c in "GC")
            assert gc_count == 4

    def test_only_valid_bases(self):
        for seq in itertools.islice(generate_unique_sequence(6, 2), 20):
            assert all(c in "ACGT" for c in seq)

    def test_raises_on_gc_count_exceeds_length(self):
        with pytest.raises(ValueError):
            next(generate_unique_sequence(4, 5))

    def test_raises_on_negative_args(self):
        with pytest.raises(ValueError):
            next(generate_unique_sequence(-1, 0))
        with pytest.raises(ValueError):
            next(generate_unique_sequence(4, -1))

    def test_raises_custom_exception_when_exhausted(self):
        # count_sequences(2, 1) = 8 — take all, next call should raise
        gen = generate_unique_sequence(2, 1)
        list(itertools.islice(gen, count_sequences(2, 1)))
        with pytest.raises(Exception, match="Have generated all possible sequences"):
            next(gen)


# ---------------------------------------------------------------------------
# create_same_tm_seq_library.py — seq filtering
# ---------------------------------------------------------------------------

class TestSeqCheck:
    def test_valid_sequence(self):
        assert seq_check("ATGCATGC") is True

    def test_triple_repeat_rejected(self):
        assert seq_check("AAATGC") is False
        assert seq_check("TTTATGC") is False
        assert seq_check("GGGTAGC") is False
        assert seq_check("CCCATGC") is False

    def test_missing_base_rejected(self):
        assert seq_check("AAATTTGG") is False  # no C
        assert seq_check("GCGCGCGC") is False  # no A or T

    def test_all_four_bases_required(self):
        assert seq_check("AGTCAGTC") is True


class TestSeqgen:
    def test_returns_set(self):
        result = seqgen(6, 3, 10)
        assert isinstance(result, set)

    def test_size_capped_at_total_possible(self):
        total = count_sequences(4, 2)
        result = seqgen(4, 2, total + 100)
        assert len(result) <= total

    def test_each_seq_has_correct_length(self):
        for seq in seqgen(6, 3, 10):
            assert len(seq) == 6

    def test_each_seq_has_correct_gc_count(self):
        for seq in seqgen(6, 3, 10):
            assert sum(1 for c in seq if c in "GC") == 3


# ---------------------------------------------------------------------------
# sequence_set_designer.py — pure string helpers
# ---------------------------------------------------------------------------

class TestCheckSeqValid:
    def test_valid_sequence(self):
        assert check_seq_valid("ATGCATGC") is True

    def test_triple_base_rejected(self):
        assert check_seq_valid("AAATGCTG") is False
        assert check_seq_valid("GCTTTTAG") is False

    def test_missing_base_rejected(self):
        assert check_seq_valid("ATATATATAT") is False  # no G or C

    def test_custom_min_polybase(self):
        # "AAAT" has 3 consecutive A → rejected at min=3, allowed at min=4
        assert check_seq_valid("AAATGCTG", min_polybase=3) is False
        assert check_seq_valid("AAATGCTG", min_polybase=4) is True

    def test_all_four_bases_required(self):
        assert check_seq_valid("AGTCAGTC") is True


class TestIsRc:
    def test_leading_dash(self):
        assert is_rc("-abc") is True

    def test_trailing_prime(self):
        assert is_rc("abc'") is True

    def test_plain_name(self):
        assert is_rc("abc") is False
        assert is_rc("domain1") is False

    def test_empty_string(self):
        # empty string has neither prefix/suffix
        assert is_rc("") is False


class TestNameRc:
    def test_forward_to_rc(self):
        assert name_rc("abc") == "abc'"

    def test_prime_rc_to_forward(self):
        assert name_rc("abc'") == "abc"

    def test_dash_rc_to_forward(self):
        assert name_rc("-abc") == "abc"

    def test_roundtrip_prime(self):
        name = "domain1"
        assert name_rc(name_rc(name)) == name

    def test_roundtrip_dash(self):
        # dash → plain, then plain → prime; not a perfect roundtrip due to mixed conventions
        assert not is_rc(name_rc("-abc"))


# ---------------------------------------------------------------------------
# sequence_optimization_functions.py — pure Python / pandas / numpy
# ---------------------------------------------------------------------------

class TestAddSpacers:
    def test_spacer_is_prepended(self):
        seqs = ["ATCG", "GCTA"]
        result = add_spacers(seqs, "TTT")
        assert result == ["TTTATCG", "TTTGCTA"]

    def test_output_length(self):
        seqs = ["ATCG", "GCTA", "TAGC"]
        result = add_spacers(seqs, "TT")
        assert len(result) == len(seqs)

    def test_empty_spacer(self):
        seqs = ["ATCG", "GCTA"]
        result = add_spacers(seqs, "")
        assert result == seqs


class TestSetColumnComplementToInf:
    @pytest.fixture
    def symmetric_df(self):
        # ATCG and CGAT are reverse complements
        seqs = ["ATCG", "CGAT"]
        # complementary pair has value -5 (strong), self-interaction has -1 (weak)
        data = np.array([[-1.0, -5.0],
                         [-5.0, -1.0]])
        return pd.DataFrame(data, index=seqs, columns=seqs)

    def test_complementary_pairs_become_inf(self, symmetric_df):
        result = set_column_complement_to_inf(symmetric_df)
        assert np.isinf(result.loc["ATCG", "CGAT"])
        assert np.isinf(result.loc["CGAT", "ATCG"])

    def test_non_complementary_values_are_finite(self, symmetric_df):
        result = set_column_complement_to_inf(symmetric_df)
        assert np.isfinite(result.loc["ATCG", "ATCG"])
        assert np.isfinite(result.loc["CGAT", "CGAT"])

    def test_returns_dataframe_with_same_index(self, symmetric_df):
        result = set_column_complement_to_inf(symmetric_df)
        assert list(result.index) == list(symmetric_df.index)
        assert list(result.columns) == list(symmetric_df.columns)

    def test_four_sequence_case(self):
        seqs = ["ATCG", "CGAT", "GCTA", "TAGC"]
        # Complementary pairs: ATCG↔CGAT, GCTA↔TAGC
        # Each comp pair has value -5, self value -1, other pairs -2
        data = np.full((4, 4), -2.0)
        np.fill_diagonal(data, -1.0)
        # Set complementary pair values to -5
        idx = {s: i for i, s in enumerate(seqs)}
        for s in seqs:
            i, j = idx[s], idx[rc(s)]
            data[i, j] = -5.0
        df = pd.DataFrame(data, index=seqs, columns=seqs)
        result = set_column_complement_to_inf(df)
        for s in seqs:
            assert np.isinf(result.loc[s, rc(s)]), f"Expected inf at ({s}, {rc(s)})"


class TestLoadEnsemble:
    def test_loads_sequences_and_complements(self, tmp_path):
        f = tmp_path / "seqs.txt"
        f.write_text("ATCGATCG\nGCTAGCTA\n")
        result = load_ensemble(f)
        # Should contain both input seqs and their RCs
        assert "ATCGATCG" in result
        assert rc("ATCGATCG") in result
        assert "GCTAGCTA" in result
        assert rc("GCTAGCTA") in result

    def test_self_complementary_sequences_excluded(self, tmp_path):
        # "ACGT" is self-complementary: rc("ACGT") = "ACGT"
        f = tmp_path / "seqs.txt"
        f.write_text("ACGT\nATCGATCG\n")
        result = load_ensemble(f)
        # Self-complementary "ACGT" should be excluded
        assert result.count("ACGT") == 0
        # Non-self-complementary one should be present
        assert "ATCGATCG" in result

    def test_no_duplicates_in_output(self, tmp_path):
        f = tmp_path / "seqs.txt"
        f.write_text("ATCGATCG\n")
        result = load_ensemble(f)
        assert len(result) == len(set(result))

    def test_returns_list(self, tmp_path):
        f = tmp_path / "seqs.txt"
        f.write_text("ATCGATCG\n")
        result = load_ensemble(f)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# NUPACK integration — minimal calls to keep test suite fast
# ---------------------------------------------------------------------------

class TestCreateTube:
    def test_returns_tube_object(self):
        import nupack
        model = nupack.Model(material='dna', celsius=37, sodium=0.05, magnesium=0.0125)
        tube = create_tube("ATCGATCG", 37, model)
        assert isinstance(tube, nupack.Tube)


class TestGetUnboundFraction:
    def test_returns_list_of_floats(self):
        result = get_unbound_fraction(["ATCGATCG"], 37)
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], float)

    def test_values_in_zero_one(self):
        result = get_unbound_fraction(["ATCGATCG", "GCATGCAT"], 37)
        for v in result:
            assert 0.0 <= v <= 1.0

    def test_high_temp_mostly_unbound(self):
        # At 80°C, an 8-nt duplex should be mostly unbound
        result = get_unbound_fraction(["ATCGATCG"], 80)
        assert result[0] > 0.7

    def test_low_temp_mostly_bound(self):
        # At 4°C, an 8-nt duplex should be mostly bound
        result = get_unbound_fraction(["ATCGATCG"], 4)
        assert result[0] < 0.5

    def test_multiple_sequences(self):
        seqs = ["ATCGATCG", "GCATGCAT", "TACGATCG"]
        result = get_unbound_fraction(seqs, 37)
        assert len(result) == len(seqs)


class TestComputeMeltingTemp:
    def test_returns_dicts(self):
        temp_range = np.array([15.0, 20.0, 25.0])
        tms, fracs = compute_melting_temp(["ATCGATCG"], temp_range)
        assert isinstance(tms, dict)
        assert isinstance(fracs, dict)

    def test_tm_key_matches_input_seq(self):
        temp_range = np.array([15.0, 20.0, 25.0])
        tms, _ = compute_melting_temp(["ATCGATCG"], temp_range)
        assert "ATCGATCG" in tms

    def test_tm_is_in_temp_range(self):
        temp_range = np.array([10.0, 20.0, 30.0, 40.0])
        tms, _ = compute_melting_temp(["ATCGATCG"], temp_range)
        assert tms["ATCGATCG"] in temp_range

    def test_unbound_fractions_are_increasing(self):
        # As temp increases, unbound fraction should increase
        temp_range = np.array([4.0, 20.0, 40.0, 70.0])
        _, fracs = compute_melting_temp(["ATCGATCG"], temp_range)
        seq_fracs = fracs["ATCGATCG"]
        assert seq_fracs[-1] > seq_fracs[0], "Unbound fraction should increase with temperature"


# ---------------------------------------------------------------------------
# Melting temperature screening
# ---------------------------------------------------------------------------

class TestMeltingTempScreen:
    def test_biopy_returns_dict(self):
        seqs = ["ATCGATCG", "GCATGCAT"]
        result = melting_temp_screen(seqs, 20.0, 5.0, "biopy")
        assert isinstance(result, dict)

    def test_biopy_all_results_within_tolerance(self):
        seqs = ["ATCGATCG", "GCATGCAT", "TACGATCG", "ATGCATGC"]
        celsius = 20.0
        tolerance = 3.0
        result = melting_temp_screen(seqs, celsius, tolerance, "biopy")
        for seq, tm in result.items():
            assert celsius - tolerance < tm < celsius + tolerance, (
                f"{seq} has Tm {tm:.1f}°C, outside [{celsius-tolerance}, {celsius+tolerance}]"
            )

    def test_biopy_very_narrow_window_reduces_results(self):
        seqs = ["ATCGATCG", "GCATGCAT", "TACGATCG", "ATGCATGC"]
        wide = melting_temp_screen(seqs, 20.0, 5.0, "biopy")
        narrow = melting_temp_screen(seqs, 20.0, 0.1, "biopy")
        assert len(narrow) <= len(wide)

    def test_biopy_far_off_target_returns_empty(self):
        # 8-nt sequences have Tm ~17-21°C; targeting 60°C should return nothing
        seqs = ["ATCGATCG", "GCATGCAT"]
        result = melting_temp_screen(seqs, 60.0, 0.5, "biopy")
        assert len(result) == 0