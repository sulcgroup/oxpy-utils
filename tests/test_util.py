"""Tests for utils/util.py: rotation_matrix, process_path, si_units, ox_units."""
import math
from pathlib import Path

import numpy as np
import pytest

from oxpy_utils.utils.util import (
    generate_distinct_colors,
    ox_units,
    process_path,
    rotation_matrix,
    si_units,
)


# ---------------------------------------------------------------------------
# rotation_matrix
# ---------------------------------------------------------------------------

class TestRotationMatrix:
    def test_shape(self):
        R = rotation_matrix([0, 0, 1], 0.5)
        assert R.shape == (3, 3)

    def test_zero_rotation_is_identity(self):
        R = rotation_matrix([1, 0, 0], 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_determinant_is_one(self):
        R = rotation_matrix([1, 1, 0], math.pi / 4)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)

    def test_is_orthogonal(self):
        R = rotation_matrix([0, 1, 0], math.pi / 3)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)

    def test_180_degree_rotation_around_z(self):
        # 180° around z: x→-x, y→-y, z→z
        R = rotation_matrix([0, 0, 1], math.pi)
        rotated = R @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(rotated, [-1.0, 0.0, 0.0], atol=1e-12)

    def test_90_degree_rotation_around_z(self):
        # 90° CCW around z: x→y
        R = rotation_matrix([0, 0, 1], math.pi / 2)
        rotated = R @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(rotated, [0.0, 1.0, 0.0], atol=1e-12)

    def test_non_unit_axis_normalized(self):
        # scaling the axis should not change the result
        R1 = rotation_matrix([0, 0, 1], math.pi / 4)
        R2 = rotation_matrix([0, 0, 5], math.pi / 4)
        np.testing.assert_allclose(R1, R2, atol=1e-12)


# ---------------------------------------------------------------------------
# process_path
# ---------------------------------------------------------------------------

class TestProcessPath:
    def test_string_converted_to_path(self):
        result = process_path("/tmp/foo.txt")
        assert isinstance(result, Path)

    def test_absolute_path_unchanged(self):
        p = Path("/absolute/path.txt")
        result = process_path(p)
        assert result == p

    def test_relative_path_prepended(self, tmp_path):
        result = process_path("subdir/file.txt", prepend=tmp_path)
        assert result == tmp_path / "subdir" / "file.txt"

    def test_dotdot_relative_path_not_prepended(self, tmp_path):
        # paths starting with ".." are left as-is (parent traversal kept intact)
        p = Path("../sibling/file.txt")
        result = process_path(p, prepend=tmp_path)
        assert result == p

    def test_tilde_expanded(self):
        result = process_path(Path("~/some_file.txt"))
        assert "~" not in str(result)
        assert result.is_absolute()

    def test_no_prepend_leaves_relative_unchanged(self):
        result = process_path("some/path.txt")
        assert result == Path("some/path.txt")


# ---------------------------------------------------------------------------
# si_units (temperature)
# ---------------------------------------------------------------------------

class TestSIUnits:
    def test_temperature_to_kelvin_dna(self):
        # 1 oxDNA unit = 3000 K
        result = si_units(1.0, "dna", "T", to="K")
        assert result == pytest.approx(3000.0)

    def test_temperature_to_celsius_dna(self):
        # 1 oxDNA unit = 3000 K = 2726.85 °C
        result = si_units(1.0, "dna", "T", to="C")
        assert result == pytest.approx(3000.0 - 273.15)

    def test_temperature_to_kelvin_rna(self):
        result = si_units(1.0, "rna", "T", to="K")
        assert result == pytest.approx(3000.0)

    def test_distance_dna(self):
        # 1 oxDNA unit = 8.518e-10 m
        result = si_units(1.0, "dna", "distance")
        assert result == pytest.approx(8.518e-10, rel=1e-3)

    def test_distance_alias_d(self):
        result = si_units(2.0, "dna", "d")
        assert result == pytest.approx(2 * 8.518e-10, rel=1e-3)

    def test_unsupported_target_raises(self):
        with pytest.raises(ValueError):
            si_units(1.0, "dna", "T", to="F")


# ---------------------------------------------------------------------------
# ox_units
# ---------------------------------------------------------------------------

class TestOxUnits:
    def test_temperature_celsius(self):
        # 37 °C → (37 + 273.15) / 3000 oxDNA units ≈ 0.1037
        result = ox_units(37.0, "dna", units="C")
        assert result == pytest.approx((37.0 + 273.15) / 3000.0, rel=1e-4)

    def test_temperature_kelvin(self):
        result = ox_units(300.0, "dna", units="K")
        assert result == pytest.approx(300.0 / 3000.0, rel=1e-6)

    def test_distance_meters(self):
        result = ox_units(8.518e-10, "dna", units="m")
        assert result == pytest.approx(1.0, rel=1e-3)

    def test_distance_nm(self):
        # 0.8518 nm == 8.518e-10 m == ~1 oxDNA unit
        result = ox_units(0.8518, "dna", units="nm")
        assert result == pytest.approx(1.0, rel=1e-3)

    def test_force_pn_round_trip(self):
        # ~48.6 pN per oxDNA force unit
        val_pn = 48.63
        ox_val = ox_units(val_pn, "dna", units="pN")
        assert ox_val == pytest.approx(1.0, rel=0.01)

    def test_units_parsed_from_string(self):
        # "37C" → parse units as "C"
        result = ox_units("37C", "dna")
        expected = ox_units(37.0, "dna", units="C")
        assert result == pytest.approx(expected)

    def test_string_no_unit_suffix_raises(self):
        # passing a numeric string with no unit suffix causes a ValueError
        with pytest.raises(ValueError):
            ox_units("37", "dna")

    def test_fahrenheit_raises(self):
        with pytest.raises(ValueError):
            ox_units(98.6, "dna", units="F")

    def test_invalid_unit_raises(self):
        with pytest.raises(ValueError):
            ox_units(1.0, "dna", units="XYZ")

    def test_celsius_to_ox_then_si_roundtrip(self):
        T_C = 37.0
        ox_val = ox_units(T_C, "dna", units="C")
        T_C_back = si_units(ox_val, "dna", "T", to="C")
        assert T_C_back == pytest.approx(T_C, abs=1e-6)

    def test_rna_vs_dna_differ(self):
        # DNA and RNA have slightly different unit scales
        dna = ox_units(1.0, "dna", units="N")
        rna = ox_units(1.0, "rna", units="N")
        assert dna != pytest.approx(rna, rel=1e-6)


# ---------------------------------------------------------------------------
# generate_distinct_colors
# ---------------------------------------------------------------------------

class TestGenerateDistinctColors:
    def test_returns_n_colors(self):
        colors = generate_distinct_colors(5)
        assert len(colors) == 5

    def test_shape(self):
        colors = generate_distinct_colors(4)
        assert colors.shape == (4, 4)  # RGBA

    def test_values_in_range(self):
        colors = generate_distinct_colors(8)
        assert (colors >= 0).all()
        assert (colors <= 1).all()

    def test_single_color(self):
        colors = generate_distinct_colors(1)
        assert colors.shape == (1, 4)