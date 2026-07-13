import json
from pathlib import Path

import numpy as np
import pytest

from oxpy_utils.utils.force import (
    Force, ForceType, get_force_type,
    zip_strands, load_forces_from_txt, load_forces_from_json,
)

EXAMPLE_FORCE_FILE = Path(__file__).parent / 'test_data' / 'duplex_forces.txt'


class TestGetForceType:
    def test_known_type(self):
        assert get_force_type("mutual_trap") is ForceType.MUTUAL_TRAP

    def test_all_enum_members_are_findable(self):
        for ft in ForceType:
            assert get_force_type(ft.value.type_name) is ft

    def test_unknown_type_raises(self):
        with pytest.raises(Exception, match="No force type named"):
            get_force_type("nonexistent_force")


class TestForceConstruction:
    def test_by_enum(self):
        f = Force(force_type=ForceType.MUTUAL_TRAP, particle=0,
                  ref_particle=1, stiff=1.0, r0=1.2, PBC=True)
        assert f.force_type is ForceType.MUTUAL_TRAP
        assert f["type"] == "mutual_trap"

    def test_by_string_type_name(self):
        f = Force(type="mutual_trap", particle=0,
                  ref_particle=1, stiff=1.0, r0=1.2, PBC=True)
        assert f.force_type is ForceType.MUTUAL_TRAP

    def test_defaults_are_filled_when_omitted(self):
        # MUTUAL_TRAP defaults: stiff=0.9, r0=1.2, PBC=True
        f = Force(force_type=ForceType.MUTUAL_TRAP, particle=0, ref_particle=1)
        assert "stiff" in f.force_params
        assert "r0" in f.force_params
        assert "PBC" in f.force_params

    def test_explicit_value_overrides_default(self):
        f = Force(force_type=ForceType.MUTUAL_TRAP, particle=0,
                  ref_particle=1, stiff=5.0)
        assert f.force_params["stiff"] == 5.0

    def test_default_particle_is_minus_one(self):
        f = Force(force_type=ForceType.MUTUAL_TRAP, ref_particle=1)
        assert f.particle == -1

    def test_missing_required_param_raises(self):
        with pytest.raises(ValueError, match="Missing"):
            # ref_particle is required and has no default
            Force(force_type=ForceType.MUTUAL_TRAP, particle=0)

    def test_excess_param_raises(self):
        with pytest.raises(ValueError, match="Excess"):
            Force(force_type=ForceType.MUTUAL_TRAP, particle=0,
                  ref_particle=1, stiff=1.0, r0=1.2, PBC=True,
                  bogus_param=99)

    def test_no_type_key_raises(self):
        with pytest.raises(Exception):
            Force(particle=0, ref_particle=1)


class TestForceMapping:
    @pytest.fixture
    def trap(self):
        return Force(force_type=ForceType.MUTUAL_TRAP, particle=3,
                     ref_particle=7, stiff=2.0, r0=1.0, PBC=False)

    def test_len_is_two_plus_params(self, trap):
        assert len(trap) == 2 + len(trap.force_params)

    def test_iter_starts_with_type_then_particle(self, trap):
        keys = list(trap)
        assert keys[0] == "type"
        assert keys[1] == "particle"

    def test_getitem_type(self, trap):
        assert trap["type"] == "mutual_trap"

    def test_getitem_particle(self, trap):
        assert trap["particle"] == "3"

    def test_getitem_param_is_string(self, trap):
        assert trap["stiff"] == "2.0"

    def test_getitem_unknown_key_raises(self, trap):
        with pytest.raises(KeyError):
            _ = trap["nonexistent"]

    def test_setitem_param(self, trap):
        trap["stiff"] = 5.0
        assert trap.force_params["stiff"] == 5.0

    def test_setitem_particle(self, trap):
        trap["particle"] = 99
        assert trap.particle == 99

    def test_setitem_unknown_key_raises(self, trap):
        with pytest.raises(KeyError):
            trap["nonexistent"] = 1

    def test_dict_unpacking_contains_type_and_particle(self, trap):
        d = dict(trap)
        assert d["type"] == "mutual_trap"
        assert d["particle"] == "3"
        assert "ref_particle" in d

    def test_numpy_array_param_serializes_comma_separated(self):
        f = Force(force_type=ForceType.HARMONIC_TRAP, particle=0,
                  pos0=np.array([1.0, 2.0, 3.0]), stiff=1.0)
        assert f["pos0"] == "1.0,2.0,3.0"


class TestZipStrands:
    def test_produces_two_forces_per_pair(self):
        forces = list(zip_strands([0, 1, 2], [5, 6, 7]))
        assert len(forces) == 6

    def test_all_are_mutual_traps(self):
        for f in zip_strands([0], [1]):
            assert f.force_type is ForceType.MUTUAL_TRAP

    def test_particles_are_mirrored(self):
        f_fwd, f_rev = list(zip_strands([0], [5]))
        assert f_fwd.particle == 0
        assert f_fwd.force_params["ref_particle"] == 5
        assert f_rev.particle == 5
        assert f_rev.force_params["ref_particle"] == 0

    def test_pbc_is_false(self):
        f, _ = list(zip_strands([0], [1]))
        assert f.force_params["PBC"] is False


class TestLoadForcesFromTxt:
    def test_load_real_force_file(self):
        forces = load_forces_from_txt(EXAMPLE_FORCE_FILE)
        assert len(forces) > 0
        assert all(isinstance(f, Force) for f in forces)
        assert all(f.force_type is ForceType.MUTUAL_TRAP for f in forces)

    def test_force_count_matches_file(self):
        txt = EXAMPLE_FORCE_FILE.read_text()
        n_braces = txt.count("}")
        forces = load_forces_from_txt(EXAMPLE_FORCE_FILE)
        assert len(forces) == n_braces

    def test_synthetic_single_force(self, tmp_path):
        txt = (
            "{\n"
            "    type = mutual_trap\n"
            "    particle = 0\n"
            "    ref_particle = 1\n"
            "    stiff = 0.5\n"
            "    r0 = 1.2\n"
            "    PBC = 1\n"
            "}\n"
        )
        f_path = tmp_path / "forces.txt"
        f_path.write_text(txt)
        forces = load_forces_from_txt(f_path)
        assert len(forces) == 1
        assert forces[0]["type"] == "mutual_trap"
        assert forces[0]["particle"] == "0"

    def test_synthetic_multiple_forces(self, tmp_path):
        block = (
            "{{\n"
            "    type = mutual_trap\n"
            "    particle = {p}\n"
            "    ref_particle = {r}\n"
            "    stiff = 1.0\n"
            "    r0 = 1.2\n"
            "    PBC = 1\n"
            "}}\n"
        )
        txt = block.format(p=0, r=1) + block.format(p=2, r=3)
        f_path = tmp_path / "forces.txt"
        f_path.write_text(txt)
        forces = load_forces_from_txt(f_path)
        assert len(forces) == 2


class TestLoadForcesFromJson:
    def test_single_force(self, tmp_path):
        data = {
            "force_0": {
                "type": "mutual_trap",
                "particle": "0",
                "ref_particle": "1",
                "stiff": "1.0",
                "r0": "1.2",
                "PBC": "1",
            }
        }
        f_path = tmp_path / "forces.json"
        f_path.write_text(json.dumps(data))
        forces = load_forces_from_json(f_path)
        assert len(forces) == 1
        assert forces[0].force_type is ForceType.MUTUAL_TRAP
        assert forces[0]["type"] == "mutual_trap"

    def test_multiple_forces(self, tmp_path):
        entry = {
            "type": "mutual_trap",
            "particle": "0",
            "ref_particle": "1",
            "stiff": "1.0",
            "r0": "1.2",
            "PBC": "1",
        }
        data = {f"force_{i}": entry for i in range(3)}
        f_path = tmp_path / "forces.json"
        f_path.write_text(json.dumps(data))
        forces = load_forces_from_json(f_path)
        assert len(forces) == 3

    def test_roundtrip_via_dict_unpacking(self, tmp_path):
        original = Force(force_type=ForceType.MUTUAL_TRAP, particle=4,
                         ref_particle=9, stiff=2.0, r0=1.1, PBC=True)
        data = {"f": dict(original)}
        f_path = tmp_path / "forces.json"
        f_path.write_text(json.dumps(data))
        loaded = load_forces_from_json(f_path)
        assert loaded[0]["type"] == original["type"]
        assert loaded[0]["particle"] == original["particle"]