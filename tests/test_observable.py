import pytest

from oxpy_utils.utils.observable import (
    Observable, ObservableColumn,
    simulation_time, distance, hb_list,
    particle_position, potential_energy,
    force_energy, kinetic_energy, pair_energy,
)


class TestObservableColumn:
    def test_export_includes_type(self):
        col = ObservableColumn("distance", particle_1=0, particle_2=1)
        d = col.export()
        assert d["type"] == "distance"
        assert d["particle_1"] == "0"
        assert d["particle_2"] == "1"

    def test_all_values_are_strings(self):
        col = ObservableColumn("hb_list", only_count=True)
        for v in col.export().values():
            assert isinstance(v, str)

    def test_type_not_settable(self):
        col = ObservableColumn("distance")
        with pytest.raises(AttributeError):
            col.type_name = "hb_list"

    def test_no_extra_attrs(self):
        col = ObservableColumn("kinetic_energy")
        assert col.export() == {"type": "kinetic_energy"}


class TestObservable:
    def test_construction_and_export_shape(self):
        obs = Observable("time.txt", 1000, ObservableColumn("step"))
        out = obs.export()["output"]
        assert out["name"] == "time.txt"
        assert out["print_every"] == "1000"
        assert len(out["cols"]) == 1
        assert out["cols"][0]["type"] == "step"

    def test_float_print_every_is_rounded_to_int(self):
        obs = Observable("out.txt", 1e3, ObservableColumn("step"))
        assert obs.print_every == 1000
        assert isinstance(obs.print_every, int)

    def test_multiple_cols(self):
        obs = Observable("out.txt", 100,
                         ObservableColumn("step"),
                         ObservableColumn("potential_energy"))
        assert len(obs) == 2

    def test_add_col_prepend(self):
        obs = Observable("out.txt", 100, ObservableColumn("step"))
        obs.add_col(ObservableColumn("hb_list"), prepend=True)
        cols = list(obs.get_cols())
        assert cols[0].type_name == "hb_list"
        assert len(obs) == 2

    def test_add_col_append(self):
        obs = Observable("out.txt", 100, ObservableColumn("step"))
        obs.add_col(ObservableColumn("hb_list"), prepend=False)
        cols = list(obs.get_cols())
        assert cols[-1].type_name == "hb_list"

    def test_get_cols_returns_deep_copies(self):
        obs = Observable("out.txt", 100, ObservableColumn("distance", particle_1=0))
        c1 = list(obs.get_cols())
        c2 = list(obs.get_cols())
        assert c1[0] is not c2[0]

    def test_set_file_name(self):
        obs = Observable("old.txt", 100, ObservableColumn("step"))
        obs.file_name = "new.txt"
        assert obs.file_name == "new.txt"

    def test_set_print_every(self):
        obs = Observable("out.txt", 100, ObservableColumn("step"))
        obs.print_every = 500
        assert obs.print_every == 500

    def test_repr_contains_key_fields(self):
        obs = Observable("energy.txt", 500, ObservableColumn("potential_energy"))
        r = repr(obs)
        assert "energy.txt" in r
        assert "500" in r
        assert "potential_energy" in r

    def test_dict_col_arg_is_accepted(self):
        obs = Observable("out.txt", 100, {"name": "step"})
        assert list(obs.get_cols())[0].type_name == "step"


class TestObservableFactoryFunctions:
    def test_simulation_time(self):
        d = simulation_time(print_every=100, name="time.txt")
        assert d["output"]["cols"][0]["type"] == "step"
        assert d["output"]["name"] == "time.txt"
        assert d["output"]["print_every"] == "100"

    def test_distance(self):
        d = distance(particle_1=0, particle_2=1, print_every=100, name="dist.txt")
        col = d["output"]["cols"][0]
        assert col["type"] == "distance"
        assert col["particle_1"] == "0"
        assert col["particle_2"] == "1"

    def test_distance_with_pbc(self):
        d = distance(particle_1=0, particle_2=1, PBC=True, print_every=100, name="dist.txt")
        assert d["output"]["cols"][0]["PBC"] == "True"

    def test_hb_list(self):
        d = hb_list(print_every=100, name="hb.txt", only_count=True)
        col = d["output"]["cols"][0]
        assert col["type"] == "hb_list"
        assert col["only_count"] == "True"

    def test_particle_position(self):
        d = particle_position(particle_id=3, print_every=100, name="pos.txt")
        col = d["output"]["cols"][0]
        assert col["type"] == "particle_position"
        assert col["particle_id"] == "3"

    def test_potential_energy(self):
        d = potential_energy(print_every=100, name="pe.txt", split=True)
        col = d["output"]["cols"][0]
        assert col["type"] == "potential_energy"
        assert col["split"] == "True"

    def test_force_energy_with_print_group(self):
        d = force_energy(print_every=100, name="fe.txt", print_group="mygroup")
        col = d["output"]["cols"][0]
        assert col["type"] == "force_energy"
        assert col["print_group"] == "mygroup"

    def test_force_energy_without_print_group(self):
        d = force_energy(print_every=100, name="fe.txt")
        col = d["output"]["cols"][0]
        assert col["type"] == "force_energy"
        assert "print_group" not in col

    def test_kinetic_energy(self):
        d = kinetic_energy(print_every=100, name="ke.txt")
        assert d["output"]["cols"][0]["type"] == "kinetic_energy"

    def test_pair_energy_both_particles(self):
        d = pair_energy(print_every=100, name="pe.txt", particle1_id=0, particle2_id=1)
        col = d["output"]["cols"][0]
        assert col["type"] == "pair_energy"
        assert col["particle1_id"] == "0"
        assert col["particle2_id"] == "1"

    def test_pair_energy_no_particles(self):
        d = pair_energy(print_every=100, name="pe.txt")
        assert d["output"]["cols"][0]["type"] == "pair_energy"

    def test_pair_energy_missing_particle2_raises(self):
        with pytest.raises(ValueError):
            pair_energy(print_every=100, name="pe.txt", particle1_id=0)