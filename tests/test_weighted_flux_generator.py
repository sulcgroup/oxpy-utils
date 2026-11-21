"""Tests for ffs/weighted_flux_generator.py: stratified_weights, WeightedSourceFluxGenerator pool handling."""
import csv
import math
import random
from collections import Counter

import pytest

from oxpy_utils.ffs.flux_generator import FFSFluxGenerator
from oxpy_utils.ffs.weighted_flux_generator import (
    ATTEMPT_LOG_COLUMNS,
    RestartOnFailFluxGenerator,
    WeightedSourceFluxGenerator,
    conf_time,
    stratified_weights,
)


class TestStratifiedWeights:
    def test_label_probability_split_over_confs(self):
        w = stratified_weights(["a", "a", "b"], {"a": 0.5, "b": 0.5})
        assert w == [0.25, 0.25, 0.5]

    def test_missing_label_gets_zero(self):
        w = stratified_weights(["a", "c"], {"a": 1.0})
        assert w == [1.0, 0.0]

    def test_all_zero_raises(self):
        with pytest.raises(ValueError):
            stratified_weights(["c"], {"a": 1.0})

    def test_draw_frequencies_follow_label_probs(self):
        labels = [0] * 10 + [1] * 2  # many more confs with label 0
        w = stratified_weights(labels, {0: 0.2, 1: 0.8})
        rng = random.Random(1)
        draws = Counter(labels[i] for i in rng.choices(range(len(labels)), weights=w, k=20000))
        assert draws[1] / 20000 == pytest.approx(0.8, abs=0.02)


@pytest.fixture
def generator(tmp_path):
    g = WeightedSourceFluxGenerator("flux", tmp_path, tmp_path / "out")
    g.source_directory = tmp_path / "source"
    return g


def make_confs(d, names):
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / n).write_text("t = 0\n")
    return [d / n for n in names]


class TestWeightedSourceFluxGenerator:
    def test_restarts_after_success_by_default(self, generator):
        assert generator.restart_after_success
        assert not FFSFluxGenerator.restart_after_success

    def test_empty_pool_falls_back_to_default(self, generator):
        assert generator.choose_source_conf(random.Random(0)) is None

    def test_pool_copied_into_source_dir(self, generator, tmp_path):
        confs = make_confs(tmp_path / "elsewhere", ["a.dat", "b.dat"])
        generator.set_source_pool(confs, [1, 3])
        assert generator.pool == ["a.dat", "b.dat"]
        assert (generator.source_directory / "a.dat").is_file()

    def test_bare_names_must_exist_in_source_dir(self, generator):
        make_confs(generator.source_directory, ["a.dat"])
        generator.set_source_pool(["a.dat"])
        with pytest.raises(FileNotFoundError):
            generator.set_source_pool(["missing.dat"])

    def test_zero_weight_never_drawn(self, generator, tmp_path):
        confs = make_confs(tmp_path / "elsewhere", ["a.dat", "b.dat"])
        generator.set_source_pool(confs, [0, 1])
        rng = random.Random(0)
        assert {generator.choose_source_conf(rng) for _ in range(200)} == {"b.dat"}

    @pytest.mark.parametrize("weights", [[1], [-1, 2], [0, 0]])
    def test_bad_weights_rejected(self, generator, tmp_path, weights):
        confs = make_confs(tmp_path / "elsewhere", ["a.dat", "b.dat"])
        with pytest.raises(ValueError):
            generator.set_source_pool(confs, weights)

    def test_duplicate_names_rejected(self, generator, tmp_path):
        a = make_confs(tmp_path / "x", ["a.dat"])[0]
        b = make_confs(tmp_path / "y", ["a.dat"])[0]
        with pytest.raises(ValueError):
            generator.set_source_pool([a, b])

    def test_write_pool(self, generator, tmp_path):
        confs = make_confs(tmp_path / "elsewhere", ["a.dat", "b.dat"])
        generator.set_source_pool(confs, [1, 3])
        generator.write_pool()
        lines = (tmp_path / "source_pool.csv").read_text().splitlines()
        assert lines[0] == "conf,weight,probability"
        assert lines[2] == "b.dat,3.0,0.75"


class TestRestartOnFailFluxGenerator:
    @pytest.fixture
    def rof(self, tmp_path):
        g = RestartOnFailFluxGenerator("flux", tmp_path, tmp_path / "out")
        g.source_directory = tmp_path / "source"
        return g

    def write_log(self, path, rows):
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ATTEMPT_LOG_COLUMNS)
            w.writeheader()
            for outcome, steps in rows:
                w.writerow({"sim": "sim0", "source_conf": "a.dat", "seed": 1, "outcome": outcome,
                            "steps": steps, "success_index": ""})

    def test_is_a_weighted_source_generator(self, rof):
        assert isinstance(rof, WeightedSourceFluxGenerator)
        assert rof.refresh_velocities

    def test_no_attempts(self, rof):
        assert math.isnan(rof.success_probability())
        assert math.isnan(rof.compute_flux())

    def test_statistics_pooled_over_worker_logs(self, rof, tmp_path):
        self.write_log(tmp_path / "attempts_p0.csv", [("success", 100), ("failure", 300)])
        self.write_log(tmp_path / "attempts_p1.csv", [("failure", 500), ("undetermined", 100)])
        assert rof.outcome_counts() == Counter(success=1, failure=2, undetermined=1)
        # undetermined attempts count toward time but not toward the probability
        assert rof.success_probability() == pytest.approx(1 / 3)
        assert rof.compute_flux() == pytest.approx(1 / 1000)

    def test_refuses_to_run_without_pool(self, rof):
        with pytest.raises(RuntimeError):
            rof.ffs_process(0, None)

    def test_conf_time(self, tmp_path):
        conf = tmp_path / "c.dat"
        conf.write_text("t = 1.5e+06\nb = 1 1 1\n")
        assert conf_time(conf) == 1500000


def test_program_selects_restart_on_fail(tmp_path):
    from oxpy_utils.ffs.ffs_program import FFSProgram
    src = tmp_path / "source"
    confs = make_confs(src, ["a.dat"])
    (tmp_path / "run").mkdir()
    (tmp_path / "run2").mkdir()
    prog = FFSProgram("p", tmp_path / "run", 1, 10, source_directory=src)
    prog.use_source_pool(confs, restart_on_fail=True)
    assert isinstance(prog.fluxer, RestartOnFailFluxGenerator)
    prog2 = FFSProgram("q", tmp_path / "run2", 1, 10, source_directory=src)
    prog2.use_source_pool(confs)
    assert type(prog2.fluxer) is WeightedSourceFluxGenerator
