"""
Tests for the FFSProgram process graph: the event log, replay across launches, success-conf lineage, and
simulation index continuation. The graph listener is driven with the messages the workers send, so no
simulations run.
"""
import json
import threading

import pytest

from oxpy_utils.ffs.ffs_interface import Comparison, FFSInterface
from oxpy_utils.ffs.ffs_program import FFSProgram
from oxpy_utils.utils.order_parameter import OrderParameter

FLUX = "initial_flux"


def interfaces():
    dist = OrderParameter("dist", "mindistance", [(0, 1)])
    bonds = OrderParameter("bonds", "bond", [(0, 1)])
    # lambda_-1, lambda_0, then two shooters (shoot1: bonds >= 2, shoot2: bonds >= 3)
    return [FFSInterface(dist, 8.0, Comparison.LT)] + [FFSInterface(bonds, n, Comparison.GEQ) for n in (1, 2, 3)]


def make_program(root, n_cpus=2):
    root.mkdir(exist_ok=True)
    (root / "source").mkdir(exist_ok=True)
    prog = FFSProgram("t", root, n_cpus, 4, source_directory=root / "source")
    prog.set_interfaces(*interfaces())
    return prog


def send(prog, *messages):
    """run the graph listener over these messages, as FFSProgram.run() does"""
    listener = threading.Thread(target=prog._graph_updater, daemon=True)
    listener.start()
    for m in messages:
        prog.graph_update_queue.put(m)
    prog.graph_update_queue.put(("TERMINATE",))
    listener.join(timeout=30)
    assert not listener.is_alive()


def first_launch(prog):
    """one flux success (worker 0, from pool conf a.dat) and one shoot1 success shot from it (worker 1)"""
    flux_dir = prog.fluxer.destination_directory
    send(prog,
         ("LAUNCH", {"n_cpus": 2}),
         ("flux", 0, 0, "equilibrate", "origin", {"source_conf": "a.dat"}),
         ("flux", 0, 1, "reset", (FLUX, 0, 0)),
         ("flux", 0, 2, "to_l-1_fwd", (FLUX, 0, 1)),
         ("flux", 0, 3, "flux_fwd", (FLUX, 0, 2)),
         ("CPY_CONF", FLUX, 0, 3, 0),
         ("flux_report", 0, 3, True),
         ("shoot", 1, 0, "shoot1", 0, {"source_conf": str(flux_dir / "success_1.dat"), "seed": 7}),
         ("CPY_CONF", "shoot1", 1, 0, 0),
         ("shoot_report", 1, 0, "shoot1", True))


def graph_summary(prog):
    g = prog.process_graph
    return set(g.nodes), set(g.edges)


def test_success_nodes_record_their_conf_and_shoots_their_parent(tmp_path):
    prog = make_program(tmp_path / "run")
    first_launch(prog)
    g = prog.process_graph
    flux_node = g.nodes[(FLUX, 0, 3)]
    assert flux_node["success_index"] == 1
    assert flux_node["success_conf"] == str(prog.fluxer.destination_directory / "success_1.dat")
    assert g.nodes[(FLUX, 0, 0)]["source_conf"] == "a.dat"
    shoot = g.nodes[("shoot1", 1, 0)]
    assert (shoot["source_step"], shoot["source_success_index"]) == (FLUX, 1)
    assert shoot["seed"] == 7 and shoot["success"] is True and shoot["success_index"] == 1
    assert ((FLUX, 0, 3), ("shoot1", 1, 0)) in g.edges
    assert g.graph["launches"] == [{"n_cpus": 2}]


def test_event_log_has_every_message_but_terminate(tmp_path):
    prog = make_program(tmp_path / "run")
    first_launch(prog)
    lines = prog.graph_events_path().read_text().splitlines()
    assert len(lines) == 10
    events = [json.loads(line) for line in lines]
    assert events[0]["cmd"][0] == "LAUNCH" and all("t" in e for e in events)
    assert events[2]["cmd"] == ["flux", 0, 1, "reset", [FLUX, 0, 0]]


def test_replay_rebuilds_graph_in_new_program(tmp_path):
    prog = make_program(tmp_path / "run")
    first_launch(prog)
    # a later launch with fewer worker processes than the events it replays
    prog2 = make_program(tmp_path / "run", n_cpus=1)
    assert prog2.replay_graph_events() == 10
    assert graph_summary(prog2) == graph_summary(prog)
    assert prog2.process_graph.nodes[("shoot1", 1, 0)]["success_conf"].endswith("shoot1/success_1.dat")
    assert prog2.graph_source_map == prog.graph_source_map


def test_shoot_links_to_conf_written_in_earlier_launch(tmp_path):
    prog = make_program(tmp_path / "run")
    first_launch(prog)
    prog2 = make_program(tmp_path / "run", n_cpus=1)
    prog2.replay_graph_events()
    shoot1_dir = prog2.shooters[0].destination_directory
    send(prog2,
         ("LAUNCH", {"n_cpus": 1}),
         ("shoot", 0, 0, "shoot2", 0, {"source_conf": str(shoot1_dir / "success_1.dat")}))
    assert (("shoot1", 1, 0), ("shoot2", 0, 0)) in prog2.process_graph.edges
    # and a third program sees both launches
    prog3 = make_program(tmp_path / "run")
    prog3.replay_graph_events()
    assert len(prog3.process_graph.graph["launches"]) == 2
    assert graph_summary(prog3) == graph_summary(prog2)


def test_next_sim_indices_continue_after_replayed_sims(tmp_path):
    prog = make_program(tmp_path / "run")
    first_launch(prog)
    assert prog.next_sim_indices(FLUX) == {0: 4}
    assert prog.next_sim_indices("shoot1") == {1: 1}
    assert prog.next_sim_indices("shoot2") == {}


def test_run_replays_and_sets_sim_index_start(tmp_path, monkeypatch):
    prog = make_program(tmp_path / "run")
    first_launch(prog)
    prog2 = make_program(tmp_path / "run")
    for step in [prog2.fluxer, *prog2.shooters]:  # no simulations: only the graph bookkeeping of run()
        monkeypatch.setattr(step, "init", lambda: None)
        monkeypatch.setattr(step, "run", lambda: None)
    prog2.run()
    assert prog2.fluxer.sim_index_start == {0: 4}
    assert prog2.shooters[0].sim_index_start == {1: 1}
    assert graph_summary(prog2) == graph_summary(prog)
    assert len(prog2.process_graph.graph["launches"]) == 2  # run() logs its own LAUNCH event


def test_old_style_messages_still_work(tmp_path):
    prog = make_program(tmp_path / "run")
    flux_success = str(prog.fluxer.destination_directory / "success_1.dat")
    prog.shooters[0].starting_confs = [flux_success]
    send(prog,
         ("flux", 0, 0, "flux_fwd", "origin"),
         ("CPY_CONF", FLUX, 0, 0, 0),
         ("shoot", 0, 0, "shoot1", 0))
    assert prog.process_graph.nodes[("shoot1", 0, 0)]["source_conf"] == flux_success
    assert ((FLUX, 0, 0), ("shoot1", 0, 0)) in prog.process_graph.edges


def test_unresolvable_parent_keeps_lineage_attrs(tmp_path):
    prog = make_program(tmp_path / "run")
    send(prog, ("shoot", 0, 0, "shoot1", 4, {"source_conf": "x/success_5.dat"}))
    node = prog.process_graph.nodes[("shoot1", 0, 0)]
    assert (node["source_step"], node["source_success_index"]) == (FLUX, 5)
    assert prog.process_graph.in_degree(("shoot1", 0, 0)) == 0


def test_truncated_last_event_is_skipped(tmp_path):
    prog = make_program(tmp_path / "run")
    first_launch(prog)
    with prog.graph_events_path().open("a") as f:
        f.write('{"t": 1, "cmd": ["flux", 0, 4, "flux_')  # cut short by a crash
    prog2 = make_program(tmp_path / "run")
    assert prog2.replay_graph_events() == 10
    assert graph_summary(prog2) == graph_summary(prog)


def test_save_and_load_graph_round_trip(tmp_path):
    prog = make_program(tmp_path / "run")
    first_launch(prog)
    prog.save_graph()
    prog2 = make_program(tmp_path / "run")
    prog2.load_graph()
    assert graph_summary(prog2) == graph_summary(prog)
    assert prog2.process_graph.graph["launches"] == [{"n_cpus": 2}]
    assert prog2.graph_source_map == prog.graph_source_map
    assert prog2.process_graph.nodes[("shoot1", 1, 0)]["shooter"] is prog2.shooters[0]


def test_load_prefers_event_log(tmp_path):
    prog = make_program(tmp_path / "run")
    first_launch(prog)
    prog2 = make_program(tmp_path / "run")
    prog2.load()
    assert graph_summary(prog2) == graph_summary(prog)
