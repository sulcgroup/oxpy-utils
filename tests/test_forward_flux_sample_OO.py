import os.path
from pathlib import Path
from unittest.mock import patch

import yaml

from oxpy_utils.ffs import ffs_cli
from oxpy_utils.ffs.ffs_interface import FFSInterface, Comparison
from oxpy_utils.ffs.ffs_program import FFSProgram
from oxpy_utils.utils.order_parameter import OrderParameter

bonds = list(zip(
    [0, 1, 2, 3, 4, 5, 6, 7],
    reversed([8, 9, 10, 11, 12, 13, 14, 15])
))

# construct order param for n bonds
native = OrderParameter("native", "bond", bonds)
# construct order parameter for distance
mindist = OrderParameter("distance", "mindistance", bonds)

example_interfaces = (
    FFSInterface(
        native,
        7,
        Comparison.LEQ
    ),

    # interface lambda_{0}
    FFSInterface(
        native,
        6,  # 2 fewer bonds,
        Comparison.LEQ
    ),

    FFSInterface(
        native,
        4,
        Comparison.LEQ
    ),
    FFSInterface(
        native,
        2,
        Comparison.LEQ
    ),

    # interface lambda_{success}
    FFSInterface(
        mindist,
        5.,  # 5 distance units
        Comparison.GT
    )
)

def test_ffs_seperation(tmp_path):

    # Path to the examples directory in your package
    # Adjust the path as necessary depending on your package structure
    examples_dir = Path(__file__).parent.parent / 'examples' / '8nt_duplex_files'

    num_cpus = max(os.cpu_count() // 2, 1) # don't eat them all
    desired_n_successes = 12
    program = FFSProgram("test_ffs",
                         tmp_path,
                         num_cpus,
                         desired_n_successes,
                         examples_dir)
    program.input_file_params["T"] = "70C" # for test purposes, very high temperature to make go fast
    program.ffs_default_input_name = "ffs" # or ffs_cuda for cuda jobs
    program.loghandler.set_verbose(True)
    print("Setting interfaces...")
    program.set_interfaces(*example_interfaces)
    print("Running program....")
    program.run()
    program.save_graph()
    program.plot_graph()
    pass

def test_load_ffs(tmp_path):
    ffs_dir = Path(".") / "test_data" / "example_ffs"
    # copy ffs dir to tmp_path
    for item in ffs_dir.iterdir():
        # must be a better way
        if item.is_dir():
            os.system(f"cp -r {item} {tmp_path}")
        else:
            os.system(f"cp {item} {tmp_path}")
    program = FFSProgram(
        "example",
        tmp_path,
        1,
        12
    )
    program.set_interfaces(*example_interfaces)
    program.loghandler.set_verbose(True)
    program.load()
    program.load_graph()
    program.plot_graph()
    program.export_results()
    pass

def test_ffs_seperation_from_yml(tmp_path):
    examples_dir = Path(__file__).parent.parent / "examples" / "8nt_duplex_files"

    ffs_config = {
        "job_name": "ffs_test_job",
        "file_dir": str(examples_dir),
        "output_dir": str(tmp_path),
        "num_cpus": max(os.cpu_count() // 2, 1),
        "desired_n_successes": 6,
        "use_cuda": False,
        "input_file_params": {"T": "70C"},
        "order_parameters": {
            "native": {
                "order_parameter": "bond",
                "nucleotide_indexes_0": [0, 1, 2, 3, 4, 5, 6, 7],
                "nucleotide_indexes_1": [14, 13, 12, 11, 10, 9, 8],
            },
            "mindist": {
                "order_parameter": "mindistance",
                "nucleotide_indexes_0": [0, 1, 2, 3, 4, 5, 6, 7],
                "nucleotide_indexes_1": [14, 13, 12, 11, 10, 9, 8],
            },
        },
        "interfaces": [
            {"op": "native",  "value": 8,   "compare": "<"},
            {"op": "native",  "value": 6,   "compare": "<="},
            {"op": "native",  "value": 4,   "compare": "<="},
            {"op": "mindist", "value": 5.0, "compare": ">"},
        ],
    }

    yml_path = tmp_path / "ffs_run.yml"
    with open(yml_path, "w") as f:
        yaml.dump(ffs_config, f)

    with patch("sys.argv", ["ipyoxdna_ffs", "--input", str(yml_path)]):
        ffs_cli.main()
