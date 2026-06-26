import argparse
from pathlib import Path

import yaml

import safe_exit

# use absolute imports because this is runnable as a script
from oxpy_utils.ffs.ffs_interface import Comparison, FFSInterface
from oxpy_utils.ffs.ffs_program import FFSProgram
from oxpy_utils.utils.order_parameter import OrderParameter

# run with python ffs_cli.py --input myinput.yml

program: FFSProgram = None

def cleanup_function():
    # Perform cleanup tasks
    if program is not None:
        program.save_graph()

safe_exit.register(cleanup_function)

def main():
    """
    main function for umbrella cli
    """
    global program
    parser = argparse.ArgumentParser(description='ipy oxdna ffs')
    parser.add_argument("--input", type=str, help="input file")
    # don't need more options, since use case is deliberately narrow

    args = parser.parse_args()
    config_yml_file: str = args.input
    if not config_yml_file.endswith(".yml"):
        raise ValueError("input file must be a .yml file")
    if not (Path(config_yml_file).is_file()):
        raise FileNotFoundError(f"input file {config_yml_file} not found")

    with open(config_yml_file, "r") as f:
        ffs_data = yaml.safe_load(f)
    
    Path(ffs_data["output_dir"]).mkdir(parents=True, exist_ok=True)

    program = FFSProgram(
        ffs_data["job_name"],
        Path(ffs_data["output_dir"]),
        ffs_data["num_cpus"],
        ffs_data["desired_n_successes"],
        Path(ffs_data["file_dir"])
    )
    # populate input - do immediately after creation
    for key, val in ffs_data["input_file_params"].items():
        program.input_file_params[key] = val
    program.ffs_default_input_name = "ffs_cuda" if ffs_data["use_cuda"] else "ffs"

    # Create order parameters
    order_params = {}
    for name, op in ffs_data["order_parameters"].items():
        order_params[name] = OrderParameter(name,
                                            op["order_parameter"],
                                            list(zip(op["nucleotide_indexes_0"],
                                                     op["nucleotide_indexes_1"])
                                                 )
                                            )

    # Setup interfaces
    interfaces = []
    for iface in ffs_data["interfaces"]:
        op = order_params[iface["op"]]
        threshold = iface["value"]
        comparison = Comparison(iface["compare"])
        interfaces.append(FFSInterface(op, threshold, comparison))
    program.set_interfaces(*interfaces)
    program.load()
    program.auto_save = True # automatically update process graph and csv files
    program.run()
    program.plot_graph()
    # save graph on exit
    cleanup_function()

if __name__ == "__main__":
    main()