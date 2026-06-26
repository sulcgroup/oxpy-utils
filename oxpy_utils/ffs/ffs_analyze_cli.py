import shutil
from pathlib import Path

import yaml

from oxpy_utils.ffs.ffs_program import FFSProgram
import argparse

# use with the same yml input as ffs_cli, but instead of running the program, just analyze and output results


def main():
    parser = argparse.ArgumentParser(description='ipy oxdna ffs analysis')
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

    if not Path(ffs_data["output_dir"]).exists():
        raise FileNotFoundError(f"output directory {ffs_data['output_dir']} not found")

    program = FFSProgram(
        ffs_data["job_name"],
        Path(ffs_data["output_dir"]),
        ffs_data["num_cpus"],
        ffs_data["desired_n_successes"],
        Path(ffs_data["file_dir"])
    )
    program.load()
    program.load_graph()
    # export individual shooter csv files
    for shooter in program.shooters:
        if (shooter.destination_directory / f"{shooter.name}_success_log.csv").is_file():
            shutil.copy(shooter.destination_directory / f"{shooter.name}_success_log.csv",
                        Path(ffs_data["output_dir"]) / f"{shooter.name}_success_log.csv")

    # export overall results to results.csv
    program.export_results()

    # export graph to process_graph.svg
    program.plot_graph()

if __name__ == "__main__":
    main()
