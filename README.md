# ipy_oxDNA
<center>
<img src="oxDNA.png">
</center>

This repository contains additional wrappers and utilities for the oxpy python package, which is associated with [oxDNA](https://github.com/lorenzo-rovigatti/oxDNA). Much of the code in this repository was derived from ipy_oxDNA, which was published in association with the following paper: 
Sample, Matthew, Michael Matthies, and Petr Šulc. "Hairygami: Analysis of DNA Nanostructure's Conformational Change Driven by Functionalizable Overhangs." arXiv preprint arXiv:2302.09109 (2023).

## Contents
- [Introduction](#introduction)
- [NVIDIA Multiprocessing Service (mps)](#how-to-run-nvidia-multiprocessing-service-mps)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Example Notebooks](#example-notebooks)
- [Contributing](#contributing)

## Introduction
oxDNA is a molecular dynamics simulation code that can be used to study the mechanical and thermodynamic properties of DNA and RNA molecules. Umbrella sampling is a highly parallelizable simulation technique that is used to calculate the free energy profiles between two particles or groups of particles. The `ipy_oxDNA` repository provides a Python interface for running oxDNA umbrella sampling simulations, allowing users to easily perform these simulations and analyze their results.

## How to run NVIDIA Multiprocessing Service (mps)
NVIDIA MPS is a specialized service offered by NVIDIA, designed to enhance the multiprocessing capabilities of CUDA-enabled GPUs. Utilizing MPS allows for the execution of multiple simulations on a single GPU, resulting in an approximate **2.5x performance increase** for specific simulation techniques such as *Umbrella Sampling, Metadynamics, and Multi-Replica simulations*.

In the absence of MPS, running concurrent simulations on a single GPU can lead to significant performance degradation. For comprehensive details on MPS, the official documentation is available at [NVIDIA MPS Documentation](https://docs.nvidia.com/deploy/mps/index.html).

```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-pipe_$SLURM_TASK_PID
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps-log_$SLURM_TASK_PID
mkdir -p $CUDA_MPS_PIPE_DIRECTORY
mkdir -p $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d
```

### Letting `SimulationManager` handle it

If you run your simulations through `SimulationManager`, you don't need to start MPS yourself:
`worker_manager()` starts a private MPS daemon for the batch, runs the queued simulations as
concurrent worker processes spread evenly over the visible GPUs, and stops the daemon (and removes
its pipe directory) afterwards, even if the batch fails.

```python
mgr = SimulationManager(max_sims_per_gpu=8)   # n_processes defaults to the CPUs available
for sim in sims:
    mgr.queue_sim(sim)
failures = mgr.worker_manager(run_when_failed=True)  # blocks; returns the error messages of failed sims
mgr.shutdown()
```

- GPUs are assigned to the simulation by `CUDA_device`, counting from the first entry of
  `CUDA_VISIBLE_DEVICES` if you have set it (e.g. a slurm allocation of GPUs 2 and 3 gives devices 0 and 1).
- `max_sims_per_gpu` defaults to 48 while MPS is in use, which is NVIDIA's documented client limit for
  Volta and newer (60 was measured on an RTX 5090). Past the limit new CUDA contexts fail to connect.
- A simulation whose worker process dies without reporting (killed, segfault, CUDA failure) counts as
  failed. With the default `run_when_failed=False` the first failure terminates the whole batch and
  exits; with `run_when_failed=True` the other simulations carry on.
- If `CUDA_MPS_PIPE_DIRECTORY` already points at a running daemon (e.g. the commands above, in your
  job script) it is used as is and left running. If `nvidia-cuda-mps-control` is missing or doesn't
  respond, a warning is issued and the simulations run without MPS. Pass `use_mps=False` to skip MPS.
- The daemon's logs are kept in `/tmp/mps-log_<pid>_<id>` (the path is printed); look there if
  simulations fail to start on the GPU.

## Prerequisites

We suggest installing in a [conda](https://www.anaconda.com/download) or [mamba](https://mamba.readthedocs.io/en/latest/index.html) environment.

**oxpy and oxDNA_analysis_tools** are installed together from the [oxDNA](https://lorenzo-rovigatti.github.io/oxDNA/install.html#python-bindings) source build.

oxpy_utils also includes bindings for [tacoxDNA](https://github.com/lorenzo-rovigatti/tacoxDNA), a collection of conversion scripts for the oxDNA file format.

**nupack** is required for sequence design features and must be installed separately:

```bash
pip install nupack -f https://nupack.org/download/latest
```

## Installation

Clone the repository and install:

```bash
git clone https://github.com/sulcgroup/oxpy-utils.git
pip install -e .
```

## Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request.
