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
