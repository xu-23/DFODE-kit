# Tutorials

This directory provides step-by-step tutorials for using DFODE-kit to develop and deploy neural network models for accelerating combustion kinetics simulations. The tutorials guide you through the complete pipeline: from data sampling and augmentation to model training and testing. We provide two example cases to demonstrate the workflow for different flame configurations.

## Pipeline Overview

DFODE-kit's workflow consists of the following stages, as detailed in the manuscript:

1. **Data Sampling**: Extract thermochemical states from canonical flame simulations using low-dimensional manifold sampling. This ensures coverage of high-dimensional composition spaces efficiently.

2. **Data Augmentation and Labeling**: Enrich datasets with physics-constrained perturbations to approximate turbulent conditions, then generate supervised labels using Cantera's CVODE solver.

3. **Model Training**: Train a neural network (e.g., MLP with GELU activations) to predict state changes, incorporating constraints for mass/energy conservation.

4. **Model Testing**: Evaluate the model via a priori (single-step predictions) and a posteriori (full CFD simulations) validations.

5. **Model Deployment**: Integrate the trained model into CFD solvers like DeepFlame for accelerated chemistry integration.

Each tutorial includes Jupyter notebooks, scripts, and example data to walk you through these stages.

## Available Tutorials

### 1. One-Dimensional Freely Propagating Flame (`oneD_freely_propagating_flame/`)

This tutorial demonstrates the pipeline for a 1D laminar premixed hydrogen/air flame under premixed conditions, validating model accuracy in reproducing flame propagation behavior. This example samples data from a single 1D flame simulation and performs a posteriori validation on the same simulation case.

- **1_sample_train/**: Covers sampling, augmentation, labeling, and training.
  - `dfode_kit_init.ipynb`: Initialize simulation parameters (e.g., equivalence ratio, temperature, pressure) via `config_dict`.
  - `Allrun`: Execute the DeepFlame simulation to generate canonical flame data.
  - Sampling: Use `dfode-kit sample` to extract states into HDF5 format (with `scalar_fields` and `mesh` groups).
  - Augmentation: Apply `dfode-kit augment` for perturbed datasets with element ratio constraints.
  - Labeling: Run `dfode-kit label` to compute ODE solutions with CVODE.
  - Training: Execute `dfode-kit train` to train the MLP model with physical constraints.
  - `dfode_kit_tutorial.ipynb`: Comprehensive Jupyter notebook providing a step-by-step guide through the entire pipeline, including code execution, explanations of each stage, and instructions on both python interface and command line utilities.

- **2_model_test/**: Test the trained model.
  - `priori/`: A priori testing via single-step predictions on labeled datasets to evaluate model accuracy against ground-truth ODE solutions from Cantera CVODE.
  - `posteriori/`: A posteriori validation by integrating the trained model into full CFD simulations, comparing flame propagation and structure against direct CVODE integration.

### 2. Two-Dimensional HIT Flame (`twoD_HIT_flame/`)

This tutorial evaluates the DNN model in a 2D propagating premixed hydrogen/air flame within homogeneous isotropic turbulence (HIT), assessing turbulence-chemistry interactions beyond laminar regimes. The setup follows configurations for capturing flame front wrinkling and turbulent burning velocities. This example samples data from multiple 1D flame simulations and performs a posteriori validation on a separate 2D HIT case.

- **1_sample_train/**: Similar pipeline as above, adapted for 2D HIT.
  - Includes initialization, simulation, sampling, augmentation, labeling, and training for HIT conditions.
  - `dfode_kit_tutorial.ipynb`: Comprehensive guide.

- **2_model_test/**: Model testing in turbulent scenarios.
  - `priori/`: Single-step predictions.
  - `posteriori/`: Full turbulent flame simulations.

## Getting Started

To get started with the tutorials, refer to `dfode_kit_tutorial.ipynb` in each `1_sample_train/` subdirectory for step-by-step guidance through the sampling, augmentation, labeling, and training steps.

Note that running the simulations requires DeepFlame to be installed. Refer to the [DeepFlame GitHub repository](https://github.com/deepmodeling/deepflame-dev) and [documentation](https://deepflame.deepmodeling.com/en/latest/) for installation instructions.