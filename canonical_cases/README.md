# Canonical Cases

This directory contains canonical flame configurations used for low-dimensional manifold sampling in DFODE-kit. A key challenge in preparing training data is achieving sufficient coverage of the relevant thermochemical composition space, which is often prohibitively high-dimensional when detailed chemistry involves tens to hundreds of species. To address this, DFODE-kit adopts a low-dimensional manifold sampling strategy, where thermochemical states are extracted from canonical flame configurations that retain the essential topology of high-dimensional turbulent flames. This approach ensures both computational efficiency and physical representativeness of the training datasets.

## Types of Canonical Flames

The choice of canonical flames depends on the target application:
- **One-dimensional laminar premixed flames**: Provide representative states for turbulent premixed flame simulations.
- **Counterflow flames**: Employed for turbulent partially-premixed or diffusion cases.
- **Zero-dimensional auto-ignition reactors**: Used for ignition-dominated conditions.
- **One-dimensional detonation shock tubes**: Used for detonation-dominated conditions.

## Available Cases

**Note**: Currently, only one example case is provided in this repository. For a broader range of canonical flame configurations (e.g., counterflow flames, auto-ignition reactors, detonation tubes), refer to the examples in the [DeepFlame repository](https://github.com/deepmodeling/deepflame-dev/tree/master/examples). Contributions of additional cases are welcome!

- **oneD_freely_propagating_flame**: One-dimensional laminar premixed flame configuration for sampling states relevant to turbulent premixed combustion. Includes initialization scripts and OpenFOAM/DeepFlame setup files.

## Usage

These canonical cases are primarily used in the sampling stages of DFODE-kit's model development pipeline to generate high-quality thermochemical datasets from low-dimensional manifolds. They can also serve as benchmarks for model testing and validation in a posteriori simulations.

For step-by-step tutorials on using these cases, refer to the `tutorials/` directory.