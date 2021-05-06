# DrMZ

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://brekmeuris.github.io/DrMZ.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://brekmeuris.github.io/DrMZ.jl/dev)
[![Build Status](https://github.com/brekmeuris/DrMZ.jl/workflows/CI/badge.svg)](https://github.com/brekmeuris/DrMZ.jl/actions)
[![Coverage](https://codecov.io/gh/brekmeuris/DrMZ.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/brekmeuris/DrMZ.jl)


This package, Deep renormalized Mori-Zwanzig (DrMZ) contains the various submodules for generating data and training operator neural networks, finding custom basis functions, solving PDEs using the custom basis functions, and constructing reduced order models.

This repository is in a pre-release state with each update potentially breaking. This module is set up to run on CPUs using double precision and during training will spawn as many threads as there are cores available. GPU support is available but needs testing and modification for single precision type consistency.

## Adding this package:
 
To add the package from GitHub link using Julia REPL:

``` ] add http_link_to_GitHub_repository ```

To add the package from direct download from GitHub using Julia REPL:

``` ] add path_to_GitHub_repository_download_location ```

## Accessing the in-progress documentation:

[Documentation](https://brekmeuris.github.io/DrMZ.jl/) complete with working links to the source.

## Running an example operator neural network:

Under the ```examples\advection_advection_diffusion_opnn``` directory there is an example script for the advection and advection-diffusion equation. The example script, ```advection_advection_diffusion_opnn.jl```, generates the data set, trains the neural network, and outputs a few results. To run this script, there are several additional packages that need to be installed in addition to the ```DrMZ``` package. To add these packages using Julia REPL:

``` ] add Parameters, LaTeXStrings, ColorSchemes, Plots, PyPlot ```
