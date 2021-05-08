# DrMZ

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://brekmeuris.github.io/DrMZ.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://brekmeuris.github.io/DrMZ.jl/dev)
[![Build Status](https://github.com/brekmeuris/DrMZ.jl/workflows/CI/badge.svg)](https://github.com/brekmeuris/DrMZ.jl/actions)
[![Coverage](https://codecov.io/gh/brekmeuris/DrMZ.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/brekmeuris/DrMZ.jl)


This package, Deep renormalized Mori-Zwanzig (DrMZ) contains the various submodules for generating data and training operator neural networks, finding custom basis functions, solving PDEs using the custom basis functions, and constructing reduced order models.

This package is in a pre-release state with each update potentially breaking. This module is set up to run on CPUs using double precision and during training will spawn as many threads as there are cores available. GPU support is available but needs testing and modification for single precision type consistency.


## Adding this package:

This package has been tested on the long-term support (LTS) release of Julia 1.0.5. [Download Julia 1.0.5](https://julialang.org/downloads/#long_term_support_release) and check out the [platform specific instructions](https://julialang.org/downloads/platform/) for instructions on adding Julia to PATH.

To add the package from GitHub link using Julia REPL:

``` ] add http_link_to_GitHub_repository ```

To add the package from direct download from GitHub using Julia REPL:

``` ] add path_to_GitHub_repository_download_location ```


## Accessing the development documentation:

[Development documentation](https://brekmeuris.github.io/DrMZ.jl/dev/) complete with working links to the source. Additionally, you can access the documentation from the Julia REPL:

```? name_of_function```

## Running an example operator neural network:

Under the ```examples``` directory there is an example script for the advection and advection-diffusion equation. The example script, ```advection_advection_diffusion_opnn.jl```, generates the data set, trains the neural network, and outputs a few results. To run this script, there are several additional packages that need to be installed/standalone installed in addition to the ```DrMZ``` package. To add these packages using Julia REPL:

``` ] add Flux, Parameters, LaTeXStrings, ColorSchemes, Plots, PyPlot ```

Once all the additional packages have been installed, the example script can be ran from a terminal session using:

```julia advection_advection_diffusion_opnn.jl```

The first time this script is ran it may take some time to indicate the dataset is being built as it will show a few warnings and install additional required packages.

## References:

The operator neural networks were implemented based on the structure presented by Lu Lu et al. (2021).

Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021). Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. Nature Machine Intelligence, 3(3), 218-229.
