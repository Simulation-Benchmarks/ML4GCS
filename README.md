# ML4GCS

[![REUSE status](https://api.reuse.software/badge/github.com/Simulation-Benchmarks/ML4GCS)](https://api.reuse.software/info/github.com/Simulation-Benchmarks/ML4GCS)

To access data and scripts interactively, open this repo on the NFDI JupyterHub:
[![NFDI](https://nfdi-jupyter.de/images/nfdi_badge.svg)](https://hub.nfdi-jupyter.de/v2/gh/Simulation-Benchmarks/ML4GCS/HEAD?system=JSC-Cloud&flavor=xl1nfdi&labpath=notebooks%2Fstart.ipynb)

To work locally, clone this repository and open the [start notebook](notebooks/start.ipynb) in a local JupyterLab session. The required data will then be downloaded on demand.
This repo uses scripting from the [11thSPE-CSP repo](https://github.com/Simulation-Benchmarks/11thSPE-CSP). If you work locally, that repo should be cloned on the same folder level as the one here.

# Access to calculating Wasserstein distance

Wasserstein distances are enabled through the package [DarSIA](https://github.com/pmgbergen/darsia). Access is enabled through a submodule. To clone and install the submodule with Python 3.13:

   ```bash
   git submodule update --init --recursive # alternatively: 
   # git submodule add https://github.com/pmgbergen/darsia external/darsia
   uv python install 3.13
   uv sync
   ```

`darsia` is then sourced from the local git submodule at `external/darsia` and installed by uv during `uv sync`.