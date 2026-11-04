
# How to include image in README.md

The most elegant way I found is to follow this post: [https://stackoverflow.com/a/42677655](https://stackoverflow.com/a/42677655).
At specific version, the image referred to is a fixed link.


# How to create a Jupyter notebook server in a docker for public access (collaboration)

```sh
docker pull ubuntu:latest
sudo docker run -itd --name mwpf-jupyter-server -p 8888:8888 -v /home/ubuntu/mwpf-jupyter/:/home/mwpf-jupyter --restart always ubuntu:latest
sudo docker exec -it mwpf-jupyter-server bash

# inside the bash
apt install python3 python3-pip wget vim -y
#     install conda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
#     install MWPF packages
pip3 install MWPF MWPF-rational --upgrade --force-reinstall
pip3 install jupyter jupyter-collaboration==2.1.5  # v3 has bug...
#     configure Jupyter lab to have consistent password every boot
conda install jupyter
jupyter lab --generate-config
vim ~/.jupyter/jupyter_lab_config.py  # modify it the following lines
# c.ServerApp.ip = '0.0.0.0'  # Allow access from any IP
# c.ServerApp.port = 8888  # Port to run on
# c.ServerApp.open_browser = False  # Do not open a browser on the server
# c.ServerApp.custom_display_url = 'jupyter.mwpf.dev'
# c.ServerApp.allow_root = True
jupyter notebook password  # set up password
#     create script to start the server automatically
vim /usr/local/bin/start-jupyter.sh  # put the following
# #!/bin/bash
# source ~/miniconda3/bin/activate
# jupyter lab --notebook-dir=/home/mwpf-jupyter
chmod +x /usr/local/bin/start-jupyter.sh

# on host machine, run the following to start the server if it's not already started
#     we'll still need a mechanism to automatically boot it
sudo docker exec -d mwpf-jupyter-server /bin/bash -c /usr/local/bin/start-jupyter.sh
```

## deploy latest package on the server

Enter the docker machine (`sudo docker exec -it mwpf-jupyter-server bash`) and
create a file `/usr/local/bin/compile-mwpf.sh` with the following content:

```sh
#!/bin/bash
source /root/.bashrc

cd /home/mwpf-jupyter/mwpf
git pull
maturin develop --release
```

Then we can run `sudo docker exec -it mwpf-jupyter-server compile-mwpf.sh` on the outer world.

# To save Github build resources

The most expensive build action is for macOS:
Your 2,420.00 included minutes used consists of 342.00 Ubuntu 2-core minutes, 328.00 Windows 2-core minutes, and 1,750.00 macOS 3-core minutes.

Since I'm using Mac for development already, I can run the following command to build it locally.

```sh
rustup target add x86_64-apple-darwin  # only once

CIBW_BUILD=cp38-* MACOSX_DEPLOYMENT_TARGET=10.12 CIBW_ARCHS_MACOS=universal2 python -m cibuildwheel --output-dir target

# build source distribution
maturin sdist --output-dir target
```

# working with HPC cluster

We use unison to synchronize development folder with HPC filesystem

A template of the unison configuration file is `hpc-mwpf.prf`

```ini
# Unison preferences
label = mwpf folder on HPC Grace Yale
root = /Users/wuyue/Documents/GitHub/mwpf
root = ssh://yw729@grace//home/yw729/project/mwpf

ignore = Regex (.*/)?node_modules(/.*)?
ignore = Regex (.*/)?target(/.*)?
ignore = Regex (.*/)?dist(/.*)?
#ignore = Regex (.*/)?slurm_jobs(/.*)?
ignore = Regex (.*/)?__pycache__(/.*)?
ignore = Name .DS_Store
ignore = Regex .*\.so

ignore = Regex .*/slurm_jobs/.*\.jobout
ignore = Regex .*/slurm_jobs/.*\.joberror

# log actions to the terminal
log = true
```

Then one could use `unison hpc-mwpf -auto -batch` to synchronize with the HPC environment.
In cases where one wants to prefer one direction over another, use `unison hpc-mwpf -auto -batch -prefer /Users/wuyue/Documents/GitHub/mwpf`.
When the number of conflicting files are small, use `unison hpc-mwpf` to manually resolve each conflict.

Once in the HPC environment, one can install the mwpf Python packages by compiling from source.

```sh
make install-py

# to uninstall:
make uninstall-py
```

This is preferred over installing from pypi because it can fully utilize the optimization with `target-cpu=native`.

## Python wheel coverage

The wheel workflow builds all four package variants for Linux, macOS universal2,
Windows x64, and Windows ARM64 using three ABI families:

| Wheel family | Supported CPython interpreters |
| --- | --- |
| `abi3` | Conventional Python 3.9 and newer |
| `cp314t` | Free-threaded Python 3.14 only |
| `abi3t` | Conventional and free-threaded Python 3.15 and newer |

Python 3.14t cannot load the other two families, so it needs a version-specific
`cp314-cp314t` wheel. PyO3 selects this ABI when building with the 3.14t
interpreter; the minimum supported Python version remains 3.9.

CI checks core decoding, pickling, and concurrent independent solvers on both
free-threaded versions, including an assertion that importing the module does
not enable the GIL. Windows ARM64 wheels are cross-compiled and cannot be
executed on the x64 build runners. Optional Stim integration depends on its
third-party dependencies' interpreter and free-threading support.

## Internal bundled HiGHS interface

Version 0.2.18 adds `mwpf._highs`, a small, minimization-only interface to the
HiGHS already compiled into every package variant, including Windows ARM64.
It needs neither `highspy` nor SciPy. It is not a full highspy replacement.

The interface supports the subset used by
[QDK's distance MILP backend](https://github.com/microsoft/qdk/blob/main/source/qdk_package/qdk/ec/_analysis/distance_solvers.py):
`Highs`, `HighsStatus`, `HighsModelStatus`, `kHighsInf`, binary/integer variables,
sparse bounded rows, and solution/status/dual-bound queries. A caller using
that subset can change its backend import to `mwpf._highs`. Supported options
are `output_flag`, `time_limit`, `mip_rel_gap`, and `mip_abs_gap`; unsupported
options or invalid inputs raise exceptions instead of being silently ignored.

```python
from mwpf import _highs as highs

model = highs.Highs()
x = model.addBinary(obj=1.0)
model.addRow(1.0, 1.0, 1, [x], [1.0])
model.run()
assert model.getModelStatus() == highs.HighsModelStatus.kOptimal
assert model.getSolution().value_valid
assert model.getSolution().col_value[x] == 1.0
assert model.getInfo().mip_dual_bound == 1.0
```

Check `getInfo().valid` and `getSolution().value_valid` before using bounds or
values after a limited solve. Infeasibility or a timeout without an incumbent
must not be treated as an all-zero witness. Mutating a model invalidates its
previous result. Independent instances may solve concurrently, with the
Python interpreter detached and HiGHS using the same single-thread scheduler
settings as the decoder; sharing a mutable model across threads is unsupported.
The same module is available as `mwpf_rational._highs`, `mwpf_incr._highs`, and
`mwpf_fast._highs`.

## Releasing to PyPI and crates.io

The [wheel workflow](../../.github/workflows/wheels.yml) publishes release-tag
builds to both registries only after all wheel, Rust, and Python test jobs
succeed. Separate publishing jobs release the `mwpf` Rust crate to crates.io
and the four Python packages to PyPI. Branch pushes build and test packages
but do not publish them.

Before creating a release tag, configure a GitHub Actions
[Trusted Publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
on each PyPI project: `mwpf`, `mwpf-rational`, `mwpf-incr`, and `mwpf-fast`.
Use these values for all four projects:

| Setting | Value |
| --- | --- |
| Repository owner | `yuewuo` |
| Repository name | `mwpf` |
| Workflow filename | `wheels.yml` |
| Environment name | `pypi` |

Also configure [Trusted Publishing](https://crates.io/docs/trusted-publishing)
under the `mwpf` crate's Settings on crates.io. Use the same repository owner,
repository name, and workflow filename, but set the environment name to
`crates-io`. The PyPI publisher configuration does not authorize crates.io
uploads. Restrict both GitHub environments to release tags, with approval
requirements if desired.

Both publish jobs use GitHub OIDC, so no permanent registry-token secrets are
required. The Rust job verifies the package before authenticating, obtains a
temporary crates.io token, and passes it only to `cargo publish --locked` through
`CARGO_REGISTRY_TOKEN`. This value is an authentication-action output, not a
GitHub secret.
The PyPI job explicitly enables attestations for every uploaded wheel and
source distribution. Once the publishers are configured, push a release tag
matching the shared package version, for example `v0.2.18`.

The two registry uploads are not atomic: one can succeed while the other fails.
Prefer rerunning only failed publishing jobs. The Rust job skips an existing
version only if its archive checksum matches the verified local package;
a different checksum or a registry lookup error fails instead of masking the
problem. PyPI keeps `skip-existing` enabled for partial-upload retries.

[Provenance attestations](https://docs.pypi.org/attestations/producing-attestations/)
are optional metadata, not a requirement for a valid wheel. They are attached
when a file is uploaded; enabling them does not retroactively add provenance
to existing releases or files skipped by `skip-existing`.
