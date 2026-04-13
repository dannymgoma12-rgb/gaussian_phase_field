# Gaussian Phase Field Crack Simulation

**MPM + Gaussian Splatting** simulation of crack propagation using the AT2 phase field damage model. Combines Material Point Method (MPM) physics with differentiable Gaussian Splatting rendering to visualize fracture in real time.

**GitHub:** https://github.com/dannymgoma12-rgb/gaussian_phase_field

---

## Table of Contents

- [What This Does](#what-this-does)
- [Requirements](#requirements)
- [Quick Start — Docker (Recommended)](#quick-start--docker-recommended)
- [Quick Start — Local (Linux / WSL2)](#quick-start--local-linux--wsl2)
- [CLI Reference](#cli-reference)
- [Docker Advanced Usage](#docker--advanced-usage)
- [Output Structure](#output-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## What This Does

This project simulates crack propagation in 3D meshes (`.obj` files) using two combined techniques:

- **MPM (Material Point Method):** A continuum mechanics solver that tracks particle-based deformation and damage through the AT2 phase field model.
- **Gaussian Splatting:** A differentiable rendering technique that visualizes the particle field as a high-quality video without mesh reconstruction.

The result is a per-frame rendered video of your mesh fracturing under configurable seismic or mechanical loading.

---

## Requirements

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.11 | Required; use Miniconda |
| PyTorch | 2.5.1 | GPU build for CUDA 12.4 |
| CUDA | 12.4 (optional) | Only needed for GPU acceleration |
| Docker | 24.0+ | Required for Docker path |
| NVIDIA Driver | 550+ | Only needed for GPU in Docker |
| FFmpeg | any | For video generation |

---

## Quick Start — Docker (Recommended)

Docker is the easiest way to run this on any OS (Windows via WSL2, macOS, Linux). It installs all dependencies automatically.

**Prerequisites:**
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- (Optional) NVIDIA GPU with driver 550+ and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

```bash
# 1. Clone the repo
git clone https://github.com/dannymgoma12-rgb/gaussian_phase_field.git
cd gaussian_phase_field

# 2. Build the Docker image (takes ~10 minutes the first time)
docker compose build

# 3a. Run with GPU (if you have an NVIDIA GPU)
docker compose --profile gpu up

# 3b. Run with CPU only (no GPU required)
docker compose --profile cpu up
```

Output will appear in the `output/` folder on your host machine automatically (Docker mounts it as a volume).

**To stop a running simulation:**
```bash
# Press Ctrl+C in the terminal, or in another terminal:
docker compose down
```

---

## Quick Start — Local (Linux / WSL2)

Use this if you want to run without Docker, or if you want to develop and modify the code.

**Prerequisites:**
- Linux or Windows with WSL2 (Ubuntu 22.04 recommended)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- NVIDIA GPU + CUDA drivers (optional, CPU fallback works)

```bash
# 1. Clone the repo
git clone https://github.com/dannymgoma12-rgb/gaussian_phase_field.git
cd gaussian_phase_field

# 2. Create the conda environment (installs Python, PyTorch, all dependencies)
conda env create -f environment_docker.yml
conda activate crack_py11

# 3. Set environment variables needed to build CUDA extensions
export CUDA_HOME=$CONDA_PREFIX
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH

# 4. Build the three Gaussian Splatting submodules (required)
pip install gaussian-splatting/submodules/diff-gaussian-rasterization --no-build-isolation
pip install gaussian-splatting/submodules/fused-ssim --no-build-isolation
pip install gaussian-splatting/submodules/simple-knn --no-build-isolation

# 5. Verify your environment
python cli.py info

# 6. Run a simulation
python cli.py run
```

> **WSL2 note:** If you are using VS Code with the WSL extension, open the repo folder in WSL (`code .` from inside WSL), and run all commands in the WSL terminal — not the Windows PowerShell.

---

## CLI Reference

All interaction with the simulation is through `cli.py`:

```bash
python cli.py <command> [options]
```

---

### `info` — Check your environment

Run this first to confirm your GPU, Python packages, and project structure are set up correctly.

```bash
python cli.py info
```

Expected output includes GPU name, PyTorch version, and a list of available configs/meshes. If any package shows `NOT INSTALLED`, see [Troubleshooting](#troubleshooting).

---

### `validate` — Check a config file

Before running a simulation, validate your config to catch missing keys or bad mesh paths.

```bash
python cli.py validate --config configs/simulation_config.yaml
```

A passing validation prints `✓ Config is valid!`. A failure prints the specific missing keys or files.

---

### `run` — Run a single simulation

```bash
# Default (auto GPU/CPU detection, uses simulation_config.yaml)
python cli.py run

# Custom mesh and resolution
python cli.py run \
  --mesh assets/meshes/bunny.obj \
  --particles 50000 \
  --frames 100 \
  --output output/my_run

# Force CPU mode (useful if GPU is causing errors)
python cli.py run --device cpu --frames 50

# Quick preview — render every 5th frame, skip video
python cli.py run --frames 20 --render-every 5 --no-video
```

**All options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--config` | `configs/simulation_config.yaml` | Path to YAML config file |
| `--device` | `auto` | `auto` (detect GPU), `cuda`, or `cpu` |
| `--mesh` | from config | Path to `.obj` mesh file |
| `--particles` | from config | Number of MPM particles (more = slower, more detail) |
| `--frames` | from config | Number of simulation frames to run |
| `--output` | from config | Output directory path |
| `--render-every` | `1` | Only render every N frames (speeds up previews) |
| `--no-video` | off | Skip final video assembly (saves time for previews) |
| `--verbose` | off | Print detailed per-frame logging |

---

### `batch` — Run multiple simulations

Useful for running a sweep of resolutions or parameters automatically.

```bash
python cli.py batch --batch-config configs/batch_runs.json

# Continue even if one run fails
python cli.py batch --batch-config configs/batch_runs.json --continue-on-error
```

Edit `configs/batch_runs.json` to define your runs:

```json
{
  "runs": [
    {
      "name": "low_res_test",
      "particles": 10000,
      "frames": 30,
      "output": "output/batch/low_res"
    },
    {
      "name": "high_res_final",
      "particles": 50000,
      "frames": 100,
      "output": "output/batch/high_res"
    }
  ]
}
```

Each run inherits from the base config and can override any parameter individually.

**Batch options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--batch-config` | required | Path to batch JSON file |
| `--base-config` | `configs/simulation_config.yaml` | Fallback config for runs that don't specify one |
| `--continue-on-error` | off | Don't stop the batch if a run fails |
| `--verbose` | off | Verbose output for all runs |

---

## Docker — Advanced Usage

```bash
# Open an interactive shell inside the container (useful for debugging)
docker compose --profile shell run crack-shell

# Run CLI with custom params inside the GPU container
docker compose --profile gpu run crack-gpu run \
  --particles 50000 --frames 100

# Run CLI in CPU-only mode
docker compose --profile cpu run crack-cpu run \
  --device cpu --particles 10000 --frames 30

# Run a batch job in Docker
docker compose --profile gpu run crack-gpu batch \
  --batch-config configs/batch_runs.json

# Force rebuild the image (after changing Dockerfile or environment)
docker compose build --no-cache
```

---

## Output Structure

After a successful run, your output directory will contain:

```
output/
├── crack_simulation.mp4       # Final rendered video
├── statistics.csv             # Per-frame metrics (damage, particle count, etc.)
├── frames/                    # Individual PNG frames
│   ├── frame_0000.png
│   ├── frame_0001.png
│   └── ...
└── diagnostics/               # Debug data for each frame
    ├── diag_frame_0000.npz
    └── ...
```

The `.npz` diagnostic files can be loaded with NumPy for custom analysis:
```python
import numpy as np
data = np.load("output/diagnostics/diag_frame_0000.npz")
print(data.files)  # list available arrays
```

---

## Configuration

Edit `configs/simulation_config.yaml` to customize the simulation:

```yaml
simulation: crack_simulation
mesh: assets/meshes/bunny.obj   # path to your .obj mesh
particles: 50000                 # total MPM particles
frames: 100                      # frames to simulate

# Material properties
young_modulus: 1.0e6             # stiffness (Pa)
poisson_ratio: 0.35              # lateral vs axial deformation ratio
Gc: 10.0                         # fracture toughness (critical energy release rate)
l0: 0.035                        # phase field regularization length

# Loading (what force causes the crack)
loading:
  type: seismic
  amplitude: 1500.0              # force amplitude
  frequency: 80.0                # oscillation frequency (Hz)
  direction: [1.0, 0.3, 0.0]    # force direction vector (x, y, z)
```

**Tips for tuning:**
- Lower `Gc` → cracks form more easily
- Lower `l0` → thinner crack bands (requires more particles for accuracy)
- Increase `amplitude` → more aggressive loading, faster fracture
- More `particles` → higher resolution but slower simulation

---

## Troubleshooting

### `python cli.py info` shows missing packages

If packages like `DiffGauss`, `SimpleKNN`, or `FAISS` show `NOT INSTALLED`, you need to build the submodules:

```bash
conda activate crack_py11
export CUDA_HOME=$CONDA_PREFIX
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH
pip install gaussian-splatting/submodules/diff-gaussian-rasterization --no-build-isolation
pip install gaussian-splatting/submodules/fused-ssim --no-build-isolation
pip install gaussian-splatting/submodules/simple-knn --no-build-isolation
```

---

### No GPU detected / CUDA errors

```bash
# Check GPU status
python cli.py info

# Fall back to CPU
python cli.py run --device cpu
```

If you have an NVIDIA GPU but it's not detected:
```bash
# Check that CUDA is visible to PyTorch
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

# Check your driver version
nvidia-smi
```

Your NVIDIA driver must be version 550 or later for CUDA 12.4. Update it from [nvidia.com/drivers](https://www.nvidia.com/Download/index.aspx).

---

### Docker GPU not working

```bash
# Test if Docker can see your GPU at all
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If that fails, install the NVIDIA Container Toolkit:
```bash
# Ubuntu / WSL2
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

---

### Docker: `image not found` or build errors

```bash
# Rebuild from scratch
docker compose build --no-cache

# Check that Docker Desktop is running
docker info
```

---

### Submodule build errors (`ModuleNotFoundError` or `ninja` errors)

Make sure ninja and cmake are installed in the conda env:
```bash
conda activate crack_py11
conda install ninja cmake -y
export CUDA_HOME=$CONDA_PREFIX
pip install gaussian-splatting/submodules/diff-gaussian-rasterization --no-build-isolation
```

If you see `CUDA extension failed to build`, your CUDA version may not match the PyTorch build. Check:
```bash
python -c "import torch; print(torch.version.cuda)"
nvcc --version
```

Both should show `12.4`. If not, reinstall the environment:
```bash
conda deactivate
conda env remove -n crack_py11
conda env create -f environment_docker.yml
```

---

### WSL2: `code .` opens in Windows instead of WSL

Make sure the VS Code WSL extension is installed, then:
```bash
# From inside your WSL terminal, in the repo directory:
cd ~/gaussian_phase_field
code .
```

This opens VS Code attached to WSL. All terminals in VS Code will then be WSL terminals.

---

### Output video not generating

FFmpeg is required for video generation. Check:
```bash
python cli.py info  # should show FFmpeg version
ffmpeg -version
```

If missing, install it:
```bash
conda activate crack_py11
conda install ffmpeg -y
```

Or skip video generation and just get frames:
```bash
python cli.py run --no-video
```

---

## Citation

If you use this project in academic work, please cite:

```bibtex
@misc{gaussian_phase_field,
  title  = {Gaussian Phase Field Crack Simulation},
  author = {Danny Gomez},
  year   = {2025},
  url    = {https://github.com/dannymgoma12-rgb/gaussian_phase_field}
}
```
