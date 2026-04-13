# ============================================================
# Gaussian Phase Field Crack Simulation
# Supports: NVIDIA GPU (CUDA 12.4) + CPU fallback
# ============================================================
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ── System dependencies ──────────────────────────────────────
RUN apt-get update && apt-get install -y \
    wget curl git build-essential ninja-build \
    ffmpeg libgl1-mesa-glx libglib2.0-0 \
    libsm6 libxext6 libxrender-dev libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Miniconda ────────────────────────────────────────────────
ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# ── Conda environment ────────────────────────────────────────
WORKDIR /app
COPY environment_docker.yml .
RUN conda env create -f environment_docker.yml && \
    conda clean -afy

# Make conda env the default
ENV PATH=$CONDA_DIR/envs/crack_py11/bin:$PATH
ENV CONDA_DEFAULT_ENV=crack_py11

# ── CUDA paths for submodule compilation ────────────────────
ENV CUDA_HOME=$CONDA_DIR/envs/crack_py11
ENV CPATH=$CONDA_DIR/envs/crack_py11/targets/x86_64-linux/include:$CPATH
ENV LD_LIBRARY_PATH=$CONDA_DIR/envs/crack_py11/lib:$LD_LIBRARY_PATH

# ── Copy project ─────────────────────────────────────────────
COPY . .

# ── Build Gaussian Splatting submodules ──────────────────────
RUN pip install \
    gaussian-splatting/submodules/diff-gaussian-rasterization \
    gaussian-splatting/submodules/fused-ssim \
    gaussian-splatting/submodules/simple-knn \
    --no-build-isolation

# ── Output directory ─────────────────────────────────────────
RUN mkdir -p output

# ── Entrypoint ───────────────────────────────────────────────
ENTRYPOINT ["python", "cli.py"]
CMD ["--help"]
