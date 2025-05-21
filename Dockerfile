# ###############################################################################
# # CoupledQuantumSystems – CPU container
# ###############################################################################

# FROM python:3.10

# RUN pip3 install --upgrade pip
# RUN pip3 install .





###############################################################################
# CoupledQuantumSystems – CUDA-12 container, fully pinned by requirements.lock
###############################################################################

# ---------- build stage ------------------------------------------------------
    ARG CUDA_TAG=12.8.0          # bump only when you really need a newer minor
    FROM nvidia/cuda:${CUDA_TAG}-devel-ubuntu22.04 AS builder
    
    # Base OS tooling
    ENV DEBIAN_FRONTEND=noninteractive
    RUN apt-get update && \
        apt-get install -y --no-install-recommends \
            python3.10 python3.10-venv python3-pip git && \
        rm -rf /var/lib/apt/lists/*
    
    # Upgrade pip & install exact wheels listed in requirements.lock
    COPY requirements.lock /tmp/requirements.lock
    RUN python3 -m pip install --upgrade pip && \
        # -------- strict pinning: -r == must install, -c == constrain everything
        python3 -m pip install --no-cache-dir \
            -r /tmp/requirements.lock \
            -c /tmp/requirements.lock \
            -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    
    # Add the current project source and build it *inside* the pinned environment
    COPY . /opt/CoupledQuantumSystems
    WORKDIR /opt/CoupledQuantumSystems
    RUN python3 -m pip install --no-cache-dir ".[jax]"
    
    # ---------- runtime stage (slimmer) ------------------------------------------
    FROM nvidia/cuda:${CUDA_TAG}-runtime-ubuntu22.04
    
    COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
    COPY --from=builder /usr/local/bin /usr/local/bin
    ENV PYTHONPATH=/usr/local/lib/python3.10/site-packages
    
    # JAX memory flags – avoid surprise OOMs
    ENV XLA_PYTHON_CLIENT_PREALLOCATE=false \
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
    
    WORKDIR /workspace                # bind-mount your scripts here
    CMD ["python3"]
    


# # Build
# docker build -t coupledquantumsystems:gpu .

# # After building, if we want to run with GPU access and your code directory mounted
# docker run --gpus all -it --rm -v $PWD:/workspace -w /workspace coupledquantumsystems:gpu
