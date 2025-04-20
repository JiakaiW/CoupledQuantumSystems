# For CPU:
# FROM python:3.10
# RUN pip3 install --upgrade pip
# RUN pip3 install --upgrade "jax[cuda12]"
# RUN pip3 install git+https://github.com/JiakaiW/CoupledQuantumSystems

########################
# ----- Stage 1 -----  #
# Build Python deps    #
########################
FROM python:3.10-slim AS builder

# Install git for pip to clone GitHub repos
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        "git+https://github.com/JiakaiW/CoupledQuantumSystems#egg=CoupledQuantumSystems[jax]" && \
    pip cache purge && \
    find /usr/local/lib/python3.10/site-packages \
         -type d \( -name "tests" -o -name "__pycache__" \) -exec rm -rf {} +
########################
# ----- Stage 2 -----  #
# Thin CUDA runtime    #
########################
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# 2. Install the smallest possible Python footprint
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 python3.10-distutils libpython3.10 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu

# 3. Copy the ready‑made Python runtime from the builder
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin

# 4. Nice convenience symlinks
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# 5. Final image entrypoint
CMD ["python3"]


# Use: in terminal run
# docker build --no-cache -t coupledquantumsystems:v15 .