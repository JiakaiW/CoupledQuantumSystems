# For CPU:
# FROM python:3.10
# RUN pip3 install --upgrade pip
# RUN pip3 install --upgrade "jax[cuda12]"
# RUN pip3 install git+https://github.com/JiakaiW/CoupledQuantumSystems

# For GPU:
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python 3.10 and basic tools
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-distutils curl git && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Symlink python3.10 as default python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    python3 -m ensurepip

# Upgrade pip and install everything directly into system environment
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    python3 -m pip install --force-reinstall --no-cache-dir git+https://github.com/JiakaiW/CoupledQuantumSystems

RUN python3 -c "import jax, pyparsing; print('JAX:', jax.__version__, '| pyparsing:', pyparsing.__version__)"


# Use: in terminal run
# docker build --no-cache -t coupledquantumsystems:v12 .