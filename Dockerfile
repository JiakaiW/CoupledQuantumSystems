# For CPU:
# FROM python:3.10
# RUN pip3 install --upgrade pip
# RUN pip3 install --upgrade "jax[cuda12]"
# RUN pip3 install git+https://github.com/JiakaiW/CoupledQuantumSystems

# For GPU:
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
# Install Python 3.10 from deadsnakes PPA
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-venv python3.10-distutils curl git && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
# Install JAX and your package
RUN python3.10 -m pip install --upgrade pip && \
    python3.10 -m pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    python3.10 -m pip install git+https://github.com/JiakaiW/CoupledQuantumSystems && \
    rm -rf ~/.cache/pip
RUN ln -sf /usr/bin/python3.10 /usr/bin/python
CMD [ "python3.10" ]

# Use: in terminal run
# docker build --no-cache -t coupledquantumsystems:v11 .