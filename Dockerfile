FROM python:3.10

RUN pip3 install --upgrade pip
RUN pip3 install --upgrade "jax[cuda12]"
RUN pip3 install git+https://github.com/JiakaiW/CoupledQuantumSystems

# Use: in terminal run
# docker build --no-cache -t coupledquantumsystems:v11 .