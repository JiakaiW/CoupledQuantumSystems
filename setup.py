import os
from setuptools import setup, find_namespace_packages

README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md")
with open(README_PATH) as readme_file:
    README = readme_file.read()

setup(
    name="CoupledQuantumSystems",
    version="0.3",
    description="CoupledQuantumSystems",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/JiakaiW/CoupledQuantumSystems",
    author="Jiakai Wang",
    author_email="jwang2648@wisc.edu",
    license="Apache 2.0",
    packages=find_namespace_packages(exclude=['notebooks']),
    install_requires=[
        "scipy==1.12.0",
        "qiskit==1.4.2",
        "qiskit_dynamics==0.5.1",
        "numpy==1.26.4",
        "qutip==4.7.5",
        "scqubits==4.0.0",
        "loky",
        "bidict",
        "nevergrad",
        "rich"
    ],
    extras_require={
        'dev': ['pytest'],
        'jax': [
            "jax[cuda12]",
            "dynamiqs",
        ]
    },
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
    ],
    keywords="Hamiltonian simulation, quantum simulation",
    python_requires=">=3.7",
    project_urls={
        # "Documentation": "https://github.com/JiakaiW/CoupledQuantumSystems",
        "Source Code": "https://github.com/JiakaiW/CoupledQuantumSystems",
    },
    include_package_data=True,
)
