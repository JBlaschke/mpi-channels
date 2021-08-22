#!/usr/bin/env python
# -*- coding: utf-8 -*-


import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()



setuptools.setup(
    name="mpi-channels",
    version="0.1.0",
    author="Johannes Blaschke",
    author_email="johannes@blaschke.science",
    description="A RemoteChannel implementation built on top of MPI RMA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JBlaschke/mpi-channels",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    isinstanceall_requires=[
        "numpy",
        "mpi4py"
    ],
)
