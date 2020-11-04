# Copyright (c) Facebook, Inc. and its affiliates.

from setuptools import find_packages, setup

setup(
    name="fairmotion",
    version="0.0.4",
    description="fairmotion is FAIR's library for human motion research",
    url="https://github.com/facebookresearch/fairmotion",
    author="FAIR Pittsburgh",
    author_email="dgopinath@fb.com",
    install_requires=[
        "black",
        "dataclasses",  # py3.6 backport required by human_body_prior
        "human_body_prior",
        "matplotlib",
        "numpy",
        "pillow",
        "pyrender==0.1.39",
        "scikit-learn",
        "scipy",
        "torch==1.6.0",
        "tqdm",
        "PyOpenGL==3.1.0",
    ],
    packages=find_packages(exclude=["tests"]),
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
