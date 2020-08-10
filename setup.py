# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

setup(
    name="fairmotion",
    version="0.0.3",
    description="fairmotion is FAIR's library for human motion research",
    url="https://github.com/fairinternal/fairmotion",
    author="FAIR Pittsburgh",
    author_email="dgopinath@fb.com",
    install_requires=[
        "black",
        "human_body_prior",
        "matplotlib",
        "numpy",
        "pillow",
        "pyrender",
        "scikit-learn",
        "scipy",
        "torch",
        "tqdm",
        "PyOpenGL==3.1.0",
    ],
    packages=find_packages(exclude=["tests"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
