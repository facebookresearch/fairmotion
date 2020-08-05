# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

setup(
    name="fairmotion",
    version="0.0.2",
    description="",
    url="https://github.com/fairinternal/mocap_processing",
    author="FAIR Pittsburgh",
    author_email="dgopinath@fb.com",
    install_requires=[
        "black",
        "jupyter",
        "matplotlib",
        "numpy",
        "pillow",
        "pyglet",
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
