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
        "matplotlib",
        "numpy",
        "pillow",
        "pyrender==0.1.39",
        "scikit-learn",
        "scipy",
        "torch==1.6.0",
        "tqdm",
        "pyglet==1.5.27",
        "PyOpenGL==3.1.0",
        "PyOpenGL-accelerate",
        "body_visualizer @ git+https://github.com/nghorbani/body_visualizer.git@be9cf756f8d1daed870d4c7ad1aa5cc3478a546c",
        "human_body_prior @ git+https://github.com/nghorbani/human_body_prior.git@0278cb45180992e4d39ba1a11601f5ecc53ee148",
    ],
    packages=find_packages(exclude=["tests"]),
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
