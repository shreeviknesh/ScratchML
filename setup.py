""" This file is used to create the binaries of the ScratchML package """
import os
import shutil
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

if 'dist' in os.listdir():
    shutil.rmtree('dist')

setup(
    name="scratchml",
    version="0.4",
    author="Shreeviknesh Sankaran",
    author_email="shreeviknesh@gmail.com",
    description="A python package with implementations of Machine Learning algorithms from scratch.",
    license="MIT",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/shreeviknesh/ScratchML",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.6',
)

shutil.rmtree('scratchml.egg-info')
shutil.rmtree('build')
