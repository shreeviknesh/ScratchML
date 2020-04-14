import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="scratchml", 
    version="0.1",
    author="Shreeviknesh Sankaran",
    author_email="shreeviknesh@gmail.com",
    description="A python package with implementations of Machine Learning algorithms from scratch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shreeviknesh/ScratchML",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

import shutil
shutil.rmtree('scratchml.egg-info')
shutil.rmtree('build')