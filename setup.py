""" Setup
"""
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="labelshift",
    version="0.1.5",
    license="MIT",
    description="A Pytorch-based toolbox for label shift estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pm25/Label-Shift-Estimation",
    author="Pin-Yen Huang",
    author_email="pyhuang97@gmail.com",
    # Note that this is a string of words separated by whitespace, not a list.
    keywords="pytorch label-shift-estimation",
    packages=find_packages(exclude=["data", "configs"]),
    include_package_data=True,
    install_requires=["torch >= 1.8", "torchvision", "transformers", "cvxpy"],
    python_requires=">=3.7",
)
