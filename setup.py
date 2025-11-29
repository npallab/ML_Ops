from setuptools import setup, find_packages

with open("README.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="ml_ops_project",
    version="0.1.0",
    packages=find_packages(),
    author="Paul",
    install_requires=requirements,
)