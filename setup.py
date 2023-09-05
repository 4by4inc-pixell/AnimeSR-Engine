"""
setup script for pip install
"""

from setuptools import setup, find_packages


REQUIREMENTS = [
    "onnx",
    "onnxruntime-gpu",
    "ortei",
    "torch",
]

INCLUDE_MODULE = ["animesr_engine"]


setup(
    name="animesr-engine",
    version="0.1.0",
    packages=find_packages(include=INCLUDE_MODULE),
    install_requires=REQUIREMENTS,
)
