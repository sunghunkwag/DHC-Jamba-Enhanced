"""
Setup script for DHC-Jamba-Enhanced
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dhc-jamba-enhanced",
    version="1.0.0",
    author="Sunghun Kwag",
    description="Hybrid Transformer-Mamba architecture for spatial-temporal modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sunghunkwag/DHC-Jamba-Enhanced",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "rl": [
            "gymnasium>=0.29.0",
            "mujoco>=3.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
)
