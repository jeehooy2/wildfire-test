from setuptools import setup, find_packages

setup(
    name="wildfire-environment",
    version="0.1.9",
    description="A gym-based multi-agent environment to simulate wildfire fighting",
    author="Wildfire Environment Team",
    packages=find_packages(),
    python_requires=">=3.8,<3.10",
    install_requires=[
        "gym==0.21.0",
        "numpy>=1.19.0,<1.24.0",
        "matplotlib>=3.3.0,<3.6.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
)

