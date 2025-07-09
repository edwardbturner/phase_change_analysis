from setuptools import setup, find_packages

setup(
    name="phase_change_analysis",
    version="0.1.0",
    description="Analysis of phase transitions in neural network training",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.60.0",
    ],
    python_requires=">=3.8",
)