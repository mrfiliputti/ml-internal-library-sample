"""
Setup script para instalação da biblioteca car-price-prediction.

Este arquivo permite instalar a biblioteca usando pip:
    pip install .
    pip install -e .  # modo desenvolvimento
"""

from setuptools import setup, find_packages
from pathlib import Path

# Lê o README para usar como descrição longa
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="car-price-prediction",
    version="1.0.0",
    author="FIAP ML Team",
    author_email="ml-team@fiap.com",
    description="Biblioteca interna de ML para predição de preços de carros",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fiap/car-price-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "flake8>=4.0.0",
            "black>=22.0.0",
            "pylint>=2.12.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
        ],
    },
    keywords="machine-learning regression car-price prediction",
    project_urls={
        "Bug Tracker": "https://github.com/fiap/car-price-prediction/issues",
        "Documentation": "https://github.com/fiap/car-price-prediction/docs",
        "Source Code": "https://github.com/fiap/car-price-prediction",
    },
)
