"""
Car Price Prediction Library

Uma biblioteca interna de Machine Learning para predição de preços de carros.
Demonstra as melhores práticas de desenvolvimento de bibliotecas ML internas.

Versão: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "FIAP ML Team"

from .data_ingestion import DataIngestion
from .data_validation import DataValidator
from .model_trainer import ModelTrainer
from .model_evaluation import ModelEvaluator
from .utils import setup_logger, save_pickle, load_pickle

__all__ = [
    "DataIngestion",
    "DataValidator",
    "ModelTrainer",
    "ModelEvaluator",
    "setup_logger",
    "save_pickle",
    "load_pickle",
]
