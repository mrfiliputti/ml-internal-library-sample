"""
Módulo de Utilidades

Funções auxiliares reutilizáveis para o projeto de ML.
Segue princípios DRY e modularidade (Aula 2).
"""

import logging
import pickle
from pathlib import Path
from typing import Any


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Configura e retorna um logger personalizado.

    Parameters
    ----------
    name : str
        Nome do logger (geralmente __name__ do módulo).
    level : int, optional
        Nível de logging (default: logging.INFO).

    Returns
    -------
    logging.Logger
        Objeto logger configurado.

    Examples
    --------
    >>> logger = setup_logger(__name__)
    >>> logger.info("Processo iniciado")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def save_pickle(obj: Any, filepath: str) -> None:
    """
    Salva um objeto Python em arquivo pickle.

    Parameters
    ----------
    obj : Any
        Objeto Python a ser salvo.
    filepath : str
        Caminho do arquivo de destino.

    Raises
    ------
    IOError
        Se houver erro ao salvar o arquivo.

    Examples
    --------
    >>> model = LinearRegression()
    >>> save_pickle(model, "models/model.pkl")
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str) -> Any:
    """
    Carrega um objeto Python de arquivo pickle.

    Parameters
    ----------
    filepath : str
        Caminho do arquivo pickle.

    Returns
    -------
    Any
        Objeto Python carregado do arquivo.

    Raises
    ------
    FileNotFoundError
        Se o arquivo não existir.
    IOError
        Se houver erro ao ler o arquivo.

    Examples
    --------
    >>> model = load_pickle("models/model.pkl")
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)
