"""
Módulo de Ingestão de Dados

Responsável por carregar e preparar dados de diferentes fontes.
Implementa padrões de design de API consistentes (Aula 7).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from .utils import setup_logger


logger = setup_logger(__name__)


class DataIngestion:
    """
    Classe para ingestão e preparação inicial de dados.

    Esta classe implementa o padrão de design consistente com métodos
    intuitivos para carregar dados de diversas fontes.

    Attributes
    ----------
    data_path : str
        Caminho para o arquivo de dados.
    data : pd.DataFrame, optional
        DataFrame com os dados carregados.

    Examples
    --------
    >>> ingestion = DataIngestion("data/cars.csv")
    >>> df = ingestion.load_data()
    >>> X_train, X_test, y_train, y_test = ingestion.split_data(test_size=0.2)
    """

    def __init__(self, data_path: str):
        """
        Inicializa o objeto DataIngestion.

        Parameters
        ----------
        data_path : str
            Caminho para o arquivo de dados.
        """
        self.data_path = data_path
        self.data: Optional[pd.DataFrame] = None
        logger.info(f"DataIngestion inicializado com path: {data_path}")

    def load_data(self, **kwargs) -> pd.DataFrame:
        """
        Carrega dados de arquivo CSV.

        Parameters
        ----------
        **kwargs
            Argumentos adicionais para pd.read_csv().

        Returns
        -------
        pd.DataFrame
            DataFrame com os dados carregados.

        Raises
        ------
        FileNotFoundError
            Se o arquivo não for encontrado.
        """
        try:
            self.data = pd.read_csv(self.data_path, **kwargs)
            logger.info(
                f"Dados carregados com sucesso: {self.data.shape[0]} linhas, "
                f"{self.data.shape[1]} colunas"
            )
            return self.data
        except FileNotFoundError:
            logger.error(f"Arquivo não encontrado: {self.data_path}")
            raise

    def generate_synthetic_data(
        self,
        n_samples: int = 1000,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Gera dados sintéticos para demonstração.

        Parameters
        ----------
        n_samples : int, optional
            Número de amostras a gerar (default: 1000).
        random_state : int, optional
            Seed para reprodutibilidade (default: 42).

        Returns
        -------
        pd.DataFrame
            DataFrame com dados sintéticos de carros.
        """
        np.random.seed(random_state)

        # Gera features correlacionadas com o preço
        year = np.random.randint(2010, 2024, n_samples)
        mileage = np.random.randint(5000, 200000, n_samples)
        engine_size = np.random.uniform(1.0, 5.0, n_samples)
        horsepower = np.random.randint(70, 400, n_samples)
        num_doors = np.random.choice([2, 4, 5], n_samples)

        # Preço baseado em correlações realistas
        price = (
            5000 +
            (year - 2010) * 2000 +
            (-mileage / 1000 * 50) +
            engine_size * 3000 +
            horsepower * 50 +
            num_doors * 500 +
            np.random.normal(0, 2000, n_samples)
        )
        price = np.maximum(price, 1000)  # Preço mínimo

        self.data = pd.DataFrame({
            'year': year,
            'mileage': mileage,
            'engine_size': engine_size,
            'horsepower': horsepower,
            'num_doors': num_doors,
            'price': price
        })

        logger.info(f"Dados sintéticos gerados: {n_samples} amostras")
        return self.data

    def split_data(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        target_column: str = 'price'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Divide os dados em conjuntos de treino e teste.

        Parameters
        ----------
        test_size : float, optional
            Proporção do conjunto de teste (default: 0.2).
        random_state : int, optional
            Seed para reprodutibilidade (default: 42).
        target_column : str, optional
            Nome da coluna alvo (default: 'price').

        Returns
        -------
        X_train : pd.DataFrame
            Features de treino.
        X_test : pd.DataFrame
            Features de teste.
        y_train : pd.Series
            Target de treino.
        y_test : pd.Series
            Target de teste.

        Raises
        ------
        ValueError
            Se os dados não foram carregados.
        """
        if self.data is None:
            raise ValueError("Dados não carregados. Execute load_data() primeiro.")

        from sklearn.model_selection import train_test_split

        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

        logger.info(
            f"Dados divididos: treino={len(X_train)}, teste={len(X_test)}"
        )

        return X_train, X_test, y_train, y_test
