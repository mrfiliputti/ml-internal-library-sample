"""
Módulo de Validação de Dados

Valida qualidade e integridade dos dados antes do treinamento.
Implementa verificações modulares e reutilizáveis (Aula 2).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from .utils import setup_logger


logger = setup_logger(__name__)


class DataValidator:
    """
    Classe para validação de qualidade de dados.

    Realiza verificações de valores nulos, outliers, tipos de dados
    e outras validações essenciais para garantir qualidade dos dados.

    Attributes
    ----------
    data : pd.DataFrame
        DataFrame a ser validado.
    validation_results : dict
        Resultados das validações executadas.

    Examples
    --------
    >>> validator = DataValidator(df)
    >>> validator.validate_all()
    >>> print(validator.validation_results)
    """

    def __init__(self, data: pd.DataFrame):
        """
        Inicializa o validador de dados.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame a ser validado.
        """
        self.data = data
        self.validation_results: Dict = {}
        logger.info("DataValidator inicializado")

    def check_missing_values(self) -> Dict[str, int]:
        """
        Verifica valores ausentes em cada coluna.

        Returns
        -------
        dict
            Dicionário com contagem de valores nulos por coluna.
        """
        missing = self.data.isnull().sum()
        missing_dict = missing[missing > 0].to_dict()

        self.validation_results['missing_values'] = missing_dict

        if missing_dict:
            logger.warning(f"Valores ausentes encontrados: {missing_dict}")
        else:
            logger.info("Nenhum valor ausente encontrado")

        return missing_dict

    def check_duplicates(self) -> int:
        """
        Verifica linhas duplicadas no dataset.

        Returns
        -------
        int
            Número de linhas duplicadas.
        """
        n_duplicates = self.data.duplicated().sum()
        self.validation_results['duplicates'] = n_duplicates

        if n_duplicates > 0:
            logger.warning(f"Encontradas {n_duplicates} linhas duplicadas")
        else:
            logger.info("Nenhuma linha duplicada encontrada")

        return n_duplicates

    def check_outliers(
        self,
        columns: Optional[List[str]] = None,
        threshold: float = 3.0
    ) -> Dict[str, int]:
        """
        Detecta outliers usando método z-score.

        Parameters
        ----------
        columns : list of str, optional
            Colunas para verificar outliers. Se None, verifica todas numéricas.
        threshold : float, optional
            Limite do z-score para considerar outlier (default: 3.0).

        Returns
        -------
        dict
            Dicionário com número de outliers por coluna.
        """
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()

        outliers_dict = {}

        for col in columns:
            z_scores = np.abs(
                (self.data[col] - self.data[col].mean()) / self.data[col].std()
            )
            n_outliers = (z_scores > threshold).sum()
            if n_outliers > 0:
                outliers_dict[col] = n_outliers

        self.validation_results['outliers'] = outliers_dict

        if outliers_dict:
            logger.warning(f"Outliers detectados: {outliers_dict}")
        else:
            logger.info("Nenhum outlier detectado")

        return outliers_dict

    def check_data_types(self, expected_types: Dict[str, str]) -> Dict[str, bool]:
        """
        Verifica se os tipos de dados estão corretos.

        Parameters
        ----------
        expected_types : dict
            Dicionário com tipos esperados {coluna: tipo}.

        Returns
        -------
        dict
            Dicionário indicando se cada coluna está com tipo correto.
        """
        type_check = {}

        for col, expected_type in expected_types.items():
            if col in self.data.columns:
                actual_type = str(self.data[col].dtype)
                type_check[col] = expected_type in actual_type
            else:
                type_check[col] = False
                logger.warning(f"Coluna {col} não encontrada no dataset")

        self.validation_results['type_check'] = type_check

        incorrect_types = {k: v for k, v in type_check.items() if not v}
        if incorrect_types:
            logger.warning(f"Tipos incorretos: {incorrect_types}")
        else:
            logger.info("Todos os tipos de dados estão corretos")

        return type_check

    def validate_all(
        self,
        expected_types: Optional[Dict[str, str]] = None
    ) -> Dict:
        """
        Executa todas as validações.

        Parameters
        ----------
        expected_types : dict, optional
            Tipos esperados para verificação.

        Returns
        -------
        dict
            Resultados de todas as validações.
        """
        logger.info("Iniciando validação completa dos dados")

        self.check_missing_values()
        self.check_duplicates()
        self.check_outliers()

        if expected_types:
            self.check_data_types(expected_types)

        logger.info("Validação completa finalizada")
        return self.validation_results

    def get_summary(self) -> str:
        """
        Retorna um resumo das validações.

        Returns
        -------
        str
            Texto com resumo das validações.
        """
        summary = []
        summary.append("=" * 50)
        summary.append("RESUMO DA VALIDAÇÃO DE DADOS")
        summary.append("=" * 50)

        if 'missing_values' in self.validation_results:
            missing = self.validation_results['missing_values']
            summary.append(f"\nValores Ausentes: {len(missing)} colunas afetadas")

        if 'duplicates' in self.validation_results:
            dups = self.validation_results['duplicates']
            summary.append(f"Linhas Duplicadas: {dups}")

        if 'outliers' in self.validation_results:
            outliers = self.validation_results['outliers']
            summary.append(f"Outliers: {len(outliers)} colunas afetadas")

        summary.append("=" * 50)

        return "\n".join(summary)
