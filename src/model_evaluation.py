"""
Módulo de Avaliação de Modelos

Avalia performance de modelos de ML com diversas métricas.
Fornece relatórios detalhados e visualizações (Aula 3).
"""

import pandas as pd
import numpy as np
from typing import Dict
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from .utils import setup_logger


logger = setup_logger(__name__)


class ModelEvaluator:
    """
    Classe para avaliação de modelos de regressão.

    Calcula métricas de performance e gera relatórios detalhados.

    Attributes
    ----------
    y_true : np.ndarray
        Valores reais do target.
    y_pred : np.ndarray
        Valores preditos pelo modelo.
    metrics : dict
        Dicionário com as métricas calculadas.

    Examples
    --------
    >>> evaluator = ModelEvaluator(y_test, predictions)
    >>> metrics = evaluator.calculate_metrics()
    >>> print(evaluator.get_report())
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ):
        """
        Inicializa o avaliador de modelos.

        Parameters
        ----------
        y_true : np.ndarray
            Valores reais do target.
        y_pred : np.ndarray
            Valores preditos pelo modelo.
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.metrics: Dict = {}
        logger.info("ModelEvaluator inicializado")

    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calcula métricas de avaliação do modelo.

        Returns
        -------
        dict
            Dicionário com métricas calculadas:
            - rmse: Root Mean Squared Error
            - mae: Mean Absolute Error
            - r2: R² Score
            - mape: Mean Absolute Percentage Error

        Examples
        --------
        >>> metrics = evaluator.calculate_metrics()
        >>> print(f"R² Score: {metrics['r2']:.3f}")
        """
        # RMSE (Root Mean Squared Error)
        rmse = np.sqrt(mean_squared_error(self.y_true, self.y_pred))

        # MAE (Mean Absolute Error)
        mae = mean_absolute_error(self.y_true, self.y_pred)

        # R² Score
        r2 = r2_score(self.y_true, self.y_pred)

        # MAPE (Mean Absolute Percentage Error)
        mape = (np.mean(
            np.abs((self.y_true - self.y_pred) / self.y_true))
            * 100)

        self.metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }

        logger.info("Métricas calculadas com sucesso")
        return self.metrics

    def get_residuals(self) -> np.ndarray:
        """
        Calcula os resíduos (erros) do modelo.

        Returns
        -------
        np.ndarray
            Array com os resíduos (y_true - y_pred).

        Examples
        --------
        >>> residuals = evaluator.get_residuals()
        >>> print(f"Resíduo médio: {np.mean(residuals):.3f}")
        """
        residuals = self.y_true - self.y_pred
        return residuals

    def calculate_prediction_intervals(
        self,
        confidence: float = 0.95
    ) -> Dict[str, np.ndarray]:
        """
        Calcula intervalos de confiança para as predições.

        Parameters
        ----------
        confidence : float, optional
            Nível de confiança (default: 0.95).

        Returns
        -------
        dict
            Dicionário com 'lower_bound' e 'upper_bound'.
        """
        residuals = self.get_residuals()
        std_residuals = np.std(residuals)

        # Z-score para intervalo de confiança
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence) / 2)

        margin = z_score * std_residuals

        return {
            'lower_bound': self.y_pred - margin,
            'upper_bound': self.y_pred + margin
        }

    def get_report(self) -> str:
        """
        Gera relatório textual com as métricas.

        Returns
        -------
        str
            Relatório formatado com todas as métricas.

        Examples
        --------
        >>> print(evaluator.get_report())
        """
        if not self.metrics:
            self.calculate_metrics()

        report = []
        report.append("=" * 60)
        report.append("RELATÓRIO DE AVALIAÇÃO DO MODELO")
        report.append("=" * 60)
        report.append(f"\nNúmero de amostras: {len(self.y_true)}")
        report.append("\nMÉTRICAS DE PERFORMANCE:")
        report.append("-" * 60)
        report.append(f"  RMSE (Root Mean Squared Error): {self.metrics['rmse']:.2f}")
        report.append(f"  MAE (Mean Absolute Error):      {self.metrics['mae']:.2f}")
        report.append(f"  R² Score:                        {self.metrics['r2']:.4f}")
        report.append(f"  MAPE (%):                        {self.metrics['mape']:.2f}%")
        report.append("-" * 60)

        # Estatísticas dos resíduos
        residuals = self.get_residuals()
        report.append("\nESTATÍSTICAS DOS RESÍDUOS:")
        report.append("-" * 60)
        report.append(f"  Média:          {np.mean(residuals):.2f}")
        report.append(f"  Desvio Padrão:  {np.std(residuals):.2f}")
        report.append(f"  Mínimo:         {np.min(residuals):.2f}")
        report.append(f"  Máximo:         {np.max(residuals):.2f}")
        report.append("=" * 60)

        return "\n".join(report)

    def compare_predictions(self, n_samples: int = 10) -> pd.DataFrame:
        """
        Compara valores reais vs preditos em uma amostra.

        Parameters
        ----------
        n_samples : int, optional
            Número de amostras a mostrar (default: 10).

        Returns
        -------
        pd.DataFrame
            DataFrame comparando valores reais e preditos.

        Examples
        --------
        >>> comparison = evaluator.compare_predictions(5)
        >>> print(comparison)
        """
        n_samples = min(n_samples, len(self.y_true))

        comparison_df = pd.DataFrame({
            'Real': self.y_true[:n_samples],
            'Predito': self.y_pred[:n_samples],
            'Erro': self.y_true[:n_samples] - self.y_pred[:n_samples],
            'Erro_%': np.abs(
                (self.y_true[:n_samples] - self.y_pred[:n_samples])
                / self.y_true[:n_samples] * 100
            )
        })

        return comparison_df
