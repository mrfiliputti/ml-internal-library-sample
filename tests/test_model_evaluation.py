"""
Testes Unitários para ModelEvaluator

Demonstra testes para avaliação de modelos (Aula 6).
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model_evaluation import ModelEvaluator


class TestModelEvaluator(unittest.TestCase):
    """Testes para a classe ModelEvaluator."""

    def setUp(self):
        """Configuração executada antes de cada teste."""
        # Cria dados de teste
        np.random.seed(42)
        self.y_true = np.array([100, 200, 300, 400, 500])
        self.y_pred = np.array([110, 190, 310, 390, 510])

        self.evaluator = ModelEvaluator(self.y_true, self.y_pred)

    def test_init(self):
        """Testa inicialização da classe."""
        np.testing.assert_array_equal(self.evaluator.y_true, self.y_true)
        np.testing.assert_array_equal(self.evaluator.y_pred, self.y_pred)
        self.assertEqual(len(self.evaluator.metrics), 0)

    def test_calculate_metrics(self):
        """Testa cálculo de métricas."""
        metrics = self.evaluator.calculate_metrics()

        # Verifica se todas as métricas foram calculadas
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)
        self.assertIn('mape', metrics)

        # Verifica se valores são razoáveis
        self.assertGreater(metrics['r2'], 0.9)  # Deve ter R² alto
        self.assertGreater(metrics['rmse'], 0)
        self.assertGreater(metrics['mae'], 0)

    def test_get_residuals(self):
        """Testa cálculo de resíduos."""
        residuals = self.evaluator.get_residuals()

        # Verifica shape
        self.assertEqual(len(residuals), len(self.y_true))

        # Verifica valores
        expected_residuals = self.y_true - self.y_pred
        np.testing.assert_array_equal(residuals, expected_residuals)

    def test_calculate_prediction_intervals(self):
        """Testa cálculo de intervalos de confiança."""
        intervals = self.evaluator.calculate_prediction_intervals(confidence=0.95)

        # Verifica se retornou limites superior e inferior
        self.assertIn('lower_bound', intervals)
        self.assertIn('upper_bound', intervals)

        # Verifica se limite inferior < superior
        self.assertTrue(
            np.all(intervals['lower_bound'] < intervals['upper_bound'])
        )

    def test_get_report(self):
        """Testa geração de relatório."""
        report = self.evaluator.get_report()

        # Verifica se é string
        self.assertIsInstance(report, str)

        # Verifica se contém informações esperadas
        self.assertIn('RELATÓRIO', report)
        self.assertIn('RMSE', report)
        self.assertIn('MAE', report)
        self.assertIn('R²', report)

    def test_compare_predictions(self):
        """Testa comparação de predições."""
        comparison = self.evaluator.compare_predictions(n_samples=3)

        # Verifica se retornou DataFrame
        self.assertIsInstance(comparison, pd.DataFrame)

        # Verifica colunas
        expected_cols = ['Real', 'Predito', 'Erro', 'Erro_%']
        self.assertListEqual(comparison.columns.tolist(), expected_cols)

        # Verifica número de linhas
        self.assertEqual(len(comparison), 3)

    def test_perfect_predictions(self):
        """Testa métricas com predições perfeitas."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        evaluator = ModelEvaluator(y_true, y_pred)
        metrics = evaluator.calculate_metrics()

        # R² deve ser 1.0
        self.assertAlmostEqual(metrics['r2'], 1.0, places=5)

        # RMSE e MAE devem ser 0
        self.assertAlmostEqual(metrics['rmse'], 0.0, places=5)
        self.assertAlmostEqual(metrics['mae'], 0.0, places=5)


if __name__ == '__main__':
    unittest.main()
