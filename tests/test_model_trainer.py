"""
Testes Unitários para ModelTrainer

Demonstra testes para treinamento de modelos (Aula 6).
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from src.model_trainer import ModelTrainer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestModelTrainer(unittest.TestCase):
    """Testes para a classe ModelTrainer."""

    def setUp(self):
        """Configuração executada antes de cada teste."""
        # Cria dados de treino sintéticos
        np.random.seed(42)
        self.X_train = pd.DataFrame(
            {"feature1": np.random.randn(100), "feature2": np.random.randn(100)}
        )
        self.y_train = pd.Series(
            2 * self.X_train["feature1"]
            + 3 * self.X_train["feature2"]
            + np.random.randn(100) * 0.1
        )

        self.X_test = pd.DataFrame(
            {"feature1": np.random.randn(20), "feature2": np.random.randn(20)}
        )

        self.trainer = ModelTrainer()

    def test_init_default(self):
        """Testa inicialização com parâmetros padrão."""
        self.assertIsNotNone(self.trainer.model)
        self.assertIsNotNone(self.trainer.scaler)
        self.assertFalse(self.trainer.is_fitted)

    def test_init_custom_model(self):
        """Testa inicialização com modelo customizado."""
        custom_model = Ridge(alpha=1.0)
        trainer = ModelTrainer(model=custom_model)

        self.assertEqual(type(trainer.model).__name__, "Ridge")

    def test_fit(self):
        """Testa treinamento do modelo."""
        trainer = self.trainer.fit(self.X_train, self.y_train)

        # Verifica se retorna self
        self.assertEqual(trainer, self.trainer)

        # Verifica se modelo foi marcado como fitted
        self.assertTrue(self.trainer.is_fitted)

    def test_predict_before_fit(self):
        """Testa erro ao predizer sem treinar."""
        with self.assertRaises(ValueError):
            self.trainer.predict(self.X_test)

    def test_predict_after_fit(self):
        """Testa predições após treinamento."""
        self.trainer.fit(self.X_train, self.y_train)
        predictions = self.trainer.predict(self.X_test)

        # Verifica shape das predições
        self.assertEqual(len(predictions), len(self.X_test))

        # Verifica se são números válidos
        self.assertFalse(np.isnan(predictions).any())

    def test_get_feature_importance(self):
        """Testa obtenção de importância das features."""
        self.trainer.fit(self.X_train, self.y_train)
        importance = self.trainer.get_feature_importance()

        # Para modelos lineares, deve retornar coeficientes
        self.assertIsNotNone(importance)
        self.assertEqual(len(importance), 2)  # 2 features

    def test_training_without_scaling(self):
        """Testa treinamento sem normalização."""
        trainer = ModelTrainer(use_scaling=False)
        trainer.fit(self.X_train, self.y_train)

        self.assertIsNone(trainer.scaler)
        self.assertTrue(trainer.is_fitted)


if __name__ == "__main__":
    unittest.main()
