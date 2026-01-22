"""
Testes Unitários para DataIngestion

Demonstra práticas de testes automatizados (Aula 6).
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Adiciona o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_ingestion import DataIngestion


class TestDataIngestion(unittest.TestCase):
    """Testes para a classe DataIngestion."""

    def setUp(self):
        """Configuração executada antes de cada teste."""
        self.test_data_path = "data/test_cars.csv"
        self.ingestion = DataIngestion(self.test_data_path)

    def test_init(self):
        """Testa inicialização da classe."""
        self.assertEqual(self.ingestion.data_path, self.test_data_path)
        self.assertIsNone(self.ingestion.data)

    def test_generate_synthetic_data(self):
        """Testa geração de dados sintéticos."""
        n_samples = 100
        df = self.ingestion.generate_synthetic_data(n_samples=n_samples)

        # Verifica se gerou o número correto de amostras
        self.assertEqual(len(df), n_samples)

        # Verifica se todas as colunas esperadas estão presentes
        expected_columns = ['year', 'mileage', 'engine_size', 'horsepower', 'num_doors', 'price']
        self.assertListEqual(df.columns.tolist(), expected_columns)

        # Verifica se não há valores nulos
        self.assertEqual(df.isnull().sum().sum(), 0)

        # Verifica ranges dos valores
        self.assertTrue(df['year'].between(2010, 2023).all())
        self.assertTrue(df['price'].min() >= 1000)

    def test_split_data(self):
        """Testa divisão dos dados."""
        # Gera dados primeiro
        self.ingestion.generate_synthetic_data(n_samples=100)

        # Divide os dados
        X_train, X_test, y_train, y_test = self.ingestion.split_data(test_size=0.2)

        # Verifica tamanhos
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_test), 20)

        # Verifica se não há sobreposição
        train_index = set(X_train.index)
        test_index = set(X_test.index)
        self.assertEqual(len(train_index.intersection(test_index)), 0)

    def test_split_data_without_loading(self):
        """Testa erro ao dividir dados sem carregar."""
        with self.assertRaises(ValueError):
            self.ingestion.split_data()


if __name__ == '__main__':
    unittest.main()
