"""
Testes Unitários para DataValidator

Demonstra testes para validação de qualidade de dados (Aula 6).
"""

import unittest
import sys
import os
import pandas as pd
from src.data_validation import DataValidator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestDataValidator(unittest.TestCase):
    """Testes para a classe DataValidator."""

    def setUp(self):
        """Configuração executada antes de cada teste."""
        # Cria DataFrame de teste
        self.df = pd.DataFrame({
            'year': [2020, 2021, 2022, 2020, 2021],
            'price': [10000, 15000, 20000, 12000, None],
            'mileage': [50000, 30000, 10000, 45000, 35000]
        })
        self.validator = DataValidator(self.df)

    def test_init(self):
        """Testa inicialização da classe."""
        self.assertIsInstance(self.validator.data, pd.DataFrame)
        self.assertEqual(len(self.validator.data), 5)

    def test_check_missing_values(self):
        """Testa detecção de valores ausentes."""
        missing = self.validator.check_missing_values()

        # Deve encontrar 1 valor ausente na coluna 'price'
        self.assertIn('price', missing)
        self.assertEqual(missing['price'], 1)

    def test_check_duplicates(self):
        """Testa detecção de duplicatas."""
        # Adiciona linha duplicada
        df_with_dup = pd.concat([self.df, self.df.iloc[[0]]], ignore_index=True)
        validator = DataValidator(df_with_dup)

        n_duplicates = validator.check_duplicates()
        self.assertEqual(n_duplicates, 1)

    def test_check_outliers(self):
        """Testa detecção de outliers."""
        # Cria DataFrame com outlier óbvio
        df = pd.DataFrame({
            'value': [10, 12, 11, 13, 1000]  # 1000 é outlier
        })
        validator = DataValidator(df)

        outliers = validator.check_outliers(threshold=2.0)
        self.assertIn('value', outliers)
        self.assertGreater(outliers['value'], 0)

    def test_check_data_types(self):
        """Testa verificação de tipos de dados."""
        expected_types = {
            'year': 'int',
            'price': 'float',
            'mileage': 'int'
        }

        type_check = self.validator.check_data_types(expected_types)

        self.assertTrue(type_check['year'])
        self.assertTrue(type_check['mileage'])

    def test_validate_all(self):
        """Testa execução de todas as validações."""
        results = self.validator.validate_all()

        # Verifica se todas as chaves esperadas estão presentes
        self.assertIn('missing_values', results)
        self.assertIn('duplicates', results)
        self.assertIn('outliers', results)

    def test_get_summary(self):
        """Testa geração de resumo."""
        self.validator.validate_all()
        summary = self.validator.get_summary()

        self.assertIsInstance(summary, str)
        self.assertIn('RESUMO DA VALIDAÇÃO', summary)


if __name__ == '__main__':
    unittest.main()
