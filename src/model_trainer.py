"""
Módulo de Treinamento de Modelos

Treina modelos de Machine Learning com API consistente e intuitiva.
Implementa padrões similares ao scikit-learn (Aula 7).
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from .utils import setup_logger, save_pickle

logger = setup_logger(__name__)


class ModelTrainer:
    """
    Classe para treinamento de modelos de regressão.

    Implementa interface consistente com métodos fit/predict
    seguindo padrões do scikit-learn (Aula 7).

    Attributes
    ----------
    model : object
        Modelo de ML (default: LinearRegression).
    scaler : StandardScaler
        Scaler para normalização de features.
    is_fitted : bool
        Indica se o modelo foi treinado.

    Examples
    --------
    >>> trainer = ModelTrainer()
    >>> trainer.fit(X_train, y_train)
    >>> predictions = trainer.predict(X_test)
    """

    def __init__(self, model: Optional[Any] = None, use_scaling: bool = True):
        """
        Inicializa o treinador de modelos.

        Parameters
        ----------
        model : object, optional
            Modelo de ML a ser usado. Se None, usa LinearRegression.
        use_scaling : bool, optional
            Se True, aplica StandardScaler (default: True).
        """
        self.model = model if model is not None else LinearRegression()
        self.use_scaling = use_scaling
        self.scaler = StandardScaler() if use_scaling else None
        self.is_fitted = False
        logger.info(
            f"ModelTrainer inicializado com modelo: {type(self.model).__name__}"
        )

    def fit(
        self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs
    ) -> "ModelTrainer":
        """
        Treina o modelo com os dados fornecidos.

        Parameters
        ----------
        X_train : pd.DataFrame
            Features de treino.
        y_train : pd.Series
            Target de treino.
        **kwargs
            Parâmetros adicionais para o método fit do modelo.

        Returns
        -------
        self : ModelTrainer
            Retorna a própria instância (padrão scikit-learn).

        Examples
        --------
        >>> trainer.fit(X_train, y_train)
        """
        logger.info("Iniciando treinamento do modelo")

        # Aplica scaling se configurado
        if self.use_scaling and self.scaler is not None:
            X_scaled = self.scaler.fit_transform(X_train)
            logger.info("Features normalizadas com StandardScaler")
        else:
            X_scaled = X_train

        # Treina o modelo
        self.model.fit(X_scaled, y_train, **kwargs)
        self.is_fitted = True

        logger.info("Treinamento concluído com sucesso")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Realiza predições com o modelo treinado.

        Parameters
        ----------
        X : pd.DataFrame
            Features para predição.

        Returns
        -------
        np.ndarray
            Predições do modelo.

        Raises
        ------
        ValueError
            Se o modelo não foi treinado.

        Examples
        --------
        >>> predictions = trainer.predict(X_test)
        """
        if not self.is_fitted:
            raise ValueError("Modelo não treinado. Execute fit() antes de predict().")

        # Aplica scaling se configurado
        if self.use_scaling and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        predictions = self.model.predict(X_scaled)
        logger.info(f"Predições realizadas para {len(predictions)} amostras")

        return predictions

    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Retorna a importância das features (para modelos lineares).

        Returns
        -------
        pd.Series or None
            Série com coeficientes do modelo, se disponível.
        """
        if not self.is_fitted:
            logger.warning("Modelo não treinado")
            return None

        if hasattr(self.model, "coef_"):
            return pd.Series(self.model.coef_, name="importance")
        else:
            logger.warning("Modelo não possui atributo 'coef_'")
            return None

    def save(self, model_path: str, scaler_path: Optional[str] = None) -> None:
        """
        Salva o modelo treinado em disco.

        Parameters
        ----------
        model_path : str
            Caminho para salvar o modelo.
        scaler_path : str, optional
            Caminho para salvar o scaler.

        Examples
        --------
        >>> trainer.save("models/model.pkl", "models/scaler.pkl")
        """
        if not self.is_fitted:
            logger.warning("Salvando modelo não treinado")

        save_pickle(self.model, model_path)
        logger.info(f"Modelo salvo em: {model_path}")

        if self.scaler is not None and scaler_path is not None:
            save_pickle(self.scaler, scaler_path)
            logger.info(f"Scaler salvo em: {scaler_path}")

    def get_params(self) -> Dict[str, Any]:
        """
        Retorna os parâmetros do modelo.

        Returns
        -------
        dict
            Parâmetros do modelo.
        """
        if hasattr(self.model, "get_params"):
            return self.model.get_params()
        else:
            return {}
