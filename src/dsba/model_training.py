"""
This module is just a convenience to train a simple classifier.
Its presence is a bit artificial for the exercice and not required to develop an MLOps platform.
The MLOps course is not about model training.
"""
import numpy as np
from dataclasses import dataclass
import logging
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import random
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from dsba.model_registry import ClassifierMetadata
from .preprocessing import split_features_and_target, preprocess_dataframe


def train_simple_classifier(
    df: pd.DataFrame, target_column: str, model_id: str
) -> tuple[ClassifierMixin, ClassifierMetadata]:
    logging.info("Start training a simple classifier")
    df = preprocess_dataframe(df)
    X, y = split_features_and_target(df, target_column)
    model = xgb.XGBClassifier(random_state=42)
    model.fit(X, y)

    logging.info("Done training a simple classifier")
    metadata = ClassifierMetadata(
        id=model_id,
        created_at=str(datetime.now()),
        algorithm="xgboost",
        target_column=target_column,
        hyperparameters={"random_state": 42},
        description="",
        performance_metrics={},
    )
    return model, metadata

def train_random_forest_classifier(
    df: pd.DataFrame, target_column: str, model_id: str
) -> tuple[ClassifierMixin, ClassifierMetadata]:
    logging.info(f"Start training a Random Forest classifier with id: {model_id}")
    df_processed = preprocess_dataframe(df.copy())
    X, y = split_features_and_target(df_processed, target_column)
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    logging.info(f"Done training Random Forest classifier: {model_id}")
    
    metadata = ClassifierMetadata(
        id=model_id,
        created_at=str(datetime.now()),
        algorithm="random_forest",
        target_column=target_column,
        hyperparameters={"random_state": 42},
        description="A Random Forest classifier",
        performance_metrics={}, 
    )
    return model, metadata

# Simple models for comparaisons  : Random, NaivePositive and Naive Negative

class RandomClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.classes_ = [0, 1]
        return self

    def predict(self, X):
        return np.random.randint(0, 2, size=len(X))

class NaivePositiveClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.classes_ = [0, 1]
        return self

    def predict(self, X):
        return np.ones(len(X), dtype=int)

class NaiveNegativeClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y=None):
        self.classes_ = [0, 1]
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

# Training functions 
def train_random_classifier(
    df: pd.DataFrame, target_column: str, model_id: str
) -> tuple[ClassifierMixin, ClassifierMetadata]:
    logging.info("Start 'training' a random classifier")
    df_processed = df.drop(columns=[target_column])
    df_processed = df.copy() 

    model = RandomClassifier()

    logging.info("Done 'training' a random classifier")
    metadata = ClassifierMetadata(
        id=model_id,
        created_at=str(datetime.now()),
        algorithm="random_classifier",
        target_column=target_column,
        hyperparameters={}, 
        description="randomly predict 0 or 1",
        performance_metrics={}, 
    )
    return model, metadata

def train_naive_positive_classifier(
    df: pd.DataFrame, target_column: str, model_id: str
) -> tuple[ClassifierMixin, ClassifierMetadata]:
    """Trains' the naive positive classifier."""
    logging.info("Start 'training' a naive positive classifier")
    df_processed = df.drop(columns=[target_column])
    df_processed = preprocess_dataframe(df_processed)

    model = NaivePositiveClassifier()

    logging.info("Done 'training' a naive positive classifier")
    metadata = ClassifierMetadata(
        id=model_id,
        created_at=str(datetime.now()),
        algorithm="naive_positive",
        target_column=target_column,
        hyperparameters={},
        description="all time predict 1",
        performance_metrics={},
    )
    return model, metadata

def train_naive_negative_classifier(
    df: pd.DataFrame, target_column: str, model_id: str
) -> tuple[ClassifierMixin, ClassifierMetadata]:
    logging.info("Start 'training' a naive negative classifier")
    df_processed = df.drop(columns=[target_column])
    df_processed = preprocess_dataframe(df_processed) 

    model = NaiveNegativeClassifier()

    logging.info("Done 'training' a naive negative classifier")
    metadata = ClassifierMetadata(
        id=model_id,
        created_at=str(datetime.now()),
        algorithm="naive_negative",
        target_column=target_column,
        hyperparameters={},
        description="all time predict 0",
        performance_metrics={},
    )
    return model, metadata




