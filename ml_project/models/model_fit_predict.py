import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from ml_project.enities import TrainingPipelineParams, LogisticRegressionParams, RandomForestParams

SklearnClassifierModel = Union[RandomForestClassifier, LogisticRegression]
ModelParams = Union[LogisticRegressionParams, RandomForestParams]


class SklearnModel:
    def __init__(self, params: ModelParams):
        self.params = params
        self.model = self.create_model(params)

    @staticmethod
    def create_model(
        train_params: ModelParams,
    ) -> SklearnClassifierModel:
        if train_params.model_type == "RandomForestClassifier":
            model = RandomForestClassifier(
                n_estimators=train_params.n_estimators,
                random_state=train_params.random_state,
            )
        elif train_params.model_type == "LogisticRegression":
            model = LogisticRegression(
                penalty=train_params.penalty,
                C=train_params.C,
                random_state=train_params.random_state,
            )
        else:
            raise NotImplementedError()
        return model

    def train_model(self, features: pd.DataFrame, target: pd.Series):
        self.model.fit(features.values, target)

    def predict_model(
        self, features: pd.DataFrame
    ) -> np.ndarray:
        predicts = self.model.predict(features)
        return predicts

    @staticmethod
    def evaluate_model(
        predicts: np.ndarray, target: pd.Series
    ) -> Dict[str, float]:
        return {
            "roc_auc_score": roc_auc_score(target, predicts),
            "accuracy_score": accuracy_score(target, predicts),
            "f1_score": f1_score(target, predicts),
        }

    def serialize_model(self, params: TrainingPipelineParams) -> str:
        with open(params.output_model_path, "wb") as f:
            pickle.dump(self.model, f)
        return params.output_model_path

    @staticmethod
    def deserialize_model(path: str) -> SklearnClassifierModel:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
