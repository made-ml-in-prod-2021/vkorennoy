import numpy as np
import pandas as pd
import pickle

from typing import Dict

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from ml_project.enities import KerasSimpleNNParams, TrainingPipelineParams


class SimpleNNModel:
    def __init__(self, params: KerasSimpleNNParams, num_features: int):
        self.params = params
        self.model = self.create_model(params, num_features)

    @staticmethod
    def create_model(params: KerasSimpleNNParams, num_features) -> Sequential:
        np.random.seed(params.random_state)
        model = Sequential()

        # Input layer
        model.add(Dense(units=params.first_layer.units,
                        input_dim=num_features,
                        kernel_initializer=params.first_layer.kernel_initializer,
                        activation=params.first_layer.activation))
        model.add(Dropout(params.dropout))

        # Hidden layer 1
        model.add(Dense(units=params.second_layer.units,
                        kernel_initializer=params.second_layer.kernel_initializer,
                        activation=params.second_layer.activation))
        model.add(Dropout(params.dropout))

        # Output layer
        model.add(Dense(units=params.output_layer.units,
                        kernel_initializer=params.output_layer.kernel_initializer,
                        activation=params.output_layer.activation))

        model.compile(loss=params.compile.loss,
                      optimizer=params.compile.optimizer,
                      metrics=[params.compile.metrics],
                      )

        return model

    def train_model(self, values: pd.DataFrame, target: pd.Series):
        self.model.fit(
            x=np.asarray(values).astype('float32'), y=target,
            epochs=self.params.train.epochs,
            batch_size=self.params.train.batch_size,
            verbose=self.params.train.verbose,
        )

    def predict_model(self, features: pd.DataFrame) -> np.ndarray:
        predict_proba = self.model.predict(np.asarray(features).astype('float32'))
        predicts = np.argmax(predict_proba, axis=1)
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
        self.model.save(
            params.output_model_path_nn, overwrite=True,
        )
        return params.output_model_path_nn

    @staticmethod
    def deserialize_model(path: str) -> Sequential:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
