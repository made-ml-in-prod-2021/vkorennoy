import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

import pickle

from ml_project.entities.feature_params import FeatureParams


def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values='?', strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown='ignore')),
        ]
    )
    return categorical_pipeline


def build_numerical_pipeline() -> Pipeline:
    num_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values='?', strategy="most_frequent")),
            ("scale", StandardScaler()),
        ]
    )
    return num_pipeline


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                "categorical_pipeline",
                build_categorical_pipeline(),
                params.categorical_features,
            ),
            (
                "numerical_pipeline",
                build_numerical_pipeline(),
                params.numerical_features,
            ),
        ]
    )
    return transformer


def serialize_transformer(transformer: ColumnTransformer, output_file: str):
    with open(output_file, "wb") as f:
        pickle.dump(transformer, f)


def deserialize_transformer(transformer_file: str) -> ColumnTransformer:
    with open(transformer_file, "rb") as f:
        transformer = pickle.load(f)
    return transformer


def make_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(transformer.transform(df))


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    target = df[params.target_col].values
    return target
