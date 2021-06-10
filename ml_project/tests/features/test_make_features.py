from typing import List

import pandas as pd
import pytest
from numpy.testing import assert_allclose

from ml_project.models import read_data
from ml_project.entities import FeatureParams
from ml_project.features import make_features, extract_target, build_transformer


@pytest.fixture
def feature_params(
    categorical_features: List[str],
    features_to_drop: List[str],
    numerical_features: List[str],
    target_col: str,
) -> FeatureParams:
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        features_to_drop=features_to_drop,
        target_col=target_col,
    )
    return params


def test_make_features(
    feature_params: FeatureParams, dataset_path: str,
):
    data = read_data(dataset_path)
    data = data.drop(feature_params.features_to_drop, axis=1)
    transformer = build_transformer(feature_params)
    transformer.fit(data)
    features = make_features(transformer, data)

    assert not pd.isnull(features).any().any()
    assert all(x not in features.columns for x in feature_params.features_to_drop)
    assert features.shape[1] > 36


def test_extract_target(feature_params: FeatureParams, dataset_path: str):
    data = read_data(dataset_path)

    target = extract_target(data, feature_params)
    assert_allclose(
        data[feature_params.target_col], target
    )
