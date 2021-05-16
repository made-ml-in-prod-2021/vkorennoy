import pytest
import os
from typing import List

from py._path.local import LocalPath

from ml_project.enities import EvalPipelineParams, FeatureParams
from ml_project.eval_pipeline import eval_pipeline


@pytest.fixture()
def eval_params(
        tmpdir: LocalPath,
        data_test_path: str,
        transformer_path: str,
        predictions_path: str,
        model_path: str,
        categorical_features: List[str],
        numerical_features: List[str],
        target_col: str,
        features_to_drop: List[str],
):
    return EvalPipelineParams(
        input_data_path=data_test_path,
        predictons_path=predictions_path,
        transformer_path=transformer_path,
        model_path=model_path,
        feature_params=FeatureParams(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            target_col=target_col,
            features_to_drop=features_to_drop,
        ),
    )


def test_eval_pipeline(eval_params: EvalPipelineParams):
    predicts = eval_pipeline(eval_params)
    assert predicts.shape[0] == 100
    assert os.path.exists(eval_params.predictons_path)
    assert set(predicts) == {0, 1}
