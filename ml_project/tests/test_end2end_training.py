import os
from typing import List
import pytest

from py._path.local import LocalPath

from ml_project.train_pipeline import train_pipeline
from ml_project.entities import (
    TrainingPipelineParams,
    SplittingParams,
    FeatureParams,
    LogisticRegressionParams,
    RandomForestParams,
    KerasSimpleNNParams,
)

from ml_project.entities.keras_model_params import (
    FirstLayerParams,
    SecondLayerParams,
    OutputLayerParams,
    TrainParams,
    CompileParams,
)

from ml_project.constants.constants import RANDOM_STATE


LINEAR_REGRESSION = "lr"
RANDOM_FOREST = "rf"
NEURAL_NETWORK = "nn"


@pytest.mark.parametrize(
    "model",
    [
        RANDOM_FOREST,
        # NEURAL_NETWORK,
        LINEAR_REGRESSION,
    ],
)
def test_train_e2e(
    tmpdir: LocalPath,
    dataset_path: str,
    transformer_train_path: str,
    categorical_features: List[str],
    numerical_features: List[str],
    target_col: str,
    features_to_drop: List[str],
    model: str,
):
    expected_output_model_path = tmpdir.join("model.pkl")
    expected_output_model_path_nn = tmpdir.join("model.h5")
    expected_metric_path = tmpdir.join("metrics.json")

    if model == "nn":
        first_layer_params = FirstLayerParams()
        second_layer_params = SecondLayerParams()
        output_layer_params = OutputLayerParams()
        train_params = TrainParams()
        compile_params = CompileParams()

        model_params = KerasSimpleNNParams(
            model_type="KerasSimpleNN",
            first_layer=first_layer_params,
            second_layer=second_layer_params,
            output_layer=output_layer_params,
            train=train_params,
            compile=compile_params,
        )
    elif model == "rf":
        model_params = RandomForestParams(model_type="RandomForestClassifier")
    elif model == "lr":
        model_params = LogisticRegressionParams(model_type="LogisticRegression")
    else:
        assert True

    params = TrainingPipelineParams(
        input_data_path=dataset_path,
        output_model_path=expected_output_model_path,
        output_model_path_nn=expected_output_model_path_nn,
        transformer_path=transformer_train_path,
        metric_path=expected_metric_path,
        splitting_params=SplittingParams(val_size=0.2, random_state=RANDOM_STATE),
        feature_params=FeatureParams(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            target_col=target_col,
            features_to_drop=features_to_drop,
        ),
        model=model_params,
    )

    real_model_path, metrics = train_pipeline(params)

    assert os.path.exists(real_model_path)
    if model == "nn":
        assert real_model_path == expected_output_model_path_nn
    else:
        assert real_model_path == expected_output_model_path

    assert os.path.exists(expected_metric_path)
    assert metrics["roc_auc_score"] >= 0
    assert metrics["accuracy_score"] >= 0
    assert metrics["f1_score"] >= 0
