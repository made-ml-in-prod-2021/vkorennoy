from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from typing import Union
import yaml

from .split_params import SplittingParams
from .feature_params import FeatureParams
from .train_params import LogisticRegressionParams, RandomForestParams
from .keras_model_params import KerasSimpleNNParams


@dataclass()
class TrainingPipelineParams:
    input_data_path: str
    output_model_path: str
    output_model_path_nn: str
    transformer_path: str
    metric_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    model: Union[LogisticRegressionParams, RandomForestParams, KerasSimpleNNParams]


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
