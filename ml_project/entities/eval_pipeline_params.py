from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from .feature_params import FeatureParams


@dataclass()
class EvalPipelineParams:
    input_data_path: str
    predictons_path: str
    transformer_path: str
    model_path: str
    feature_params: FeatureParams


EvalPipelineParamsSchema = class_schema(EvalPipelineParams)
