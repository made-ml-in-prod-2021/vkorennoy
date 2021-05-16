import logging
import sys
import os
import pandas as pd

from omegaconf import DictConfig
import hydra

from ml_project.entities import (
    EvalPipelineParams,
    EvalPipelineParamsSchema,
)
from ml_project.features import make_features, deserialize_transformer
from ml_project.models import (
    SklearnModel,
)


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def eval_pipeline(eval_pipeline_params: EvalPipelineParams):
    logger.info(f"start eval pipeline with params {eval_pipeline_params}")
    data = pd.read_csv(eval_pipeline_params.input_data_path)
    data = data.drop(eval_pipeline_params.feature_params.features_to_drop, axis=1)

    logger.info(f"data.shape is {data.shape}")

    transformer = deserialize_transformer(eval_pipeline_params.transformer_path)
    features = make_features(transformer, data)
    logger.info(f"data_features.shape is {features.shape}")

    model = SklearnModel.deserialize_model(eval_pipeline_params.model_path)
    predicts = model.predict(
        features,
    )

    pd.DataFrame(predicts).to_csv(eval_pipeline_params.predictons_path, header=False)
    logger.info(f"predictions could be found at {eval_pipeline_params.predictons_path}")
    return predicts


@hydra.main(config_path="configs", config_name="eval_config")
def eval_pipeline_command(cfg: DictConfig):
    os.chdir(hydra.utils.to_absolute_path("."))
    schema = EvalPipelineParamsSchema()
    params = schema.load(cfg)
    eval_pipeline(params)


if __name__ == "__main__":
    eval_pipeline_command()
