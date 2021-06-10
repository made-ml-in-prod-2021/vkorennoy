import logging
import sys
import os
import json

from omegaconf import DictConfig
import hydra

from ml_project.entities import (
    TrainingPipelineParamsSchema,
    TrainingPipelineParams,
)
from ml_project.features import make_features, extract_target, build_transformer, serialize_transformer
from ml_project.models import (
    SklearnModel,
    SimpleNNModel,
    read_data,
    split_train_val_data,
)


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_pipeline(training_pipeline_params: TrainingPipelineParams):
    logger.info(f"start train pipeline with params {training_pipeline_params}")
    data = read_data(training_pipeline_params.input_data_path)
    data = data.drop(training_pipeline_params.feature_params.features_to_drop, axis=1)

    logger.info(f"data.shape is {data.shape}")
    train_df, val_df = split_train_val_data(
        data, training_pipeline_params.splitting_params
    )

    val_target = extract_target(val_df, training_pipeline_params.feature_params)
    train_target = extract_target(train_df, training_pipeline_params.feature_params)
    train_df = train_df.drop(training_pipeline_params.feature_params.target_col, 1)
    val_df = val_df.drop(training_pipeline_params.feature_params.target_col, 1)

    logger.info(f"train_df.shape is {train_df.shape}")
    logger.info(f"val_df.shape is {val_df.shape}")
    transformer = build_transformer(training_pipeline_params.feature_params)
    transformer.fit(train_df)
    serialize_transformer(transformer, training_pipeline_params.transformer_path)

    train_features = make_features(transformer, train_df)
    val_features = make_features(transformer, val_df)

    logger.info(f"train_features.shape is {train_features.shape}")
    if training_pipeline_params.model.model_type == "KerasSimpleNN":
        model = SimpleNNModel(training_pipeline_params.model, len(train_features.columns))
    else:
        model = SklearnModel(training_pipeline_params.model)

    model.train_model(
        train_features, train_target
    )

    logger.info(f"val_features.shape is {val_features.shape}")
    predicts = model.predict_model(
        val_features,
    )
    metrics = model.evaluate_model(
        predicts,
        val_target,
    )
    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"metrics is {metrics}")
    path_to_model = model.serialize_model(training_pipeline_params)

    return path_to_model, metrics


@hydra.main(config_path="configs", config_name="train_config")
def train_pipeline_command(cfg: DictConfig):
    os.chdir(hydra.utils.to_absolute_path("."))
    schema = TrainingPipelineParamsSchema()
    params = schema.load(cfg)
    train_pipeline(params)


if __name__ == "__main__":
    train_pipeline_command()
