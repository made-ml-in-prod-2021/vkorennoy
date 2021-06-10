import os
import pickle

import pandas as pd
import click
from sklearn.metrics import roc_auc_score, accuracy_score


@click.command("predict")
@click.option("--data-path")
@click.option("--model-path")
@click.option("--model-name")
@click.option("--metrics-path")
@click.option("--metrics-name")
def predict(data_path: str, model_path: str, model_name: str, metrics_path: str, metrics_name: str):
    data = pd.read_csv(os.path.join(data_path, "data_val.csv"))
    target = pd.read_csv(os.path.join(data_path, "target_val.csv"))

    with open(os.path.join(model_path, model_name), 'rb') as f:
        model = pickle.load(f)

    predictions = model.predict(data)

    roc_auc = roc_auc_score(target, predictions)
    accuracy = accuracy_score(target, predictions)

    os.makedirs(metrics_path, exist_ok=True)

    with open(os.path.join(metrics_path, metrics_name), "w") as f:
        f.write(f"roc_auc: {roc_auc}, accuracy: {accuracy}")


if __name__ == '__main__':
    predict()
