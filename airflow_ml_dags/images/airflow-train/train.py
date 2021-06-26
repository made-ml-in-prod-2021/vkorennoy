import os

from numpy import genfromtxt
import click
import pickle
from sklearn.linear_model import LogisticRegression


@click.command("train")
@click.option("--input-dir")
@click.option("--model-path")
@click.option("--model-name")
def train(input_dir: str, model_path: str, model_name: str):
    model = LogisticRegression()

    data = genfromtxt(os.path.join(input_dir, "data_train.csv"), delimiter=',')
    target = genfromtxt(os.path.join(input_dir, "target_train.csv"), delimiter=',')
    model.fit(data, target)

    with open(os.path.join(model_path, model_name), 'wb') as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    train()
