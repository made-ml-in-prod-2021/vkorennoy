import os

import click
from sklearn.model_selection import train_test_split
from numpy import genfromtxt
import numpy as np


@click.command("split")
@click.option("--input-dir")
@click.option("--output-dir")
def split(input_dir: str, output_dir: str):
    data = genfromtxt(os.path.join(input_dir, "data.csv"), delimiter=',')
    target = genfromtxt(os.path.join(input_dir, "target.csv"), delimiter=',')
    data_target = np.hstack((data, target[..., np.newaxis]))
    train, val = train_test_split(data_target, test_size=0.3)

    data_train = train[:, :-1]
    target_train = train[:, -1]

    data_val = train[:, :-1]
    target_val = train[:, -1]

    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(os.path.join(output_dir, "data_train.csv"), data_train, delimiter=",")
    np.savetxt(os.path.join(output_dir, "target_train.csv"), target_train, delimiter=",")

    np.savetxt(os.path.join(output_dir, "data_val.csv"), data_val, delimiter=",")
    np.savetxt(os.path.join(output_dir, "target_val.csv"), target_val, delimiter=",")


if __name__ == "__main__":
    split()
