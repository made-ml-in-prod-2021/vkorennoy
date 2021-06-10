import os
import numpy as np

import click
from sklearn.datasets import make_classification


@click.command("generate")
@click.option("--output-dir")
def generate(output_dir: str):
    data, target = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=0)

    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(os.path.join(output_dir, "data.csv"), data, delimiter=",")
    np.savetxt(os.path.join(output_dir, "target.csv"), target, delimiter=",")


if __name__ == '__main__':
    generate()
