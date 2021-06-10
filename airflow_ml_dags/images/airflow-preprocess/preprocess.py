import os
import pandas as pd
import click
from sklearn.preprocessing import StandardScaler
import numpy as np


@click.command("preprocess")
@click.option("--input-dir")
@click.option("--output-dir")
def preprocess(input_dir: str, output_dir):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target.csv"))

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(os.path.join(output_dir, "data.csv"), scaled_data, delimiter=",")
    np.savetxt(os.path.join(output_dir, "target.csv"), target, delimiter=",")


if __name__ == '__main__':
    preprocess()
