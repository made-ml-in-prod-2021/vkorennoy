import os

import pytest
from typing import List


@pytest.fixture()
def dataset_path():
    curdir = os.path.dirname(__file__)
    return os.path.join(curdir, "data", "train_data_sample.csv")


@pytest.fixture()
def data_test_path():
    curdir = os.path.dirname(__file__)
    project_root = os.path.dirname(curdir)
    return os.path.join(project_root, "data", "test_data_sample.csv")


@pytest.fixture()
def transformer_train_path():
    curdir = os.path.dirname(__file__)
    project_root = os.path.dirname(curdir)
    return os.path.join(project_root, "models", "transformer.pkl")


@pytest.fixture()
def transformer_eval_path():
    curdir = os.path.dirname(__file__)
    return os.path.join(curdir, "models", "transformer.pkl")


@pytest.fixture()
def model_eval_path():
    curdir = os.path.dirname(__file__)
    return os.path.join(curdir, "models", "model.pkl")


@pytest.fixture()
def predictions_path():
    curdir = os.path.dirname(__file__)
    return os.path.join(curdir, "data", "predictions.csv")



@pytest.fixture()
def target_col():
    return "Biopsy"


@pytest.fixture()
def categorical_features() -> List[str]:
    return [
        "STDs:vaginal condylomatosis",
        "STDs:syphilis",
        "STDs:pelvic inflammatory disease",
        "STDs:genital herpes",
        "STDs:molluscum contagiosum",
        "STDs:HIV",
        "STDs:Hepatitis B",
        "STDs:HPV",
        "STDs:cervical condylomatosis",
        "STDs:AIDS",
        "Dx:Cancer",
        "Dx:CIN",
        "Dx:HPV",
        "Dx",
        "Smokes",
        "IUD",
        "Hormonal Contraceptives",
        "STDs",
        "STDs:condylomatosis",
        "STDs:vulvo-perineal condylomatosis",
    ]


@pytest.fixture
def numerical_features() -> List[str]:
    return [
        "Number of sexual partners",
        "First sexual intercourse",
        "Num of pregnancies",
        "Age",
        "Smokes (packs/year)",
        "Smokes (years)",
        "Hormonal Contraceptives (years)",
        "IUD (years)",
        "STDs: Time since first diagnosis",
        "STDs: Time since last diagnosis",
        "STDs: Number of diagnosis",
        "STDs (number)",
    ]


@pytest.fixture()
def features_to_drop() -> List[str]:
    return [
        "Citology",
        "Hinselmann",
        "Schiller",
    ]
