from ml_project.models import split_train_val_data, read_data
from ml_project.entities import SplittingParams

from ml_project.constants.constants import RANDOM_STATE


def test_load_dataset(dataset_path: str, target_col: str):
    data = read_data(dataset_path)
    assert len(data) == 50
    assert target_col in data.keys()


def test_split_dataset(tmpdir, dataset_path: str):
    val_size = 0.2
    splitting_params = SplittingParams(random_state=RANDOM_STATE, val_size=val_size,)
    data = read_data(dataset_path)
    train, val = split_train_val_data(data, splitting_params)
    assert train.shape[0] == 40
    assert val.shape[0] == 10
