from dataclasses import dataclass, field
from ml_project.constants.constants import RANDOM_STATE


@dataclass()
class LogisticRegressionParams:
    model_type: str = field(default="LogisticRegression")
    penalty: str = field(default="l2")
    C: float = field(default=1.0)
    random_state: int = field(default=RANDOM_STATE)


@dataclass()
class RandomForestParams:
    model_type: str = field(default="RandomForestClassifier")
    n_estimators: int = field(default=100)
    random_state: int = field(default=RANDOM_STATE)
