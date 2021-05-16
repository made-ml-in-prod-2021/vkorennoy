from dataclasses import dataclass, field
from ml_project.constants.constants import RANDOM_STATE


@dataclass()
class SplittingParams:
    val_size: float = field(default=0.2)
    random_state: int = field(default=RANDOM_STATE)
