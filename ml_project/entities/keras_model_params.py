from dataclasses import dataclass, field
from ml_project.constants.constants import RANDOM_STATE


@dataclass()
class FirstLayerParams:
    units: int = field(default=500)
    input_dim: int = field(default=50)
    kernel_initializer: str = field(default="uniform")
    activation: str = field(default="relu")


@dataclass()
class SecondLayerParams:
    units: int = field(default=200)
    kernel_initializer: str = field(default="uniform")
    activation: str = field(default="relu")


@dataclass()
class OutputLayerParams:
    units: int = field(default=1)
    kernel_initializer: str = field(default="uniform")
    activation: str = field(default="sigmoid")


@dataclass()
class CompileParams:
    loss: str = field(default="binary_crossentropy")
    optimizer: str = field(default="adam")
    metrics: str = field(default="AUC")


@dataclass()
class TrainParams:
    epochs: int = field(default=20)
    batch_size: int = field(default=200)
    verbose: int = field(default=2)


@dataclass()
class KerasSimpleNNParams:
    first_layer: FirstLayerParams
    second_layer: SecondLayerParams
    output_layer: OutputLayerParams
    train: TrainParams
    compile: CompileParams
    model_type: str = field(default="KerasSimpleNN")
    random_state: int = field(default=RANDOM_STATE)
    dropout: int = field(default=0.5)
