from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelConfig:
    name: str = "random_forest"

@dataclass
class RFConfig(ModelConfig):
    n_estimators: int = 100
    max_depth: Optional[int] = None

@dataclass
class SVMConfig(ModelConfig):
    kernel: str = "rbf"
    C: float = 1.0
    gamma: str = "scale"

@dataclass
class DatasetConfig:
    test_size: float = 0.2
    random_state: int = 42

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    rf: RFConfig = field(default_factory=RFConfig)
    svm: SVMConfig = field(default_factory=SVMConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
