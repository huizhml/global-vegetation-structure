from dataclasses import dataclass
from omegaconf import MISSING

@dataclass
class ClassConfig:
    _target_: str = MISSING
    target_type: str = 'class'
    target_method: str = MISSING

@dataclass
class FunctionConfig:
    _target_: str = MISSING
    target_type: str = 'function'