from dataclasses import dataclass, field
from omegaconf import MISSING

@dataclass
class ClassConfig:
    _target_: str = MISSING
    target_type: str = 'class'
    target_method: str = MISSING
    func_args: dict = field(default_factory=lambda: {})

@dataclass
class FunctionConfig:
    _target_: str = MISSING
    target_type: str = 'function'