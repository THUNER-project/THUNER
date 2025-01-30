import numpy as np
from typing import Any, Dict
from pathlib import Path
import yaml
import inspect
from pydantic import Field, model_validator, BaseModel


def convert_value(value: Any) -> Any:
    """
    Convenience function to convert options attributes to types serializable as yaml.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [convert_value(v) for v in value.tolist()]
    if isinstance(value, BaseOptions):
        fields = value.model_fields.keys()
        return {field: convert_value(getattr(value, field)) for field in fields}
    if isinstance(value, dict):
        return {convert_value(k): convert_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [convert_value(v) for v in value]
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, type):
        # return full name of type, i.e. including module
        return f"{inspect.getmodule(value).__name__}.{value.__name__}"
    if type(value) is np.float32:
        return float(value)
    if inspect.isroutine(value):
        module = inspect.getmodule(value)
        return f"{module.__name__}.{value.__name__}"
    return value


class BaseOptions(BaseModel):
    """
    The base class for all options classes. This class is built on the pydantic
    BaseModel class, which is similar to python dataclasses but with type checking.
    """

    type: str = Field(None, description="Type of the options class.")

    # Allow arbitrary types in the options classes.
    class Config:
        arbitrary_types_allowed = True

    # Ensure that floats in all options classes are np.float32
    @model_validator(mode="after")
    def convert_floats(cls, values):
        for field in values.model_fields:
            if type(getattr(values, field)) is float:
                setattr(values, field, np.float32(getattr(values, field)))
        return values

    @model_validator(mode="after")
    def _set_type(cls, values):
        if values.type is None:
            values.type = cls.__name__
        return values

    def to_dict(self) -> Dict[str, Any]:
        fields = self.model_fields.keys()
        return {field: convert_value(getattr(self, field)) for field in fields}

    def to_yaml(self, filepath: str):
        Path(filepath).parent.mkdir(exist_ok=True, parents=True)
        with open(filepath, "w") as f:
            kwargs = {"default_flow_style": False, "allow_unicode": True}
            kwargs = {"sort_keys": False}
            yaml.dump(self.to_dict(), f, **kwargs)
