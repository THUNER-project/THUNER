"""Display options classes."""

from pathlib import PosixPath
from typing import Callable, Any
from pydantic import Field, model_validator
from thuner.utils import BaseOptions, Retrieval
from thuner.log import setup_logger
from thuner.config import get_outputs_directory


logger = setup_logger(__name__)


__all__ = [
    "VisualizeOptions",
    "FigureOptions",
    "ObjectRuntimeOptions",
    "RuntimeOptions",
    "HorizontalAttributeOptions",
]


class VisualizeOptions(BaseOptions):
    """Base class for visualization options."""

    _desc = "The local parent directory in which to save the figures."
    parent_local: str | PosixPath | None = Field(None, description=_desc)
    _desc = "The style of the figures. See thuner.visualize.styles for options."
    style: str = Field("presentation", description=_desc)
    _desc = "Filepath to the regridder weights."
    weights_filepath: str | None = Field(None, description=_desc)

    @model_validator(mode="after")
    def validate_parent_local(cls, values):
        """Ensure that the parent_local directory is set."""
        if values.parent_local is None:
            values.parent_local = str(get_outputs_directory() / "visualize")
        return values


class FigureOptions(BaseOptions):
    """Base class for figure options."""

    _desc = "The base name of the figure."
    name: str = Field(..., description=_desc)
    _desc = "The function used to generate the figure."
    function: Callable | str | None = Field(..., description=_desc)
    _desc = "The style of the figure."
    style: str = Field(None, description=_desc)
    _desc = "Whether to animate the figure."
    animate: bool = Field(True, description=_desc)
    _desc = "Whether to use a single color for the object masks."
    single_color: bool = Field(False, description=_desc)
    _desc = "The template for the figure. This is typically created during runtime."
    template: Any = Field(None, description=_desc)


class ObjectRuntimeOptions(VisualizeOptions):
    """Class for a given object's runtime visualization options."""

    _desc = "The object to generate runtime figures for."
    name: str = Field(..., description=_desc)
    _desc = "The types of figures to generate."
    figures: list[FigureOptions] = Field(..., description=_desc)
    _desc = "Whether to animate the figures."
    animate: bool = Field(True, description=_desc)
    _desc = "Whether to use a single color for the object masks."
    single_color: bool = Field(False, description=_desc)

    @model_validator(mode="after")
    def initialize_figures(cls, values):
        """Initialize the figure options."""
        for fig in values.figures:
            fig.style = values.style
            fig.animate = values.animate
            fig.single_color = values.single_color
        return values


class RuntimeOptions(BaseOptions):
    """Class for runtime visualization options."""

    _desc = "The objects to generate runtime figures for."
    objects: dict[str, ObjectRuntimeOptions] = Field({}, description=_desc)


class HorizontalAttributeOptions(VisualizeOptions):
    """Class for horizontal attribute visualization options."""

    _desc = "The base name of the figure."
    name: str = Field(..., description=_desc)
    _desc = "The name of the object to visualize."
    object_name: str = Field(..., description=_desc)
    _desc = "Handlers for all the attributes to be shown in the figure."
    attribute_handlers: dict[str, list[Any]] = Field(None, description=_desc)
    _desc = "Function and keyword arguments used to generate the figure."
    method: Retrieval | Callable | None = Field(None, description=_desc)
    _desc = "The template for the figure. This is typically created during runtime."
    template: Any = Field(None, description=_desc)
    _desc = "Whether to use a single color for the object masks."
    single_color: bool = Field(False, description=_desc)
    _desc = "Time step in seconds used to convert velocities to displacements."
    conversion_time_step: float = Field(3600, description=_desc)
    _desc = "Show the altitudes the objects were detected at as titles."
    altitude_titles: bool = Field(True, description=_desc)

    @model_validator(mode="after")
    def _initialize_method(cls, values):
        """Initialize the method for generating the figure."""
        if values.method is None:
            func = "thuner.visualize.attribute.detected_horizontal"
            values.method = Retrieval(function=func)
        return values


class GroupedHorizontalAttributeOptions(HorizontalAttributeOptions):
    """Class for grouped horizontal attribute visualization options."""

    _desc = "The member objects to visualize."
    member_objects: list[str] = Field(["convective", "anvil"], description=_desc)

    @model_validator(mode="after")
    def _check_dictionary_keys(cls, values):
        """Check member_objects are valid keys in attribute_handlers dictionary."""
        keys = list(values.attribute_handlers.keys())
        lengths = [len(keys), len(values.member_objects)]
        if len(set(lengths)) != 1:
            message = "The number of member objects and attribute handlers must match."
            raise ValueError(message)
        return values

    @model_validator(mode="after")
    def _initialize_method(cls, values):
        """Initialize the method for generating the figure."""
        if values.method is None:
            func = "thuner.visualize.attribute.grouped_horizontal"
            values.method = Retrieval(function=func)
        return values
