"""Display options classes."""

from pathlib import PosixPath
from typing import Any, Callable
from pydantic import Field, model_validator
from thuner.utils import BaseOptions, Retrieval, CallableField
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

    parent_local: str | PosixPath | None = Field(
        None,
        description="The local parent directory in which to save the figures.",
    )
    style: str = Field(
        "presentation",
        description=(
            "The style of the figures. See thuner.visualize.styles for options."
        ),
    )
    weights_filepath: str | None = Field(
        None,
        description="Filepath to the regridder weights.",
    )

    @model_validator(mode="after")
    def validate_parent_local(cls, values):
        """Ensure that the parent_local directory is set."""
        if values.parent_local is None:
            values.parent_local = str(get_outputs_directory() / "visualize")
        return values


class FigureOptions(BaseOptions):
    """Base class for figure options."""

    name: str = Field(..., description="The base name of the figure.")
    function: CallableField = Field(
        ...,
        description="The function used to generate the figure.",
    )
    style: str = Field(None, description="The style of the figure.")
    animate: bool = Field(True, description="Whether to animate the figure.")
    single_color: bool = Field(
        False,
        description="Whether to use a single color for the object masks.",
    )
    template: Any = Field(
        None,
        description=(
            "The template for the figure. This is typically created during runtime."
        ),
    )


class ObjectRuntimeOptions(VisualizeOptions):
    """Class for a given object's runtime visualization options."""

    name: str = Field(..., description="The object to generate runtime figures for.")
    figures: list[FigureOptions] = Field(
        ...,
        description="The types of figures to generate.",
    )
    animate: bool = Field(True, description="Whether to animate the figures.")
    single_color: bool = Field(
        False,
        description="Whether to use a single color for the object masks.",
    )

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

    objects: dict[str, ObjectRuntimeOptions] = Field(
        {},
        description="The objects to generate runtime figures for.",
    )


class HorizontalAttributeOptions(VisualizeOptions):
    """Class for horizontal attribute visualization options."""

    name: str = Field(..., description="The base name of the figure.")
    object_name: str = Field(..., description="The name of the object to visualize.")
    attribute_handlers: dict[str, list[Any]] = Field(
        None,
        description="Handlers for all the attributes to be shown in the figure.",
    )
    method: Retrieval | Callable | None = Field(
        None,
        description="Function and keyword arguments used to generate the figure.",
    )
    template: Any = Field(
        None,
        description=(
            "The template for the figure. This is typically created during runtime."
        ),
    )
    single_color: bool = Field(
        False,
        description="Whether to use a single color for the object masks.",
    )
    conversion_time_step: float = Field(
        3600,
        description="Time step in seconds used to convert velocities to displacements.",
    )
    altitude_titles: bool = Field(
        True,
        description="Show the altitudes the objects were detected at as titles.",
    )

    @model_validator(mode="after")
    def _initialize_method(cls, values):
        """Initialize the method for generating the figure."""
        if values.method is None:
            func = "thuner.visualize.attribute.detected_horizontal"
            values.method = Retrieval(function=func)
        return values


class GroupedHorizontalAttributeOptions(HorizontalAttributeOptions):
    """Class for grouped horizontal attribute visualization options."""

    member_objects: list[str] = Field(
        ["convective", "anvil"],
        description="The member objects to visualize.",
    )

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
