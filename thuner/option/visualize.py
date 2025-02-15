"""Display options functions."""

from pathlib import PosixPath
from typing import Callable, Any
from pydantic import Field, model_validator
from thuner.utils import BaseOptions
from thuner.log import setup_logger
from thuner.config import get_outputs_directory


logger = setup_logger(__name__)


class VisualizeOptions(BaseOptions):
    """Base class for visualization options."""

    _desc = "The local parent directory in which to save the figures."
    parent_local: str | PosixPath | None = Field(None, description=_desc)
    _desc = "The style of the figures. See thuner.visualize.styles for options."
    style: str = Field("presentation", description=_desc)

    @model_validator(mode="after")
    def validate_parent_local(cls, values):
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
    _desc = "The attributes to display."
    attributes: list[str] = Field(
        ["ambient", "relative_velocity", "velocity", "offset"], description=_desc
    )
    _desc = "Whether to perform quality control."
    quality_control: bool = Field(True, description=_desc)
    _desc = "The fields to display."
    fields: list[str] = Field(["reflectivity"], description=_desc)
    _desc = "The extent of the figure."
    extent: list[float] | None = Field(None, description=_desc)
    _desc = "The template for the figure. This is typically created during runtime."
    template: Any = Field(None, description=_desc)
    _desc = "Whether to use a single color for the object masks."
    single_color: bool = Field(False, description=_desc)


def horizontal_attribute_options(
    name,
    save=True,
    parent_local=None,
    attributes=None,
    quality_control=True,
    fields=None,
    extent=None,
    template=None,
    single_color=False,
    style="paper",
):
    """Default options for horizontal attribute visualization."""

    # Set the default object attributes to display
    if attributes is None:
        attributes = ["ambient", "relative_velocity", "velocity", "offset"]
    # Set the default dataset fields to display
    if fields is None:
        fields = ["reflectivity"]

    options = {
        **boilerplate_options(name, save, parent_local),
        "attributes": attributes,
        "quality_control": quality_control,
        "fields": fields,
        "extent": extent,
        "template": template,
        "style": style,
        "single_color": single_color,
    }
    return options


# def save_display_options(
#     display_options, options_directory=None, filename="visualize", append_time=False
# ):
#     """Save the display options."""

#     if options_directory is None:
#         options_directory = get_outputs_directory() / "options/visualize"
#     utils.save_options(
#         display_options, filename, options_directory, append_time=append_time
#     )
