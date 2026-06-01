"""Default visualization configurations and attribute handlers."""

import thuner.option.visualize as visualize_option
import thuner.visualize.runtime as vis_runtime
from thuner.utils import Retrieval, AttributeHandler, store_path

__all__ = [
    "runtime",
    "synthetic_runtime",
    "build_velocity_handler",
    "build_horizontal_text_handler",
    "build_displacement_handler",
    "build_orientation_handler",
    "detected_attribute_handlers",
    "grouped_attribute_handlers",
]


def runtime(visualize_directory, objects=["mcs"]):
    """Build default options for runtime visualization."""

    objects_dict = {}
    for obj in objects:
        match_figure = visualize_option.FigureOptions(
            name="tint_match",
            function=vis_runtime.visualize_tint_match,
        )
        figures = visualize_option.ObjectRuntimeOptions(
            name=obj,
            parent_local=visualize_directory,
            figures=[match_figure],
        )
        objects_dict[figures.name] = figures
    visualize_options = visualize_option.RuntimeOptions(objects=objects_dict)
    return visualize_options


def synthetic_runtime(visualize_directory):
    """Build default options for runtime visualization."""

    match_figure = visualize_option.FigureOptions(
        name="match",
        function=vis_runtime.visualize_tint_match,
    )
    convective_figures = visualize_option.ObjectRuntimeOptions(
        name="convective",
        parent_local=visualize_directory,
        figures=[match_figure],
    )

    objects_dict = {convective_figures.name: convective_figures}
    visualize_options = visualize_option.RuntimeOptions(objects=objects_dict)
    return visualize_options


def build_velocity_handler(
    output_parent,
    attributes,
    quality_variables,
    name="velocity",
    color="tab:purple",
    label="Object Velocity",
    reverse=False,
):
    """Convenience function to build a velocity attribute handler."""
    velocity_filepath = str(store_path(output_parent, "analysis", "velocities"))
    quality_filepath = str(store_path(output_parent, "analysis", "quality"))
    vis_func = "thuner.visualize.attribute.velocity_horizontal"
    vis_kwargs = {"color": color, "reverse": reverse}
    method = Retrieval(function=vis_func, keyword_arguments=vis_kwargs)
    leg_func = "thuner.visualize.horizontal.displacement_legend_artist"
    leg_kwargs = {"color": color, "label": label}
    legend_method = Retrieval(function=leg_func, keyword_arguments=leg_kwargs)
    return AttributeHandler(
        name=name,
        attributes=attributes,
        filepath=velocity_filepath,
        method=method,
        label=label,
        legend_method=legend_method,
        quality_filepath=quality_filepath,
        quality_variables=quality_variables,
    )


def build_horizontal_text_handler(
    output_parent, attributes, quality_variables, name="universal_id"
):
    """Convenience function to build a horizontal text attribute handler."""
    velocity_filepath = str(store_path(output_parent, "analysis", "velocities"))
    quality_filepath = str(store_path(output_parent, "analysis", "quality"))
    vis_func = "thuner.visualize.attribute.text_horizontal"
    vis_kwargs = {"labelled_attribute": "universal_id"}
    method = Retrieval(function=vis_func, keyword_arguments=vis_kwargs)
    return AttributeHandler(
        name=name,
        attributes=attributes,
        filepath=velocity_filepath,
        method=method,
        label="Object ID",
        quality_filepath=quality_filepath,
        quality_variables=quality_variables,
    )


def build_displacement_handler(
    output_parent,
    attributes,
    quality_variables,
    name="offset",
    color="tab:blue",
    label="Stratiform Offset",
    reverse=False,
):
    """Convenience function to build a displacement attribute handler."""
    group_filepath = str(store_path(output_parent, "attributes", "mcs", "group"))
    quality_filepath = str(store_path(output_parent, "analysis", "quality"))
    vis_func = "thuner.visualize.attribute.displacement_horizontal"
    vis_kwargs = {"color": color, "reverse": reverse}
    method = Retrieval(function=vis_func, keyword_arguments=vis_kwargs)
    leg_kwargs = {"color": color, "label": label}
    leg_func = "thuner.visualize.horizontal.displacement_legend_artist"
    legend_method = Retrieval(function=leg_func, keyword_arguments=leg_kwargs)
    return AttributeHandler(
        name=name,
        attributes=attributes,
        method=method,
        filepath=group_filepath,
        label="Stratiform Offset",
        legend_method=legend_method,
        quality_filepath=quality_filepath,
        quality_variables=quality_variables,
    )


def build_orientation_handler(
    output_parent,
    quality_variables,
    attributes=["major", "orientation"],
    name="orientation",
    style="presentation",
    label="Major Axis",
):
    """Convenience function to build an orientation attribute handler."""
    ellipse_filepath = str(
        store_path(output_parent, "attributes", "mcs", "convective", "ellipse")
    )
    quality_filepath = str(store_path(output_parent, "analysis", "quality"))
    vis_func = "thuner.visualize.attribute.orientation_horizontal"
    method = Retrieval(function=vis_func)
    label = "Major Axis"
    leg_func = "thuner.visualize.horizontal.orientation_legend_artist"
    leg_kwargs = {"label": label, "style": style}
    legend_method = Retrieval(function=leg_func, keyword_arguments=leg_kwargs)
    return AttributeHandler(
        name=name,
        attributes=attributes,
        method=method,
        filepath=ellipse_filepath,
        label=label,
        quality_filepath=quality_filepath,
        legend_method=legend_method,
        quality_variables=quality_variables,
    )


def detected_attribute_handlers(
    output_parent, object_name, style="presentation", attributes=None
):
    """Build default attribute handlers for detected objects."""
    if attributes is None:
        attributes = ["universal_id", "velocity"]
    base_qualities = ["contained", "duration"]

    velocity_handler = build_velocity_handler(
        output_parent=output_parent,
        attributes=["u", "v"],
        quality_variables=base_qualities + ["velocity"],
        name="velocity",
        color="tab:purple",
        label="System Velocity",
    )

    id_handler = build_horizontal_text_handler(
        output_parent=output_parent,
        attributes=["universal_id"],
        quality_variables=base_qualities,
    )

    return {object_name: [id_handler, velocity_handler]}


def grouped_attribute_handlers(
    output_parent, style="presentation", member_objects=None, attributes=None
):
    """Build default attribute handlers for grouped objects."""

    if member_objects is None:
        member_objects = ["convective", "anvil"]
    if attributes is None:
        # Initialize dictionary containing the attribute names for each member object.
        attributes = {k: [] for k in member_objects}
        conv_attr = ["universal_id", "velocity", "offset", "orientation"]
        anv_attr = ["universal_id", "offset"]
        attributes[member_objects[0]] = conv_attr
        attributes[member_objects[1]] = anv_attr

    base_qualities = ["convective_contained", "anvil_contained", "duration"]

    velocity_handler = build_velocity_handler(
        output_parent=output_parent,
        attributes=["u", "v"],
        quality_variables=base_qualities + ["velocity"],
        name="velocity",
        color="tab:purple",
        label="System Velocity",
    )

    ambient_handler = build_velocity_handler(
        output_parent=output_parent,
        attributes=["u_ambient", "v_ambient"],
        quality_variables=base_qualities,
        name="ambient",
        color="tab:red",
        label="Ambient Wind",
    )

    shear_handler = build_velocity_handler(
        output_parent=output_parent,
        attributes=["u_shear", "v_shear"],
        quality_variables=base_qualities + ["shear"],
        name="shear",
        color="darkblue",
        label="Ambient Shear",
    )

    relative_handler = build_velocity_handler(
        output_parent=output_parent,
        attributes=["u_relative", "v_relative"],
        quality_variables=base_qualities + ["relative_velocity"],
        color="darkgreen",
        label="Relative System Velocity",
        name="relative",
    )

    inflow_handler = build_velocity_handler(
        output_parent=output_parent,
        attributes=["u_relative", "v_relative"],
        quality_variables=base_qualities + ["relative_velocity"],
        color="darkgreen",
        label="System Relative Inflow",
        name="inflow",
        reverse=True,
    )

    id_handler = build_horizontal_text_handler(
        output_parent=output_parent,
        attributes=["universal_id"],
        quality_variables=base_qualities,
    )

    offset_handler_convective = build_displacement_handler(
        output_parent=output_parent,
        attributes=["x_offset", "y_offset"],
        quality_variables=base_qualities + ["offset"],
    )
    offset_handler_anvil = build_displacement_handler(
        output_parent=output_parent,
        attributes=["x_offset", "y_offset"],
        quality_variables=base_qualities + ["offset"],
        reverse=True,
    )

    orientation_handler = build_orientation_handler(
        output_parent,
        quality_variables=base_qualities + ["axis_ratio"],
        style=style,
    )

    all_conv = [id_handler, velocity_handler, ambient_handler]
    all_conv += [shear_handler, relative_handler, offset_handler_convective]
    all_conv += [inflow_handler, orientation_handler]
    all_anvil = [id_handler, offset_handler_anvil]
    conv_handlers = [h for h in all_conv if h.name in attributes[member_objects[0]]]
    anvil_handlers = [h for h in all_anvil if h.name in attributes[member_objects[1]]]

    return dict(zip(member_objects, [conv_handlers, anvil_handlers]))
