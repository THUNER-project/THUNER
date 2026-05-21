"""Write object attributes to the unified zarr store.

Each attribute table is stored as an xarray ``Dataset`` under a hierarchical
group inside ``<output_directory>/<store_name>``. The store name comes from
``thuner.config.get_zarr_store_name`` (default ``output.zarr``).
``AttributeType`` metadata is distributed: the dataset's ``.attrs`` carries
type-level fields plus a compact ``groups`` map preserving ``AttributeGroup``
structure, while each variable's ``.attrs`` carries that attribute's own
metadata (``description``, ``units``, ``data_type``, ``precision``,
``retrieval``, ``attribute_group``).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from thuner.utils import format_time, convert_value
from thuner.log import setup_logger
from thuner.config import get_zarr_store_name
import thuner.attribute.utils as utils
import thuner.write.utils as write_utils
from thuner.option.track import BaseObjectOptions
from thuner.option.attribute import Attribute, AttributeGroup, AttributeType

logger = setup_logger(__name__)


def _retrieval_to_dict(retrieval):
    """Serialise a Retrieval into a plain dict (or None)."""
    if retrieval is None:
        return None
    return convert_value(retrieval)


def _attribute_variable_attrs(attribute: Attribute) -> dict:
    """Build the per-variable .attrs dict describing a single Attribute."""
    data_type = convert_value(attribute.data_type)
    attrs = {
        "description": attribute.description or "",
        "data_type": data_type,
    }
    # ``units`` is reserved by xarray's CF time encoding on datetime variables,
    # so route the human-readable units through a separate key for those.
    units = attribute.units or ""
    if data_type == "numpy.datetime64":
        if units:
            attrs["time_units"] = units
    else:
        attrs["units"] = units
    if attribute.precision is not None:
        attrs["precision"] = int(attribute.precision)
    retrieval = _retrieval_to_dict(attribute.retrieval)
    if retrieval is not None:
        attrs["retrieval"] = retrieval
    return attrs


def _apply_attribute_type_metadata(ds: xr.Dataset, attribute_type: AttributeType):
    """Write distributed metadata onto ``ds`` and its variables in place."""
    if attribute_type is None:
        return

    for entry in attribute_type.attributes:
        if isinstance(entry, AttributeGroup):
            for attr in entry.attributes:
                if attr.name in ds.variables:
                    ds[attr.name].attrs.update(_attribute_variable_attrs(attr))
        elif isinstance(entry, Attribute):
            if entry.name in ds.variables:
                ds[entry.name].attrs.update(_attribute_variable_attrs(entry))
        else:
            raise TypeError(f"Unsupported attribute entry: {type(entry)!r}")

    ds.attrs["name"] = attribute_type.name
    ds.attrs["description"] = attribute_type.description or ""
    if attribute_type.dataset is not None:
        ds.attrs["dataset"] = attribute_type.dataset
    ds.attrs["index_columns"] = utils.get_indexes(attribute_type)


def _df_to_dataset(
    df: pd.DataFrame, attribute_type: AttributeType | None
) -> xr.Dataset:
    """Convert an attributes DataFrame to a long-form xarray Dataset."""
    df_reset = df.reset_index()
    ds = df_reset.to_xarray()
    if "index" in ds.coords:
        ds = ds.drop_vars("index")
    if "index" in ds.dims:
        ds = ds.rename_dims({"index": "record"})
    _apply_attribute_type_metadata(ds, attribute_type)
    write_utils.pin_datetime_encoding(ds)
    return ds


def store_path(output_directory) -> Path:
    """Return ``<output_directory>/<configured zarr store name>``."""
    return Path(output_directory) / get_zarr_store_name()


def write_attributes(store_path_, group, df, attribute_type):
    """Write or append a single attribute table to its group in the store.

    The first write creates the group. Subsequent writes append along the
    ``record`` dim.
    """
    if df is None or len(df) == 0:
        return
    precision_dict = utils.get_precision_dict(attribute_type)
    df = df.round(precision_dict)
    df = df.sort_index()
    ds = _df_to_dataset(df, attribute_type)
    store_path_ = Path(store_path_)
    store_path_.parent.mkdir(parents=True, exist_ok=True)
    if write_utils.zarr_group_exists(store_path_, group):
        ds.to_zarr(store_path_, group=group, mode="a", append_dim="record")
    else:
        ds.to_zarr(store_path_, group=group, mode="a")
    logger.debug("Wrote %s records to %s::%s", len(df), store_path_, group)


def write_setup(object_tracks, object_options, output_directory):
    """Log the time window being flushed and return the last-write timestamp."""
    object_name = object_options.name
    _last_write_time = object_tracks._last_write_time
    write_interval = np.timedelta64(object_options.write_interval, "h")
    last_write_str = format_time(_last_write_time, filename_safe=False, day_only=False)
    next_write_time = _last_write_time + write_interval
    current_str = format_time(next_write_time, filename_safe=False, day_only=False)
    message = (
        f"Writing {object_name} attributes from {last_write_str} to "
        f"{current_str}, inclusive and non-inclusive, respectively."
    )
    logger.info(message)
    return last_write_str


def write(object_tracks, object_options: BaseObjectOptions, output_directory):
    """Flush accumulated attributes for one object to the zarr store."""
    if object_options.attributes is None:
        return

    write_setup(object_tracks, object_options, output_directory)

    object_name = object_options.name
    store = store_path(output_directory)
    group_prefix = f"attributes/{object_name}"

    member_attribute_options = object_options.attributes.member_attributes
    if member_attribute_options is not None:
        recorded_member_attr = object_tracks.attributes.member_attributes
        for obj in member_attribute_options.keys():
            for attribute_type in member_attribute_options[obj].attribute_types:
                group = f"{group_prefix}/{obj}/{attribute_type.name}"
                rec_attr = recorded_member_attr[obj][attribute_type.name]
                df = utils.attributes_dataframe(rec_attr, attribute_type)
                write_attributes(store, group, df, attribute_type)

    obj_attr = object_tracks.attributes.attribute_types
    for attribute_type in object_options.attributes.attribute_types:
        attributes = obj_attr[attribute_type.name]
        group = f"{group_prefix}/{attribute_type.name}"
        df = utils.attributes_dataframe(attributes, attribute_type)
        write_attributes(store, group, df, attribute_type)

    attr_options = object_options.attributes
    object_tracks.attributes = utils.AttributesRecord(attribute_options=attr_options)


def write_final(tracks, track_options, output_directory):
    """Flush any remaining accumulated attributes to the unified zarr store."""
    for index, level_options in enumerate(track_options.levels):
        for object_options in level_options.objects:
            obj_name = object_options.name
            obj_tracks = tracks.levels[index].objects[obj_name]
            write(obj_tracks, object_options, output_directory)


def write_attribute(output_directory, *parts, df, attribute_type=None):
    """Write a DataFrame to a group in the unified zarr store.

    ``parts`` are joined with ``/`` to form the group path inside the
    configured store under ``output_directory``. Returns the rounded/sorted
    DataFrame.
    """
    store = store_path(output_directory)
    group = "/".join(str(p) for p in parts)
    write_attributes(store, group, df, attribute_type)
    if attribute_type is not None:
        df = df.round(utils.get_precision_dict(attribute_type)).sort_index()
    return df
