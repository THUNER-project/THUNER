"""General utilities for object attributes."""

from pathlib import Path
from pydantic import BaseModel, ConfigDict, model_validator
import numpy as np
import pandas as pd
import xarray as xr
from thuner.option.attribute import Attribute, AttributeGroup, AttributeType, Attributes
from thuner.log import setup_logger
from thuner.utils import store_path

logger = setup_logger(__name__)


__all__ = [
    "read_attribute_zarr",
    "read_attribute",
    "AttributesRecord",
    "time_offset",
]


def get_ids(object_tracks, matched, member_object):
    """Get object ids from the match record to avoid recalculating."""
    current_attributes = object_tracks.current_attributes
    if matched:
        id_name = "universal_id"
    else:
        id_name = "id"
    if member_object is not None:
        ids = current_attributes.member_attributes[member_object]["core"][id_name]
    else:
        ids = current_attributes.attribute_types["core"][id_name]
    return ids


def get_nearest_points(
    stacked_mask: xr.DataArray | xr.Dataset,
    id_number: int,
    ds: xr.Dataset | xr.DataArray,
):
    """
    Get the nearest points in a tagging dataset to a given object id.

    Parameters
    ----------
    stacked_mask : xarray.DataArray
        mask containing object ids with latitude and longitude stacked into a new
        dimension called 'points'.
    id_number : int
        Object ID number to get from stacked_mask.
    ds : xarray.Dataset | xarray.DataArray
        Tagging dataset containing latitude and longitude.

    Returns
    -------
    list[tuple]
        List of tuples containing the latitude and longitude of the nearest points.
    """
    points = stacked_mask.where(stacked_mask == id_number, drop=True).points.values
    lats, lons = zip(*points)
    lats_da = xr.DataArray(list(lats), dims="points")
    lons_da = xr.DataArray(list(lons), dims="points") % 360
    ds_grid = ds[["latitude", "longitude"]]
    ds_points = ds_grid.sel(latitude=lats_da, longitude=lons_da, method="nearest")
    ds_lats = ds_points.latitude.values.tolist()
    ds_lons = ds_points["longitude"].values.tolist()
    return list(set(zip(ds_lats, ds_lons)))


def _init_attr_type(attribute_type: AttributeType):
    """Initialize attributes lists for a given attribute type."""
    attributes = {}
    for attr in attribute_type.attributes:
        if isinstance(attr, AttributeGroup):
            for attr_attr in attr.attributes:
                attributes[attr_attr.name] = []
        elif isinstance(attr, Attribute):
            attributes[attr.name] = []
        else:
            raise ValueError(f"Unknown type {attr.type}.")
    return attributes


class AttributesRecord(BaseModel):
    """
    Class for storing attributes recorded during the tracking process
    """

    # Allow arbitrary types in the class.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    attribute_options: Attributes
    name: str = None
    attribute_types: dict | None = None
    member_attributes: dict | None = None

    @model_validator(mode="after")
    def _check_name(self):
        if self.name is None:
            self.name = self.attribute_options.name
        elif self.name != self.attribute_options.name:
            raise ValueError("Name must match attribute_options name.")
        return self

    @model_validator(mode="after")
    def _initialize_attributes(self):
        options = self.attribute_options
        if options is None:
            return self
        self.attribute_types = {}
        for attr_type in options.attribute_types:
            self.attribute_types[attr_type.name] = _init_attr_type(attr_type)
        if options.member_attributes is not None:
            self.member_attributes = {}
            for obj, obj_attributes in options.member_attributes.items():
                obj_attr = {}
                for attr_type in obj_attributes.attribute_types:
                    obj_attr[attr_type.name] = _init_attr_type(attr_type)
                self.member_attributes[obj] = obj_attr
        return self


# Mapping of string representations to actual data types
string_to_data_type = {
    "float": float,
    "int": int,
    "datetime64[s]": "datetime64[s]",
    "bool": bool,
    "str": str,
}


def time_offset():
    """Convenience function to build a TimeOffset attribute."""
    return Attribute(
        name="time_offset",
        data_type=int,
        units="min",
        description="Time offset in minutes from object detection time.",
    )


def setup_interp(
    attribute_group: AttributeGroup,
    input_records,
    object_tracks,
    dataset: str,
    member_object: str = None,
):
    name = object_tracks.name
    excluded = ["time", "id", "universal_id", "latitude", "longitude", "altitude"]
    excluded += ["time_offset"]
    attributes = attribute_group.attributes
    names = [attr.name for attr in attributes if attr.name not in excluded]
    tag_input_records = input_records.tag
    current_time = object_tracks.times[-1]

    # Get object centers
    if member_object is None:
        core_attributes = object_tracks.current_attributes.attribute_types["core"]
    else:
        core_attributes = object_tracks.current_attributes.member_attributes
        core_attributes = core_attributes[member_object]["core"]

    ds = tag_input_records[dataset].dataset
    ds["longitude"] = ds["longitude"] % 360
    return name, names, ds, core_attributes, current_time


def get_current_mask(object_tracks, matched=False):
    """Get the appropriate previous mask."""
    if matched:
        mask_type = "matched_masks"
    else:
        mask_type = "masks"
    mask = getattr(object_tracks, mask_type)[-1]
    return mask


def attribute_from_core(attribute, object_tracks, member_object):
    """Get attribute from core object properties."""
    # Check if grouped object
    current_attributes = object_tracks.current_attributes
    if member_object is not None and member_object is not object_tracks.name:
        member_attr = current_attributes.member_attributes
        attr = member_attr[member_object]["core"][attribute.name]
    else:
        core_attr = current_attributes.attribute_types["core"]
        attr = core_attr[attribute.name]
    return {attribute.name: attr}


def attributes_dataframe(recorded_attributes, attribute_type):
    """Create a pandas DataFrame from object attributes dictionary."""

    data_types = get_data_type_dict(attribute_type)
    data_types.pop("time")
    df = pd.DataFrame(recorded_attributes).astype(data_types)
    multi_index = ["time"]
    if "time_offset" in recorded_attributes.keys():
        multi_index.append("time_offset")
    if "universal_id" in recorded_attributes.keys():
        id_index = "universal_id"
    else:
        id_index = "id"
    multi_index.append(id_index)
    if "altitude" in recorded_attributes.keys():
        multi_index.append("altitude")
    df.set_index(multi_index, inplace=True)
    df.sort_index(inplace=True)
    return df


def get_indexes(attribute_type: AttributeType):
    """Get the indexes for the attribute DataFrame."""
    all_indexes = ["time", "time_offset", "event_start", "universal_id", "id"]
    all_indexes += ["altitude"]
    indexes = []
    for attribute in attribute_type.attributes:
        if isinstance(attribute, AttributeGroup):
            for attr in attribute.attributes:
                if attr.name in all_indexes:
                    indexes.append(attr.name)
        else:
            if attribute.name in all_indexes:
                indexes.append(attribute.name)
    return indexes


def get_names(attribute_type: AttributeType):
    """Get the names of the attributes in the attribute type."""
    names = []
    for attribute in attribute_type.attributes:
        if isinstance(attribute, AttributeGroup):
            for attr in attribute.attributes:
                names.append(attr.name)
        else:
            names.append(attribute.name)
    return names


def get_precision_dict(attribute_type: AttributeType):
    """Get precision dictionary for attribute options."""
    precision_dict = {}
    for attribute in attribute_type.attributes:
        if isinstance(attribute, AttributeGroup):
            for attr in attribute.attributes:
                if attr.data_type == float:
                    precision_dict[attr.name] = attr.precision
        else:
            if attribute.data_type == float:
                precision_dict[attribute.name] = attribute.precision
    return precision_dict


def get_data_type_dict(attribute_type: AttributeType):
    """Get precision dictionary for attribute options."""
    data_type_dict = {}
    for attribute in attribute_type.attributes:
        if isinstance(attribute, AttributeGroup):
            # If the attribute is a group, get data type for each attribute in group
            for attr in attribute.attributes:
                data_type_dict[attr.name] = attr.data_type
        else:
            data_type_dict[attribute.name] = attribute.data_type
    return data_type_dict


def _attribute_df_from_dataset(ds, columns=None, times=None):
    """Convert an opened attribute ``Dataset`` to the standard multi-indexed DataFrame.

    Shared by ``read_attribute_zarr`` regardless of whether the dataset came from
    opening a single zarr group or from navigating an already-open ``DataTree``. The
    group's index columns are read from ``ds.attrs["index_columns"]`` — persisted by
    the writer — so the reader doesn't need to reconstruct an ``AttributeType``.
    """
    df = ds.to_dataframe()
    if df.index.name == "record" or "record" in df.index.names:
        df = df.reset_index(drop=True)

    # Missing strings round-trip from zarr as "" (untouched) or "nan" (e.g.
    # written by stitching's relabel_parents) — fold both back to NaN.
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    for c in obj_cols:
        df[c] = df[c].where(~df[c].isin(["", "nan"]), np.nan)

    if times is not None and "time" in df.columns:
        df = df[df["time"].isin(list(times))]

    indexes = [c for c in ds.attrs.get("index_columns", []) if c in df.columns]
    if columns is not None:
        keep = indexes + [c for c in columns if c not in indexes and c in df.columns]
        df = df[keep]
    if indexes:
        df = df.set_index(indexes).sort_index()
    return df


def read_attribute_zarr(store_path, group, columns=None, times=None, tree=None):
    """Read a zarr attribute group and return a multi-indexed DataFrame.

    Parameters
    ----------
    store_path : str | Path
        Path to the unified zarr store for a run.
    group : str
        Group path within the store, e.g. ``attributes/cell/core``.
    columns : list[str], optional
        Subset of (non-index) columns to keep.
    times : iterable, optional
        Subset of times to keep (matched against the 'time' column).
    tree : xr.DataTree, optional
        An already-open DataTree for the store (e.g. from ``xr.open_datatree``). When
        given, the group is read from the tree instead of re-opening ``store_path`` —
        useful when reading many groups from the same store.
    """
    if tree is not None:
        ds = tree["/" + group.lstrip("/")].to_dataset()
    else:
        ds = xr.open_zarr(store_path, group=group)
    return _attribute_df_from_dataset(ds, columns=columns, times=times)


def read_attribute(output_directory, *parts, columns=None, times=None, tree=None):
    """Read an attribute table from the unified zarr store.

    ``parts`` are joined with ``/`` to form the group path inside
    ``<output_directory>/<configured zarr store name>``. Pass ``tree`` (from
    ``xr.open_datatree``) to read from an already-open DataTree instead of
    re-opening the store.
    """
    store = store_path(output_directory)
    group = "/".join(str(p) for p in parts)
    return read_attribute_zarr(store, group, columns=columns, times=times, tree=tree)
