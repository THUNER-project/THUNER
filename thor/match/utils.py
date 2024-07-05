"""General matching methods."""


def get_masks(object_tracks, object_options, matched=False):
    """Get the appropriate current and previous masks for matching."""
    mask_type = matched * "_matched" + "_mask"
    current_mask = object_tracks[f"current{mask_type}"]
    previous_mask = object_tracks[f"previous{mask_type}s"][-1]
    if "grouping" in object_options.keys():
        matched_object = object_options["tracking"]["options"]["matched_object"]
        current_mask = current_mask[f"{matched_object}_mask"]
        if previous_mask is not None:
            previous_mask = previous_mask[f"{matched_object}_mask"]
    return current_mask, previous_mask


def get_grids(object_tracks, object_options):
    """Get the appropriate current and previous grids for matching."""
    current_grid = object_tracks["current_grid"]
    previous_grid = object_tracks["previous_grids"][-1]
    if "grouping" in object_options.keys():
        matched_object = object_options["tracking"]["options"]["matched_object"]
        current_grid = current_grid[f"{matched_object}_grid"]
        if previous_grid is not None:
            previous_grid = previous_grid[f"{matched_object}_grid"]
    return current_grid, previous_grid
