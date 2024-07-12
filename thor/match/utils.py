"""General matching methods."""


def get_masks(object_tracks, object_options, matched=False, num_previous=1):
    """Get the appropriate current and previous masks for matching."""
    mask_type = matched * "_matched" + "_mask"
    current_mask = object_tracks[f"current{mask_type}"]
    previous_masks = [
        object_tracks[f"previous{mask_type}s"][-i] for i in range(1, num_previous + 1)
    ]
    masks = [current_mask] + previous_masks
    if "grouping" in object_options.keys():
        matched_object = object_options["tracking"]["options"]["matched_object"]
        for i in range(len(masks)):
            if masks[i] is not None:
                masks[i] = masks[i][f"{matched_object}_mask"]
    return masks


def get_grids(object_tracks, object_options, num_previous=1):
    """Get the appropriate current and previous grids for matching."""
    current_grid = object_tracks["current_grid"]
    previous_grids = [
        object_tracks["previous_grids"][-i] for i in range(1, num_previous + 1)
    ]
    grids = [current_grid] + previous_grids
    if "grouping" in object_options.keys():
        matched_object = object_options["tracking"]["options"]["matched_object"]
        for i in range(len(grids)):
            if grids[i] is not None:
                grids[i] = grids[i][f"{matched_object}_grid"]
    return grids
