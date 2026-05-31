"""General input data processing."""

from thuner.log import setup_logger
from thuner.utils import time_in_dataset_range

logger = setup_logger(__name__)


def boilerplate_update(
    time, input_record, track_options, dataset_options, grid_options
):
    """Update the dataset."""

    earliest_time = time + dataset_options.start_buffer
    latest_time = time + dataset_options.end_buffer
    cond = not time_in_dataset_range(earliest_time, input_record.dataset)
    cond = cond or not time_in_dataset_range(latest_time, input_record.dataset)
    if cond:
        dataset_options.update_input_record(
            time, input_record, track_options, grid_options
        )


def update_track_input_records(
    time,
    track_input_records,
    track_options,
    data_options,
    grid_options,
    output_directory,
):
    """
    Update the per-time-step input records (grids, masks and boundaries).

    For each tracking dataset we (1) rotate the previous time step's "next" grid and
    boundary data into their deques so the deques stay aligned with one another, one
    entry per time step; (2) load a new file if ``time`` falls outside the currently
    loaded dataset; and (3) extract this time step's grid and boundary data from the
    current dataset. Boundary data mirrors the grids deque so visualization and quality
    control can index them consistently.
    """
    for name in track_input_records.keys():
        input_record = track_input_records[name]
        dataset_options = data_options.dataset_by_name(name)

        # 1. Rotate the previous "next" grid/boundary data into the deques. These were
        # set on the previous iteration; loading a new file below does not touch them.
        if input_record.next_grid is not None:
            input_record.grids.append(input_record.next_grid)
            input_record.domain_masks.append(input_record.next_domain_mask)
            input_record.boundary_masks.append(input_record.next_boundary_mask)
            input_record.boundary_coodinates.append(
                input_record.next_boundary_coordinates
            )

        # 2. Load a new file if needed. This updates input_record.dataset and, for
        # file-based datasets, the per-file next_boundary_coordinates.
        boilerplate_update(
            time=time,
            input_record=input_record,
            track_options=track_options,
            dataset_options=dataset_options,
            grid_options=grid_options,
        )

        # 3. Extract this time step's grid and boundary data from the current dataset.
        # The domain/boundary masks live in the dataset as data variables; they are
        # absent for some datasets (e.g. synthetic), in which case they stay None.
        if len(dataset_options.fields) > 1:
            raise ValueError("Only one field allowed for track datasets.")
        field = dataset_options.fields[0]
        dataset = input_record.dataset
        input_record.next_grid = dataset_options.grid_from_dataset(dataset, field, time)
        if "domain_mask" in dataset:
            input_record.next_domain_mask = dataset["domain_mask"]
        if "boundary_mask" in dataset:
            input_record.next_boundary_mask = dataset["boundary_mask"]

        if dataset_options.filepaths is None:
            return
        input_record._time_list.append(time)
        filepath = dataset_options.filepaths[input_record._current_file_index]
        input_record._filepath_list.append(filepath)


def update_tag_input_records(
    time, tag_input_records, track_options, data_options, grid_options
):
    """Update the tag input records."""
    if time is None:
        return
    for name in tag_input_records.keys():
        input_record = tag_input_records[name]
        dataset_options = data_options.dataset_by_name(name)
        boilerplate_update(
            time=time,
            input_record=input_record,
            track_options=track_options,
            dataset_options=dataset_options,
            grid_options=grid_options,
        )
