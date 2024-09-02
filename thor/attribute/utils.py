"""General utilities for object attributes."""


def initialize_attributes(object_tracks, object_options):
    object_tracks["attribute"] = {}
    for key in object_options["attribute"].keys():
        object_tracks["attribute"][key] = {
            attr: [] for attr in object_options["attribute"][key].keys()
        }
