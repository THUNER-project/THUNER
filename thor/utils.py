"General utilities for the thor package."


def format_string_list(strings):
    """
    Format a list of strings into a human-readable string.

    Parameters
    ----------
    strings : list of str
        List of strings to be formatted.

    Returns
    -------
    formatted_string : str
        The formatted string.
    """
    if len(strings) > 1:
        formatted_string = ", ".join(strings[:-1]) + " or " + strings[-1]
        return formatted_string
    elif strings:
        return strings[0]
    else:
        raise ValueError("strings must be an iterable of strings'.")


def drop_time(time):
    """
    Drop the time component of a datetime64 object.

    Parameters
    ----------
    time : np.datetime64
        Datetime object.

    Returns
    -------
    date : np.datetime64
        Date object.
    """
    return time.astype("datetime64[D]").astype("datetime64[s]")
