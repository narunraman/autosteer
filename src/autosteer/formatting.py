"""Formatting filled values for question text and options: rounding, placeholders, duplicate options."""
import re

import numpy as np


def format_value(value):
    if isinstance(value, float) or isinstance(value, np.float64):
        return round(value, 2)
    elif isinstance(value, str):
        # Round every decimal number in one pass. (The old replace-based
        # version corrupted strings where one number was a substring of
        # another, e.g. '1.5' inside '21.53', and clashed with literal '{0}'.)
        # A number written with at most 2 decimals is left as written: the filler chose that format ("36.00%" next
        # to "30.34%"); rounding it again printed "36.0%", so the computed option was the only one with 2 decimals.
        return re.sub(r'\d+\.\d+', lambda m: m.group(0) if len(m.group(0).split('.')[1]) <= 2
                      else str(round(float(m.group(0)), 2)), value)
    elif isinstance(value, int) or isinstance(value, np.int64):
        return value
    elif isinstance(value, tuple):
        return tuple(format_value(val) for val in value)
    else:
        msg = f"Value must be a tuple, string, float, or int, it is of type: {type(value)}"
        raise ValueError(msg)

def format_fill_values(values):
    """
    Format fill values for filling templates.

    Args:
        values (dict): Dictionary containing fill values.

    Returns:
        dict: Formatted fill values.
    """
    # if there are numbers in the values, ensure they are rounded to two decimals


    for key, value in values.items():
        if isinstance(value, list):
            if isinstance(value[0], list):
                values[key] = [[format_value(lst) for lst in val] for val in value]
            else:
                values[key] = [format_value(val) for val in value]
        else:
            values[key] = format_value(value)

    return values

def extract_fstring_keys(fstring):
    """Return the set of placeholder keys in a format string, ignoring escaped braces."""
    import string
    return {field_name for _, field_name, _, _ in string.Formatter().parse(fstring) if field_name is not None}


def exist_duplicate_options(options_list):
    """
    Check if there are duplicate options in the options list.

    Args:
        options_list (list): List of lists of options.

    Returns:
        bool: True if there are duplicate options, False otherwise.
    """
    all_options = [format_value(option) for options in options_list for option in options]
    return len(all_options) != len(set(all_options))
