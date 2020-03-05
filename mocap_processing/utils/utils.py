import numpy as np


def str_to_axis(s):
    if s == "x":
        return np.array([1.0, 0.0, 0.0])
    elif s == "y":
        return np.array([0.0, 1.0, 0.0])
    elif s == "z":
        return np.array([0.0, 0.0, 1.0])
    else:
        raise Exception


def get_index(index_dict, key):
    if isinstance(key, int):
        return key
    elif isinstance(key, str):
        return index_dict[key]
    else:
        return index_dict[key.name]
