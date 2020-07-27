import numpy as np
import os
import random


def str_to_axis(s):
    if s == "x":
        return np.array([1.0, 0.0, 0.0])
    elif s == "y":
        return np.array([0.0, 1.0, 0.0])
    elif s == "z":
        return np.array([0.0, 0.0, 1.0])
    else:
        raise Exception


def axis_to_str(a):
    if np.array_equal(a, [1.0, 0.0, 0.0]):
        return "x"
    elif np.array_equal(a, [0.0, 1.0, 0.0]):
        return "y"
    elif np.array_equal(a, [0.0, 0.0, 1.0]):
        return "z"
    else:
        raise Exception


def get_index(index_dict, key):
    if isinstance(key, int):
        return key
    elif isinstance(key, str):
        return index_dict[key]
    else:
        return index_dict[key.name]


def correct_antipodal_quaternions(quat):
    """
    Adapted from https://github.com/eth-ait/spl/blob/master/preprocessing/
    preprocess_dip.py#L64
    Removes discontinuities coming from antipodal representation of quaternions
    At time step t it checks which representation, q or -q, is closer to time
    step t-1 and chooses the closest one.

    Args:
        quat: numpy array of shape (N, K, 4) where N is the number of frames
            and K the number of joints. K is optional, i.e. can be 0.

    Returns: numpy array of shape (N, K, 4) with fixed antipodal representation
    """
    assert len(quat.shape) == 3 or len(quat.shape) == 2
    assert quat.shape[-1] == 4

    if len(quat.shape) == 2:
        quat_r = quat[:, np.newaxis].copy()
    else:
        quat_r = quat.copy()

    def dist(x, y):
        return np.sqrt(np.sum((x - y) ** 2, axis=-1))

    # Naive implementation looping over all time steps sequentially.
    # For a faster implementation check the QuaterNet paper.
    quat_corrected = np.zeros_like(quat_r)
    quat_corrected[0] = quat_r[0]
    for t in range(1, quat.shape[0]):
        diff_to_plus = dist(quat_r[t], quat_corrected[t - 1])
        diff_to_neg = dist(-quat_r[t], quat_corrected[t - 1])

        # diffs are vectors
        qc = quat_r[t]
        swap_idx = np.where(diff_to_neg < diff_to_plus)
        qc[swap_idx] = -quat_r[t, swap_idx]
        quat_corrected[t] = qc
    quat_corrected = np.squeeze(quat_corrected)
    return quat_corrected


def files_in_dir(
    path, ext=None, keyword=None, sort=False, sample_mode=None,
    sample_num=None,
):
    """Returns list of files in `path` directory.

    Args:
        path: Path to directory to list files from
        ext: Extension of files to be listed
        keyword: Return file if filename contains `keyword`
        sort: Sort files by filename in the returned list
        sample_mode: str; Use this option to return subset of files from `path`
            directory. `sample_mode` takes values 'sequential' to return first
            `sample_num` files, or 'shuffle' to return `sample_num` number of
            files randomly
        sample_num: Number of files to return
    """
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            add = True
            if ext is not None and not file.endswith(ext):
                add = False
            if keyword is not None and keyword not in file:
                add = False
            if add:
                files.append(os.path.join(r, file))
    if sort:
        files.sort()

    if sample_num is None:
        sample_num = len(files)
    else:
        sample_num = min(sample_num, len(files))

    if sample_mode is None:
        pass
    elif sample_mode == 'sequential':
        files = files[:sample_num]
    elif sample_mode == 'shuffle':
        files = random.shuffle(files)[:sample_num]
    else:
        raise NotImplementedError

    return files


def _apply_fn_agnostic_to_vec_mat(input, fn):
    output = np.array([input]) if input.ndim == 1 else input
    output = np.apply_along_axis(fn, 1, output)
    return output[0] if input.ndim == 1 else output
