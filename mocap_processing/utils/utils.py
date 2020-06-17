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


def normalize(v):
    is_list = type(v) == list
    length = np.linalg.norm(v)
    if length > 0.:
        norm_v = np.array(v)/length
        if is_list:
            return list(norm_v)
        else:
            return norm_v
    else:
        return v


def correct_antipodal_quaternions(quat):
    """
    Copied from https://github.com/eth-ait/spl/blob/master/preprocessing/
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
