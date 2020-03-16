import numpy as np
from mocap_processing.utils import constants
from mocap_processing.utils import conversions


def get_random_Q():
    Q = np.random.rand(4)
    Q = Q / np.linalg.norm(Q)
    return Q


def get_random_R():
    Q = get_random_Q()
    return conversions.Q2R(Q)


def get_random_T():
    R = get_random_R()
    p = np.random.rand(3)
    T = constants.eye_T
    T[:3, :3] = R
    T[:3, 3] = p
    return T
