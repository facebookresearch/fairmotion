import numpy as np

from fairmotion.ops import conversions
from fairmotion.utils import constants


def get_random_Q(shape=()):
    Q = np.random.rand(*(shape + (4,)))
    Q = Q / np.linalg.norm(Q)[..., np.newaxis]
    return Q


def get_random_R(shape):
    Q = get_random_Q(shape)
    return conversions.Q2R(Q)


def get_random_T(shape):
    R = get_random_R(shape)
    p = np.random.rand(*(shape + (3,)))
    T = np.zeros(shape + (4, 4))
    T[...] = constants.eye_T()
    T[..., :3, :3] = R
    T[..., :3, 3] = p
    return T
