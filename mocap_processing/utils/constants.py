import numpy as np


EPS = np.finfo(float).eps

eye_R = np.array([
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0., 1.]], float
)

eye_T = np.array([
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 1., 0.],
    [0., 0., 0., 1.]], float
)

zero_p = np.array([0., 0., 0.], float)

zero_R = np.zeros((3, 3))
