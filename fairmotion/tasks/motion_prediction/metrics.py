# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
from fairmotion.ops import conversions


def euler_diff(predictions, targets):
    """
    Computes the Euler angle error as in previous work, following
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/translate.py#L207
    Args:
        predictions: np array of predicted joint angles represented as rotation matrices, i.e. in shape
          (..., n_joints, 3, 3)
        targets: np array of same shape as `predictions`

    Returns:
        The Euler angle error an np array of shape (..., )
    """
    assert predictions.shape[-1] == 3 and predictions.shape[-2] == 3
    assert targets.shape[-1] == 3 and targets.shape[-2] == 3
    n_joints = predictions.shape[-3]

    ori_shape = predictions.shape[:-3]
    preds = np.reshape(predictions, [-1, 3, 3])
    targs = np.reshape(targets, [-1, 3, 3])

    euler_preds = conversions.R2E(preds)  # (N, 3)
    euler_targs = conversions.R2E(targs)  # (N, 3)

    # reshape to (-1, n_joints*3) to be consistent with previous work
    euler_preds = np.reshape(euler_preds, [-1, n_joints*3])
    euler_targs = np.reshape(euler_targs, [-1, n_joints*3])

    # l2 error on euler angles
    idx_to_use = np.where(np.std(euler_targs, 0) > 1e-4)[0]
    euc_error = np.power(
        euler_targs[:, idx_to_use] - euler_preds[:, idx_to_use],
        2,
    )
    euc_error = np.sqrt(np.sum(euc_error, axis=1))  # (-1, ...)

    # reshape to original
    return np.reshape(euc_error, ori_shape)