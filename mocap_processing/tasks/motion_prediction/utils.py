import numpy as np
import os
from functools import partial
from multiprocessing import Pool

from mocap_processing.models import decoders, encoders, seq2seq
from mocap_processing.tasks.motion_prediction import dataset as motion_dataset
from mocap_processing.utils import constants, conversions


def apply_ops(input, ops):
    output = input
    for op in ops:
        output = op(output)
    return output


def unflatten_angles(arr, rep):
    """
    Unflatten from (batch_size, num_frames, num_joints*ndim) to
    (batch_size, num_frames, num_joints, ndim) for each angle format
    """
    if rep == "aa":
        return arr.reshape(arr.shape[:-1] + (-1, 3))
    elif rep == "quat":
        return arr.reshape(arr.shape[:-1] + (-1, 4))
    elif rep == "rotmat":
        return arr.reshape(arr.shape[:-1] + (-1, 3, 3))


def flatten_angles(arr, rep):
    """
    Unflatten from (batch_size, num_frames, num_joints, ndim) to
    (batch_size, num_frames, num_joints*ndim) for each angle format
    """
    if rep == "aa":
        return arr.reshape(arr.shape[:-2] + (-1))
    elif rep == "quat":
        return arr.reshape(arr.shape[:-2] + (-1))
    elif rep == "rotmat":
        # original dimension is (batch_size, num_frames, num_joints, 3, 3)
        return arr.reshape(arr.shape[:-3] + (-1))


def multiprocess_convert(arr, convert_fn):
    pool = Pool(40)
    result = list(pool.map(convert_fn, arr))
    return result


def convert_fn_to_R(rep):
    ops = [partial(unflatten_angles, rep=rep)]
    if rep == "aa":
        ops.append(partial(multiprocess_convert, convert_fn=conversions.A2R))
    elif rep == "quat":
        ops.append(partial(multiprocess_convert, convert_fn=conversions.Q2R))
    elif rep == "rotmat":
        ops.append(lambda x: x)
    ops.append(np.array)
    return ops


def identity(x):
    return x


def convert_fn_from_R(rep):
    if rep == "aa":
        convert_fn = conversions.R2A
    elif rep == "quat":
        convert_fn = conversions.R2Q
    elif rep == "rotmat":
        convert_fn = identity
    return convert_fn


def unnormalize(arr, mean, std):
    return arr * (std + constants.EPSILON) + mean


def create_dir_if_absent(path):
    if not os.path.exists(path):
        os.makedirs(path)


def prepare_dataset(train_path, valid_path, test_path, batch_size, device):
    dataset = {}
    for split, split_path in zip(
        ["train", "test", "validation"],
        [train_path, valid_path, test_path]
    ):
        mean, std = None, None
        if split in ["test", "validation"]:
            mean = dataset["train"].dataset.mean
            std = dataset["train"].dataset.std
        dataset[split] = motion_dataset.get_loader(
            split_path, batch_size, device, mean, std,
        )
    return dataset, mean, std


def prepare_model(input_dim, hidden_dim, device, architecture="seq2seq"):
    if architecture == "seq2seq":
        enc = encoders.LSTMEncoder(
            input_dim=input_dim, hidden_dim=hidden_dim
        ).to(device)
        dec = decoders.LSTMDecoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            device=device,
        ).to(device)
        model = seq2seq.Seq2Seq(enc, dec)
    elif architecture == "tied_seq2seq":
        model = seq2seq.TiedSeq2Seq(input_dim, hidden_dim, 1, device)
    elif architecture == "transformer_encoder":
        model = seq2seq.TransformerModel(input_dim, hidden_dim, 4, hidden_dim, 4)
    elif architecture == "transformer":
        model = seq2seq.FullTransformerModel(
            input_dim, hidden_dim, 4, hidden_dim, 4,
        )
    model = model.to(device)
    model.zero_grad()
    model.double()
    return model
