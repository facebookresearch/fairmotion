# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import logging
import numpy as np
import os
import pickle

from fairmotion.data import amass_dip
from fairmotion.processing import operations
from fairmotion.tasks.motion_prediction import utils
from multiprocessing import Pool


logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def split_into_windows(motion, window_size, stride):
    """
    Split motion object into list of motions with length window_size with
    the given stride.
    """
    n_windows = (motion.num_frames() - window_size) // stride + 1
    motion_ws = [
        operations.cut(motion, start, start + window_size)
        for start in stride * np.arange(n_windows)
    ]
    return motion_ws


def process_file(ftuple, create_windows, convert_fn, lengths):
    src_len, tgt_len = lengths
    root_dir, f, file_id = ftuple
    motion = amass_dip.load(os.path.join(root_dir, f))
    motion.name = file_id

    if create_windows is not None:
        window_size, window_stride = create_windows
        if motion.num_frames() < window_size:
            return [], []
        matrices = [
            convert_fn(motion.rotations()) for motion in
            split_into_windows(motion, window_size, window_stride)
        ]
    else:
        matrices = [convert_fn(motion.rotations())]

    return (
        [matrix[:src_len, ...].reshape((src_len, -1)) for matrix in matrices],
        [
            matrix[src_len:src_len + tgt_len, ...].reshape((tgt_len, -1))
            for matrix in matrices
        ],
    )


def process_split(
    all_fnames, output_path, rep, src_len, tgt_len, create_windows=None,
):
    """
    Process data into numpy arrays.

    Args:
        all_fnames: List of filenames that should be processed.
        output_path: Where to store numpy files.
        rep: If the output data should be rotation matrices, quaternions or
            axis angle.
        create_windows: Tuple (size, stride) of windows that should be
            extracted from each sequence or None otherwise.

    Returns:
        Some meta statistics (how many sequences processed etc.).
    """
    assert rep in ["aa", "rotmat", "quat"]
    convert_fn = utils.convert_fn_from_R(rep)

    pool = Pool(40)
    data = list(pool.starmap(
        process_file,
        [
            (ftuple, create_windows, convert_fn, (src_len, tgt_len))
            for ftuple in all_fnames
        ],
    ))
    src_seqs, tgt_seqs = [], []
    for worker_data in data:
        s, t = worker_data
        src_seqs.extend(s)
        tgt_seqs.extend(t)
    logging.info(f"Processed {len(src_seqs)} sequences")
    pickle.dump((src_seqs, tgt_seqs), open(output_path, "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Location of the downloaded and unpacked zip file.",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Where to store pickle files."
    )
    parser.add_argument(
        "--split-dir",
        default="./data",
        help="Where the text files defining the data splits are stored.",
    )
    parser.add_argument(
        "--rep", type=str, help="Angle representation to convert data to",
        choices=["aa", "quat", "rotmat"]
    )
    parser.add_argument(
        "--src-len",
        type=int,
        default=120,
        help="Number of frames fed as input motion to the model",
    )
    parser.add_argument(
        "--tgt-len",
        type=int,
        default=24,
        help="Number of frames to be predicted by the model",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=180,
        help="Window size for test and val, in frames.",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=120,
        help="Window stride for test and val, in frames. This is also used as "
        "training window size",
    )

    args = parser.parse_args()

    # Load training, validation and test split.
    def _read_fnames(from_):
        with open(from_, "r") as fh:
            lines = fh.readlines()
            return [line.strip() for line in lines]

    train_fnames = _read_fnames(
        os.path.join(args.split_dir, "training_fnames.txt")
    )
    valid_fnames = _read_fnames(
        os.path.join(args.split_dir, "validation_fnames.txt")
    )
    test_fnames = _read_fnames(
        os.path.join(args.split_dir, "test_fnames.txt")
    )

    logging.info(
        f"Pre-determined splits: {len(train_fnames)} train, "
        f"{len(valid_fnames)} valid, {len(test_fnames)} test."
    )

    # Load all available filenames from the source directory.
    train_fnames_avail = []
    test_fnames_avail = []
    valid_fnames_avail = []
    for root_dir, dir_names, file_names in os.walk(args.input_dir):
        dir_names.sort()
        for f in sorted(file_names):
            if not f.endswith(".pkl"):
                continue
            # Extract name of the database.
            db_name = os.path.split(
                os.path.dirname(os.path.join(root_dir, f))
            )[1]
            db_name = (
                "_".join(db_name.split("_")[1:])
                if "AMASS" in db_name
                else db_name.split("_")[0]
            )
            file_id = "{}/{}".format(db_name, f)

            if file_id in train_fnames:
                train_fnames_avail.append((root_dir, f, file_id))
            elif file_id in valid_fnames:
                valid_fnames_avail.append((root_dir, f, file_id))
            elif file_id in test_fnames:
                test_fnames_avail.append((root_dir, f, file_id))
            else:
                # This file was rejected by us because its total sequence
                # length is smaller than 180 (3 seconds)
                pass

    tot_files = (
        len(train_fnames_avail) + len(test_fnames_avail)
        + len(valid_fnames_avail)
    )
    logging.info(f"found {len(train_fnames_avail)} training files")
    logging.info(f"found {len(valid_fnames_avail)} validation files")
    logging.info(f"found {len(test_fnames_avail)} test files")

    output_path = os.path.join(args.output_dir, args.rep)
    utils.create_dir_if_absent(output_path)

    logging.info("Processing training data ...")
    train_dataset = process_split(
        train_fnames_avail,
        os.path.join(output_path, "train.pkl"),
        rep=args.rep,
        src_len=args.src_len,
        tgt_len=args.tgt_len,
        create_windows=(args.window_size, args.window_stride),
    )

    logging.info("Processing validation data ...")
    process_split(
        valid_fnames_avail,
        os.path.join(output_path, "validation.pkl"),
        rep=args.rep,
        src_len=args.src_len,
        tgt_len=args.tgt_len,
        create_windows=(args.window_size, args.window_stride),
    )

    logging.info("Processing test data ...")
    process_split(
        test_fnames_avail,
        os.path.join(output_path, "test.pkl"),
        rep=args.rep,
        src_len=args.src_len,
        tgt_len=args.tgt_len,
        create_windows=(args.window_size, args.window_stride),
    )
