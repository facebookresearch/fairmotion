import argparse
import numpy as np
import os
import pickle

from mocap_processing.data import amass_dip
from mocap_processing.processing import operations
from mocap_processing.utils import conversions


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


def process_split(
    all_fnames, output_path, compute_stats, rep, create_windows=None,
):
    """
    Process data into numpy arrays.

    Args:
        all_fnames: List of filenames that should be processed.
        output_path: Where to store numpy files.
        compute_stats: Whether to compute and store normalization statistics.
        rep: If the output data should be rotation matrices, quaternions or
            axis angle.
        create_windows: Tuple (size, stride) of windows that should be
            extracted from each sequence or None otherwise.

    Returns:
        Some meta statistics (how many sequences processed etc.).
    """
    assert rep in ["aa", "rotmat", "quat"]
    if rep == "aa":
        convert_fn = conversions.R2A
    elif rep == "quat":
        convert_fn = conversions.R2Q
    elif rep == "rotmat":
        convert_fn = lambda x: x

    print(
        "storing into {} computing stats {}".format(
            output_path, "YES" if compute_stats else "NO"
        )
    )

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    src = []
    tgt = []
    for idx in range(len(all_fnames)):
        root_dir, f, file_id = all_fnames[idx]
        motion = amass_dip.load(os.path.join(root_dir, f))
        motion.name = file_id
        # db_name = file_id.split("/")[0]

        if create_windows is not None:
            window_size, window_stride = create_windows
            if motion.num_frames() < window_size:
                continue
            matrices = [
                convert_fn(motion.to_matrix()) for motion in
                split_into_windows(motion, window_size, window_stride)
            ]
        else:
            matrices = [convert_fn(motion.to_matrix())]
        src.extend(
            [matrix[window_stride, ...] for matrix in matrices]
        )
        tgt.extend(
            [
                motion.to_matrix()[window_size - window_stride, ...]
                for matrix in matrices
            ]
        )
    pickle.dump((src, tgt), open(output_path, "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Location of the downloaded and unpacked zip file.",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Where to store the tfrecords."
    )
    parser.add_argument(
        "--split_dir",
        default="./data",
        help="Where the text files defining the data splits are stored.",
    )
    parser.add_argument(
        "--as_quat", action="store_true", help="Whether to convert data to "
        "quaternions."
    )
    parser.add_argument(
        "--as_aa", action="store_true", help="Whether to convert data to "
        "angle-axis."
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=180,
        help="Window size for test and val, in frames.",
    )
    parser.add_argument(
        "--window_stride",
        type=int,
        default=120,
        help="Window stride for test and val, in frames. This is also used as "
        "training window size",
    )

    args = parser.parse_args()

    assert not (
        args.as_quat and args.as_aa
    ), "must choose between quaternion or angle-axis representation"

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

    print(
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
    print(f"found {len(train_fnames_avail)} training files")
    print(f"found {len(valid_fnames_avail)} validation files")
    print(f"found {len(test_fnames_avail)} test files")

    print("process training data ...")
    rep = "quat" if args.as_quat else "aa" if args.as_aa else "rotmat"

    train_dataset = process_split(
        train_fnames_avail,
        os.path.join(args.output_dir, rep, "training"),
        compute_stats=True,
        rep=rep,
        create_windows=None
    )

    print("Processing validation data ...")
    process_split(
        valid_fnames_avail,
        os.path.join(args.output_dir, rep, "validation"),
        compute_stats=False,
        rep=rep,
        create_windows=(args.window_size, args.window_stride),
    )

    print("Processing test data ...")
    process_split(
        test_fnames_avail,
        os.path.join(args.output_dir, rep, "test"),
        compute_stats=False,
        rep=rep,
        create_windows=(args.window_size, args.window_stride),
    )
