# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import logging
import numpy as np
import os
import pickle

from fairmotion.data import bvh
from fairmotion.ops import conversions
from fairmotion.utils import utils


logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def process_sample(data, rep):
    assert rep in ["aa", "rotmat", "quat"]
    convert_fn = conversions.convert_fn_from_R(rep)
    data = convert_fn(data)
    data = utils.flatten_angles(data, rep)
    return data


def create_samples(
    positive_samples,
    negative_samples,
    rep,
    num_observed,
    skip_frames=1,
    negative_sample_ratio=5,
):
    samples = []
    for sample in positive_samples:
        length = sample.shape[0]
        num_poses_to_extract = num_observed + 1
        indices = np.arange(num_poses_to_extract) * skip_frames
        for i in range(length - num_poses_to_extract * skip_frames):
            samples.append(np.take(sample, i + indices, axis=0))

    data = []
    labels = []
    num_negative_samples = len(negative_samples)
    for sample in samples:
        data.append(sample)
        labels.append(1)
        for _ in range(negative_sample_ratio):
            rand_sample_idx = np.random.randint(num_negative_samples)
            rand_pose_idx = np.random.randint(
                len(negative_samples[rand_sample_idx])
            )
            negative_sample = sample.copy()
            negative_sample[-1] = negative_samples[rand_sample_idx][
                rand_pose_idx
            ]
            data.append(negative_sample)
            labels.append(0)
    return process_sample(np.array(data), rep), np.array(labels)


def read_content(filepath):
    content = []
    with open(filepath) as f:
        for line in f:
            content.append(line.strip())
    return content


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", required=True, help="Where to store pickle files."
    )
    parser.add_argument(
        "--rep",
        type=str,
        default="aa",
        help="Angle representation to convert data to",
        choices=["aa", "quat", "rotmat"],
    )
    parser.add_argument(
        "--num-observed",
        type=int,
        default=5,
        help="Number of observed poses in history",
    )
    parser.add_argument(
        "--frames-between-poses",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--file-list-folder",
        type=str,
    )

    args = parser.parse_args()

    positive_files = read_content(
        os.path.join(os.path.join(args.file_list_folder, "positive.txt"))
    )
    negative_files = read_content(
        os.path.join(os.path.join(args.file_list_folder, "negative.txt"))
    )

    logging.info("Loading files...")
    positive_samples = [
        motion.rotations() for motion in bvh.load_parallel(positive_files)
    ]
    negative_samples = [
        motion.rotations() for motion in bvh.load_parallel(negative_files)
    ]

    output_path = os.path.join(args.output_dir, args.rep)
    utils.create_dir_if_absent(output_path)
    output_path = os.path.join(
        output_path,
        (
            f"data_num_observed_{args.num_observed}_skip_frames_"
            f"{args.frames_between_poses}.pkl"
        )
    )

    logging.info("Creating dataset...")
    dataset = create_samples(
        positive_samples,
        negative_samples,
        rep=args.rep,
        num_observed=args.num_observed,
        skip_frames=args.frames_between_poses,
    )
    pickle.dump(dataset, open(output_path, "wb"))
