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


def select_poses(data, num_poses, skip_frames):
    selected_data = []
    for sample in data:
        length = sample.shape[0]
        num_poses_to_extract = num_poses
        indices = np.arange(num_poses_to_extract) * skip_frames
        for i in range(length - num_poses_to_extract * skip_frames):
            selected_data.append(np.take(sample, i + indices, axis=0))
    return selected_data


def create_train_samples(
    positive_samples,
    negative_samples,
    rep,
    num_observed,
    skip_frames=2,
    negative_sample_ratio=5,
):
    selected_positive_poses = select_poses(
        positive_samples,
        num_poses=num_observed + 1,
        skip_frames=skip_frames,
    )

    data = []
    labels = []
    num_negative_samples = len(negative_samples)
    for sample in selected_positive_poses:
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

    metadata = {
        "num_observed": num_observed,
        "rep": rep,
        "skip_frames": skip_frames,
    }
    return process_sample(np.array(data), rep), np.array(labels), metadata


def create_valid_samples(
    positive_samples,
    negative_samples,
    rep,
    num_observed,
    skip_frames=2,
):
    selected_positive_poses = select_poses(
        positive_samples,
        num_poses=num_observed + 1,
        skip_frames=skip_frames,
    )
    selected_negative_poses = select_poses(
        negative_samples,
        num_poses=num_observed + 1,
        skip_frames=skip_frames,
    )

    data = selected_positive_poses + selected_negative_poses
    labels = (
        [1] * len(selected_positive_poses) +
        [0] * len(selected_negative_poses)
    )

    metadata = {
        "num_observed": num_observed,
        "rep": rep,
        "skip_frames": skip_frames,
    }
    return process_sample(np.array(data), rep), np.array(labels), metadata


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

    logging.info("Loading files...")
    samples = []
    for file_list_name in [
        "positive.txt",
        "negative.txt",
        "valid_positive.txt",
        "valid_negative.txt"
    ]:
        files = read_content(
            os.path.join(os.path.join(args.file_list_folder, file_list_name))
        )
        motions = bvh.load_parallel(files)
        samples.append(
            [
                motion.rotations() for motion in motions
            ]
        )
        num_frames = sum([motion.num_frames() for motion in motions])
        logging.info(f"{file_list_name}: {num_frames} frames")

    output_path = os.path.join(args.output_dir, args.rep)
    utils.create_dir_if_absent(output_path)
    train_path = os.path.join(
        output_path,
        (
            f"train_num_observed_{args.num_observed}_skip_frames_"
            f"{args.frames_between_poses}.pkl"
        )
    )

    logging.info("Creating dataset...")
    train_dataset = create_train_samples(
        samples[0],
        samples[1],
        rep=args.rep,
        num_observed=args.num_observed,
        skip_frames=args.frames_between_poses,
    )
    pickle.dump(train_dataset, open(train_path, "wb"))

    valid_path = os.path.join(
        output_path,
        (
            f"valid_num_observed_{args.num_observed}_skip_frames_"
            f"{args.frames_between_poses}.pkl"
        )
    )
    valid_dataset = create_valid_samples(
        samples[2],
        samples[3],
        rep=args.rep,
        num_observed=args.num_observed,
        skip_frames=args.frames_between_poses,
    )
    pickle.dump(valid_dataset, open(valid_path, "wb"))
