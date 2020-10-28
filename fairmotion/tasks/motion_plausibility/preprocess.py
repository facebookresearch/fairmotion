# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import logging
import numpy as np
import os
import pickle

from fairmotion.data import bvh
from fairmotion.ops import conversions, math
from fairmotion.tasks.motion_plausibility import (
    featurizer as mp_featurizer,
    options,
)
from fairmotion.utils import utils


logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def select_poses(motions, num_poses, skip_frames):
    selected_poses = []
    for motion in motions:
        length = motion.num_frames()
        num_poses_to_extract = num_poses
        indices = np.arange(num_poses_to_extract) * skip_frames
        for i in range(length - num_poses_to_extract * skip_frames):
            poses_i = []
            for j in i + indices:
                poses_i.append(motion.poses[j])
            selected_poses.append(poses_i)
    return selected_poses


def get_negative_samples_facing(
    poses, negative_motions, skip_frames, negative_sample_ratio=5
):
    num_negative_motions = len(negative_motions)
    negative_samples = []
    for _ in range(negative_sample_ratio):
        rand_sample_idx = np.random.randint(skip_frames, num_negative_motions)
        rand_pose_idx = np.random.randint(
            negative_motions[rand_sample_idx].num_frames()
        )
        rand_pose = negative_motions[rand_sample_idx].poses[rand_pose_idx]
        prev_rand_pose = negative_motions[rand_sample_idx].poses[
            rand_pose_idx - skip_frames
        ]
        rel_root_xform = np.dot(
            math.invertT(prev_rand_pose.get_root_transform()),
            rand_pose.get_root_transform()
        )
        negative_sample = poses.copy()

        new_root_xform = np.dot(poses[-2].get_root_transform(), rel_root_xform)
        rand_pose.set_transform(0, new_root_xform, local=True)

        negative_sample[-1] = rand_pose
        negative_samples.append(negative_sample)
    return negative_samples


def get_negative_samples(poses, negative_motions, negative_sample_ratio=5):
    num_negative_motions = len(negative_motions)
    negative_samples = []
    for _ in range(negative_sample_ratio):
        rand_sample_idx = np.random.randint(1, num_negative_motions)
        rand_pose_idx = np.random.randint(
            negative_motions[rand_sample_idx].num_frames()
        )
        negative_sample = poses.copy()
        negative_sample[-1] = negative_motions[rand_sample_idx].poses[
            rand_pose_idx
        ]
        negative_samples.append(negative_sample)
    return negative_samples


def create_train_samples(
    positive_motions,
    negative_motions,
    rep,
    featurizer,
    num_observed,
    skip_frames=2,
    negative_sample_ratio=5,
):
    selected_positive_poses = select_poses(
        positive_motions,
        num_poses=num_observed + 1,
        skip_frames=skip_frames,
    )

    data = []
    labels = []
    for sample in selected_positive_poses:
        data.append(sample)
        labels.append(1)
        negative_samples = []
        if isinstance(featurizer, mp_featurizer.FacingPositionFeaturizer):
            negative_samples = get_negative_samples_facing(
                sample, negative_motions, skip_frames=skip_frames,
            )
        else:
            negative_samples = get_negative_samples(
                sample, negative_motions
            )
        data.extend(negative_samples)
        labels.extend([0] * len(negative_samples))

    metadata = {
        "num_observed": num_observed,
        "rep": rep,
        "skip_frames": skip_frames,
    }
    pose_vectors = np.array(
        [featurizer.featurize_all(poses) for poses in data]
    )
    return pose_vectors, np.array(labels), metadata


def create_valid_samples(
    positive_motions,
    negative_motions,
    rep,
    featurizer,
    num_observed,
    skip_frames=2,
):
    selected_positive_poses = select_poses(
        positive_motions,
        num_poses=num_observed + 1,
        skip_frames=skip_frames,
    )
    selected_negative_poses = select_poses(
        negative_motions,
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
    pose_vectors = np.array(
        [featurizer.featurize_all(poses) for poses in data]
    )
    return pose_vectors, np.array(labels), metadata


def read_content(filepath):
    content = []
    with open(filepath) as f:
        for line in f:
            content.append(line.strip())
    return content


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    options.add_preprocess_args(parser)
    options.add_preprocess_motion_args(parser)
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
        samples.append(motions)
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
    if args.feature_type == "facing":
        featurizer = mp_featurizer.FacingPositionFeaturizer()
    else:
        featurizer = mp_featurizer.RotationFeaturizer(rep=args.rep)
    logging.info("Creating dataset...")
    train_dataset = create_train_samples(
        samples[0],
        samples[1],
        rep=args.rep,
        num_observed=args.num_observed,
        skip_frames=args.frames_between_poses,
        featurizer=featurizer,
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
        featurizer=featurizer,
    )
    pickle.dump(valid_dataset, open(valid_path, "wb"))
