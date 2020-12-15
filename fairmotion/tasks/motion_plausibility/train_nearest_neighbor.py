import argparse
import logging
import numpy as np
import os
from fairmotion.data import bvh
from fairmotion.tasks.motion_plausibility import (
    featurizer as mp_featurizer,
    model,
    preprocess,
)
from fairmotion.utils import utils

logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def compute_mean_score(pose_seqs, nn_model):
    scores = []
    for pose_seq in pose_seqs:
        scores.append(nn_model.score(pose_seq))
    return np.array(scores).mean()


def get_pose_seqs(samples):
    pose_seqs = []
    for motion in samples:
        for pose_idx in range(1, len(motion.poses) - args.num_observed):
            pose_seqs.append(
                motion.poses[pose_idx - 1: pose_idx + args.num_observed]
            )
    return pose_seqs


def main(args):
    utils.create_dir_if_absent(args.save_model_path)
    samples = preprocess.fetch_samples_from_file_lists(
        file_lists=[
            "positive.txt",
            "valid_positive.txt",
            "valid_negative.txt",
        ],
        file_list_folder=args.file_list_folder,
    )
    
    featurizer = mp_featurizer.get_featurizer(args.feature_type)
    nn_model = model.NearestNeighbor(featurizer)
    
    train_pose_seqs = get_pose_seqs(samples[0])
    pos_valid_pose_seqs = get_pose_seqs(samples[1])
    neg_valid_pose_seqs = get_pose_seqs(samples[2])
    
    nn_model.fit(train_pose_seqs)
    
    model_path = os.path.join(args.save_model_path, "best.model")
    nn_model.save(model_path)

    valid_nn_model = model.NearestNeighbor.load(model_path)
    logging.info(
        "Positive validation score: "
        f"{compute_mean_score(pos_valid_pose_seqs, valid_nn_model)}"
    )
    logging.info(
        "Negative validation score: "
        f"{compute_mean_score(neg_valid_pose_seqs, valid_nn_model)}"
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Nearest neighbor motion plausibility model training"
    )
    parser.add_argument(
        "--rep",
        type=str,
        default="aa",
        help="Angle representation to convert data to",
        choices=["aa", "quat", "rotmat"],
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        choices=["facing", "rotation"],
        default="rotation",
    )
    parser.add_argument(
        "--file-list-folder", type=str,
    )
    parser.add_argument(
        "--save-model-path",
        type=str,
        help="Path to store saved models",
        required=True,
    )
    parser.add_argument(
        "--num-observed",
        type=int,
        help="Number of poses in a sequence",
        default=5,
    )
    args = parser.parse_args()
    main(args)