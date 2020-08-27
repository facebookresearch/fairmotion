# Copyright (c) Facebook, Inc. and its affiliates.

"""
Example command:
python fairmotion/tasks/clustering/generate_features.py \
    --type kinetic \
    --folder $PATH_WITH_BVH_FILES \
    --output-folder $PATH_TO_STORE_FEATURES \
    --cpu 10
"""

import argparse
import numpy as np
import os
from multiprocessing import Pool
from fairmotion.tasks.clustering.features import kinetic, manual
from fairmotion.tasks.clustering.features import utils as feat_utils
from fairmotion.data import bvh


thresholds = None


def extract_manual_features(motion):
    features = []
    f = manual.ManualFeatures(motion, feat_utils.Skeleton.PFNN)
    for _ in range(1, motion.num_frames(), 30):
        pose_features = []
        pose_features.append(
            f.f_nmove("neck", "rhip", "lhip", "rwrist", 1.8 * f.hl)
        )
        pose_features.append(
            f.f_nmove("neck", "lhip", "rhip", "lwrist", 1.8 * f.hl)
        )
        pose_features.append(
            f.f_nplane("chest", "neck", "neck", "rwrist", 0.2 * f.hl)
        )
        pose_features.append(
            f.f_nplane("chest", "neck", "neck", "lwrist", 0.2 * f.hl)
        )
        pose_features.append(
            f.f_move("belly", "chest", "chest", "rwrist", 1.8 * f.hl)
        )
        pose_features.append(
            f.f_move("belly", "chest", "chest", "lwrist", 1.8 * f.hl)
        )
        pose_features.append(
            f.f_angle("relbow", "rshoulder", "relbow", "rwrist", [0, 110])
        )
        pose_features.append(
            f.f_angle("lelbow", "lshoulder", "lelbow", "lwrist", [0, 110])
        )
        pose_features.append(
            f.f_nplane(
                "lshoulder", "rshoulder", "lwrist", "rwrist", 2.5 * f.sw
            )
        )
        pose_features.append(
            f.f_move("lwrist", "rwrist", "rwrist", "lwrist", 1.4 * f.hl)
        )
        pose_features.append(
            f.f_move("rwrist", "root", "lwrist", "root", 1.4 * f.hl)
        )
        pose_features.append(
            f.f_move("lwrist", "root", "rwrist", "root", 1.4 * f.hl)
        )
        pose_features.append(f.f_fast("rwrist", 2.5 * f.hl))
        pose_features.append(f.f_fast("lwrist", 2.5 * f.hl))
        pose_features.append(
            f.f_plane("root", "lhip", "ltoes", "rankle", 0.38 * f.hl)
        )
        pose_features.append(
            f.f_plane("root", "rhip", "rtoes", "lankle", 0.38 * f.hl)
        )
        pose_features.append(
            f.f_nplane("zero", "y_unit", "y_min", "rankle", 1.2 * f.hl)
        )
        pose_features.append(
            f.f_nplane("zero", "y_unit", "y_min", "lankle", 1.2 * f.hl)
        )
        pose_features.append(
            f.f_nplane("lhip", "rhip", "lankle", "rankle", 2.1 * f.hw)
        )
        pose_features.append(
            f.f_angle("rknee", "rhip", "rknee", "rankle", [0, 110])
        )
        pose_features.append(
            f.f_angle("lknee", "lhip", "lknee", "lankle", [0, 110])
        )
        pose_features.append(f.f_fast("rankle", 2.5 * f.hl))
        pose_features.append(f.f_fast("lankle", 2.5 * f.hl))
        pose_features.append(
            f.f_angle("neck", "root", "rshoulder", "relbow", [25, 180])
        )
        pose_features.append(
            f.f_angle("neck", "root", "lshoulder", "lelbow", [25, 180])
        )
        pose_features.append(
            f.f_angle("neck", "root", "rhip", "rknee", [50, 180])
        )
        pose_features.append(
            f.f_angle("neck", "root", "lhip", "lknee", [50, 180])
        )
        pose_features.append(
            f.f_plane("rankle", "neck", "lankle", "root", 0.5 * f.hl)
        )
        pose_features.append(
            f.f_angle("neck", "root", "zero", "y_unit", [70, 110])
        )
        pose_features.append(
            f.f_nplane("zero", "minus_y_unit", "y_min", "rwrist", -1.2 * f.hl)
        )
        pose_features.append(
            f.f_nplane("zero", "minus_y_unit", "y_min", "lwrist", -1.2 * f.hl)
        )
        pose_features.append(f.f_fast("root", 2.3 * f.hl))
        features.append(pose_features)
        f.next_frame()
    return features


def extract_kinetic_features(motion, thresholds, up_vec):
    features = kinetic.KineticFeatures(
        motion, 1 / motion.fps, thresholds, up_vec,
    )
    kinetic_feature_vector = []
    for i in range(motion.skel.num_joints()):
        positions_mean, positions_stddev = features.local_position_stats(i)
        feature_vector = np.hstack(
            [
                features.average_kinetic_energy_horizontal(i),
                features.average_kinetic_energy_vertical(i),
                features.average_energy_expenditure(i),
            ]
        )
        kinetic_feature_vector.extend(feature_vector)
    return kinetic_feature_vector


def extract_features(filepath, feature_type, thresholds=None, up_vec="z"):
    motion = bvh.load(filepath)
    if feature_type == "manual":
        return extract_manual_features(motion)
    else:
        return extract_kinetic_features(motion, thresholds, up_vec)


def wrapper_extract_features(inputs):
    global thresholds
    filepath = inputs[0]
    feature_type = inputs[1]
    up_vec = inputs[2]

    features = extract_features(filepath, feature_type, thresholds, up_vec)
    filename = filepath.split("/")[-1]
    with open(
        os.path.join(args.output_folder, "features.tsv"), "a",
    ) as all_features:
        if args.type == "manual":
            all_features.write(
                filename
                + ":"
                + "\t".join(
                    [
                        str(int(datapoint))
                        for pose_features in features
                        for datapoint in pose_features
                    ]
                )
                + "\n"
            )
        else:
            all_features.write(
                filename + ":" + "\t".join([str(f) for f in features]) + "\n"
            )
    print(f"Finished processing {filepath}")


def main(args):
    global thresholds
    thresholds = None
    os.makedirs(args.output_folder, exist_ok=True)
    pool = Pool(args.cpu)
    for root, _, files in os.walk(args.folder, topdown=False):
        pool.map(
            wrapper_extract_features,
            [
                (os.path.join(root, filename), args.type, args.up_vec)
                for filename in files
            ],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract features from BVH files"
    )
    parser.add_argument(
        "--type", nargs="?", choices=["manual", "kinetic"], default="manual",
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="Folder with files to extract features from",
        required=True,
    )
    parser.add_argument("--output-folder", type=str, required=True)
    parser.add_argument("--up-vec", nargs="?", choices=["y", "z"], default="z")
    parser.add_argument("--cpu", type=int, default=40)
    args = parser.parse_args()
    main(args)
