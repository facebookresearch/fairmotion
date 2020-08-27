# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import os
import tqdm
from fairmotion.data import bvh
from fairmotion.ops import motion as motion_ops


def split_bvh(filepath, time_window, output_folder):
    motion = bvh.load(filepath)
    frames_per_time_window = time_window * motion.fps
    for num, i in enumerate(
        range(0, motion.num_frames(), int(frames_per_time_window / 2))
    ):
        motion_slice = motion_ops.cut(motion, i, i + frames_per_time_window)
        filepath_slice = os.path.join(
            output_folder,
            filepath.split(".")[-2].split("/")[-1] + "_" + str(num) + ".bvh",
        )
        bvh.save(motion_slice, filepath_slice)


def main(args):
    os.makedirs(args.output_folder, exist_ok=True)
    for root, _, files in os.walk(args.folder, topdown=False):
        for filename in tqdm.tqdm(files):
            filepath = os.path.join(root, filename)
            split_bvh(filepath, args.time_window, args.output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split files in a folder to overlapping n second clips"
    )
    parser.add_argument(
        "--time-window", type=int, help="overlapping time window in seconds"
    )
    parser.add_argument("--folder", type=str)
    parser.add_argument("--output-folder", type=str)
    args = parser.parse_args()
    main(args)
