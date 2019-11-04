import argparse
import os
from ..pfnn import Quaternions, Animation, BVH


def create_anim_from_indices(anim, start, end):
    if end > len(anim):
        return None
    return Animation.Animation(
        anim.rotations[start: end],
        anim.positions[start: end],
        anim.orients,
        anim.offsets,
        anim.parents,
    )

def split_bvh(filepath, time_window, output_folder):
    anim, joints, time_per_frame = BVH.load(filepath)
    frames_per_time_window = int(time_window/time_per_frame)
    for num, i in enumerate(range (0, len(anim), frames_per_time_window/2)):
        anim_slice = create_anim_from_indices(anim, i, i+frames_per_time_window)
        filepath_slice = os.path.join(output_folder, filepath.split(".")[-2].split("/")[-1] + "_" + str(num) + ".bvh")
        if anim_slice is not None:
            BVH.save(
                filepath_slice,
                anim_slice,
                joints,
                time_per_frame,
            )
                


def main(args):
    for root, _, files in os.walk(args.folder, topdown=False):
        for filename in files:
            filepath = os.path.join(root, filename)
            split_bvh(filepath, args.time_window, args.output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split files in a folder to overlapping n second clips")
    parser.add_argument("--time-window", type=int, help="overlapping time window in seconds")
    parser.add_argument("--folder", type=str)
    parser.add_argument("--output-folder", type=str)
    args = parser.parse_args()
    main(args)
