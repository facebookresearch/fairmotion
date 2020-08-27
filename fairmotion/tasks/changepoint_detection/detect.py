# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from fairmotion.data import bvh


def plot_changepoints(acc_norm, segs, plot_path):
    plt.plot(range(len(acc_norm)), acc_norm)
    plt.plot(segs, acc_norm[segs], "o")
    plt.ylabel("Norm of joint acceleration")
    plt.xlabel("Frame number")
    plt.savefig(plot_path, format="svg")
    plt.show()


def main(args):
    motion = bvh.load(args.input_file)
    trajectory = motion.positions(local=False).reshape(motion.num_frames(), -1)
    print("Total length (in s) ", motion.length())

    # Assumes you have a trajectory that has axis 0 as time and axis 1 as
    # state dimensions.
    acc = np.diff(trajectory, n=2, axis=0)
    acc_norm = np.linalg.norm(acc, axis=1)
    segs = find_peaks(acc_norm, height=0.075)[0]
    # sega = argrelextrema(acc_norm, np.greater, order=8)[0]

    print("Number of detected segments ", len(segs))

    if args.output_plot is not None:
        plot_changepoints(acc_norm, segs, args.output_plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Acceleration based changepoint detection"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="BVH file for which we find changepoints",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        help="File to store 2D plot with changepoints",
        default=None,
    )
    args = parser.parse_args()
    main(args)
