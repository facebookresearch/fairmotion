# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import logging

from fairmotion.data import bvh
from fairmotion.core import velocity
from fairmotion.tasks.motion_graph import motion_graph as graph
from fairmotion.utils import utils


logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


if __name__ == "__main__":
    parser = parser = argparse.ArgumentParser(
        description="Motion graph construction and exploration"
    )
    parser.add_argument("--motion-files", action="append", help="Motion Files")
    parser.add_argument(
        "--motion-folder", help="Folder that contains motion files"
    )
    parser.add_argument(
        "--output-bvh",
        type=str,
        required=True,
        help="Resulting motion stored as bvh",
    )
    parser.add_argument("--v-up-skel", type=str, default="y")
    parser.add_argument("--v-face-skel", type=str, default="z")
    parser.add_argument("--v-up-env", type=str, default="z")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--base-length", type=float, default=1.5)
    parser.add_argument("--blend-length", type=float, default=0.5)
    parser.add_argument("--diff-threshold", type=float, default=5.0)
    parser.add_argument("--w-joint-pos", type=float, default=8.0)
    parser.add_argument("--w-joint-vel", type=float, default=2.0)
    parser.add_argument("--w-root-pos", type=float, default=0.5)
    parser.add_argument("--w-root-vel", type=float, default=1.5)
    parser.add_argument("--w-ee-pos", type=float, default=3.0)
    parser.add_argument("--w-ee-vel", type=float, default=1.0)
    parser.add_argument("--w-trajectory", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=10)

    args = parser.parse_args()

    # Load motions
    motion_files = args.motion_files if args.motion_files else []
    motion_files = (
        motion_files + utils.files_in_dir(args.motion_folder, ext="bvh")
        if args.motion_folder
        else motion_files
    )

    motions = bvh.load_parallel(
        motion_files,
        scale=args.scale,
        v_up_skel=utils.str_to_axis(args.v_up_skel),
        v_face_skel=utils.str_to_axis(args.v_face_skel),
        v_up_env=utils.str_to_axis(args.v_up_env),
    )

    skel = motions[0].skel
    motions_with_velocity = []
    for motion in motions:
        motion.set_skeleton(skel)
        motions_with_velocity.append(
            velocity.MotionWithVelocity.from_motion(motion)
        )

    logging.info(f"Loaded {len(motions_with_velocity)} files")

    # Construct Motion Graph
    mg = graph.MotionGraph(
        motions=motions_with_velocity,
        motion_files=motion_files,
        skel=skel,
        base_length=args.base_length,
        blend_length=args.blend_length,
        verbose=True,
    )
    mg.construct(
        w_joints=None,
        w_joint_pos=args.w_joint_pos,
        w_joint_vel=args.w_joint_vel,
        w_root_pos=args.w_root_pos,
        w_root_vel=args.w_root_vel,
        w_ee_pos=args.w_ee_pos,
        w_ee_vel=args.w_ee_vel,
        w_trajectory=args.w_trajectory,
        diff_threshold=args.diff_threshold,
        num_workers=args.num_workers,
    )
    mg.reduce(method="scc")

    # del motions[:]

    motion, _ = mg.create_random_motion(length=10.0, start_node=0)
    bvh.save(motion, filename=args.output_bvh)
