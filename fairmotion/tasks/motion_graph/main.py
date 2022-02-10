# Copyright (c) Facebook, Inc. and its affiliates.

'''
python test_motion_graph.py --v-up-env y --length 30 --num-files 5 --motion-folder XXX --output-bvh-folder YYY
'''

import argparse
import logging

from fairmotion.data import bvh
from fairmotion.core import velocity
from fairmotion.tasks.motion_graph import motion_graph as graph
from fairmotion.utils import utils
from fairmotion.ops import conversions, motion as motion_ops

import os

logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

if __name__ == "__main__":
    parser = parser = argparse.ArgumentParser(
        description="Motion graph construction and exploration"
    )
    parser.add_argument(
        "--motion-files", action="append", default=[],
        help="Motion Files")
    parser.add_argument(
        "--motion-folder", action="append", default=[],
        help="Folder that contains motion files"
    )
    parser.add_argument(
        "--output-bvh-folder",
        type=str,
        required=True,
        help="Resulting motion stored as bvh",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        required=True,
        help="Number of files to generate",
    )
    parser.add_argument(
        "--length",
        type=float,
        required=True,
        help="Number of files to generate",
    )
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--v-up-skel", type=str, default="y")
    parser.add_argument("--v-face-skel", type=str, default="z")
    parser.add_argument("--v-up-env", type=str, default="z")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--base-length", type=float, default=2.0)
    parser.add_argument("--stride-length", type=float, default=0.1)
    parser.add_argument("--blend-length", type=float, default=0.2)
    parser.add_argument("--compare-length", type=float, default=0.2)
    parser.add_argument("--diff-threshold", type=float, default=5.0)
    parser.add_argument("--w-joint-pos", type=float, default=50.0)
    parser.add_argument("--w-joint-vel", type=float, default=0.01)
    parser.add_argument("--w-root-pos", type=float, default=50.0)
    parser.add_argument("--w-root-vel", type=float, default=0.01)
    parser.add_argument("--w-ee-pos", type=float, default=50.0)
    parser.add_argument("--w-ee-vel", type=float, default=0.01)
    parser.add_argument("--w-trajectory", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--num-comparison", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=10)

    args = parser.parse_args()

    # Load motions
    motion_files = args.motion_files
    if len(args.motion_folder) > 0:
        for d in args.motion_folder:
            motion_files += utils.files_in_dir(d, ext="bvh")

    if args.verbose:
        print("-----------Motion Files-----------")
        print(motion_files)
        print("----------------------------------")

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
        motion_ops.resample(motion, args.fps)
        motions_with_velocity.append(
            velocity.MotionWithVelocity.from_motion(motion)
        )

    logging.info(f"Loaded {len(motions_with_velocity)} files")

    ''' 
    Construct Motion Graph
    We assume all motions have the same
        skeleton hierarchy
        fps
    '''
    mg = graph.MotionGraph(
        motions=motions_with_velocity,
        motion_files=motion_files,
        skel=skel,
        fps=args.fps,
        base_length=args.base_length,
        stride_length=args.stride_length,
        compare_length=args.compare_length,
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

    print("Nodes %d, Edges %d"%(mg.graph.number_of_nodes(), mg.graph.number_of_edges()))
    print(list(mg.graph.nodes))

    mg.reduce(method="scc")

    print("Nodes %d, Edges %d"%(mg.graph.number_of_nodes(), mg.graph.number_of_edges()))
    print(list(mg.graph.nodes))

    cnt = 0
    visit_weights = {}
    visit_discount_factor = 0.1
    nodes = list(mg.graph.nodes)
    for n in nodes:
        visit_weights[n] = 1.0
    while cnt < args.num_files:
        m, _ = mg.create_random_motion(
            length=args.length, 
            blend_length=args.blend_length,
            start_node=None, 
            visit_weights=visit_weights,
            visit_discount_factor=visit_discount_factor)
        bvh.save(m, filename=os.path.join(args.output_bvh_folder, "%03d.bvh"%cnt))
        cnt += 1
        print('\r%d/%d completed'%(cnt, args.num_files))
