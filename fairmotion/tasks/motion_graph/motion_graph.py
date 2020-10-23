# Copyright (c) Facebook, Inc. and its affiliates.

import gzip
import logging
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pickle
import random
import tqdm

from fairmotion.data import bvh
from fairmotion.core import similarity, velocity
from fairmotion.ops import conversions, motion as motion_ops
from fairmotion.utils import utils


logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def create_nodes(
    motion_idx, motions, base_length, stride_length, compare_length, fps
):
    """
    Creates nodes for the graph by slicing motion sequences based on length
    parameters. A node will have a motion with (base_length+compare_length)
    sec long. Nodes will be created by following a given motion in a sliding
    window fashion. The sliding window will have (stride_lengh) sec overlap.
    """
    res = []
    frames_base = int(base_length * fps)
    frames_compare = int(compare_length * fps)
    frames_stride = int(stride_length * fps)

    # This ensures that the original motion is always connected
    while frames_base % frames_stride != 0 and frames_stride > 1:
        frames_stride -= 1

    motion = motions[motion_idx]
    # We ignore files that are shorter than default length
    if motion.length() < base_length + compare_length:
        return res

    frames_start = np.arange(
        0,
        motion.num_frames() - frames_base - frames_compare - 1,
        frames_stride,
    )
    frames_end = frames_start + frames_base
    motion_idx_array = np.full(shape=frames_start.shape, fill_value=motion_idx)
    res = np.stack([motion_idx_array, frames_start, frames_end]).transpose()
    return res


def compare_and_connect_edge(
    node_id,
    nodes,
    motions,
    frames_compare,
    w_joints,
    w_joint_pos,
    w_joint_vel,
    w_root_pos,
    w_root_vel,
    w_ee_pos,
    w_ee_vel,
    w_trajectory,
    diff_threshold,
    num_comparison,
    verbose,
):
    node = nodes[node_id]
    res = []
    num_nodes = len(nodes)

    motion_idx = node["motion_idx"]
    frame_end = node["frame_end"]

    for j in range(num_nodes):
        motion_idx_j = nodes[j]["motion_idx"]
        frame_start_j = nodes[j]["frame_start"]
        diff_pose = 0.0
        diff_root_ee = 0.0
        diff_trajectory = 0.0

        for k in range(
            0, frames_compare + 1, (frames_compare + 1) // num_comparison
        ):
            pose = motions[motion_idx].get_pose_by_frame(frame_end + k)
            vel = motions[motion_idx].get_velocity_by_frame(frame_end + k)
            pose_j = motions[motion_idx_j].get_pose_by_frame(frame_start_j + k)
            vel_j = motions[motion_idx_j].get_velocity_by_frame(
                frame_start_j + k
            )
            if k == 0:
                T_ref = pose.get_facing_transform()
                T_ref_j = pose_j.get_facing_transform()
            diff_pose += similarity.pose_similarity(
                pose, pose_j, vel, vel_j, w_joint_pos, w_joint_vel, w_joints
            )
            diff_root_ee += similarity.root_ee_similarity(
                pose,
                pose_j,
                vel,
                vel_j,
                w_root_pos,
                w_root_vel,
                w_ee_pos,
                w_ee_vel,
                T_ref,
                T_ref_j,
            )
            if w_trajectory > 0.0:
                R, p = conversions.T2Rp(pose.get_facing_transform())
                R_j, p_j = conversions.T2Rp(pose_j.get_facing_transform())
                if k > 0:
                    d = np.dot(R_prev.transpose(), p - p_prev)
                    d_j = np.dot(R_j_prev.transpose(), p_j - p_j_prev)
                    d = d - d_j
                    diff_trajectory += np.dot(d, d)
                R_prev, p_prev = R, p
                R_j_prev, p_j_prev = R_j, p_j
        diff_pose /= num_comparison
        diff_root_ee /= num_comparison
        diff_trajectory /= num_comparison
        diff = diff_pose + diff_root_ee + diff_trajectory

        if diff <= diff_threshold:
            res.append((diff, node_id, j))
    return res


def flatten(l):
    return [elem for sublist in l for elem in sublist]


class MotionGraph(object):
    def __init__(
        self,
        motions,
        motion_files,
        skel,
        fps=30,
        base_length=1.5,
        stride_length=1.5,
        blend_length=0.5,
        compare_length=1.0,
        scale=1.0,
        verbose=False,
    ):
        self.graph = nx.DiGraph()
        self.motions = motions
        self.motion_files = motion_files
        self.skel = skel
        self.fps = fps
        self.base_length = base_length
        self.stride_length = stride_length
        self.blend_length = blend_length
        self.compare_length = compare_length
        self.frames_blend = int(self.blend_length * self.fps)
        self.frames_compare = int(self.compare_length * self.fps)
        self.scale = scale
        self.verbose = verbose
        assert len(self.motions) > 0
        assert base_length >= stride_length
        assert base_length > blend_length
        assert compare_length >= blend_length
        assert fps > 0

    def construct(
        self,
        w_joints=None,
        w_joint_pos=0.4,
        w_joint_vel=0.1,
        w_root_pos=0.4,
        w_root_vel=0.6,
        w_ee_pos=0.4,
        w_ee_vel=0.6,
        w_trajectory=0.5,
        diff_threshold=1.0,
        num_comparison=3,
        num_workers=1,
    ):
        assert len(self.motions) > 0, "No motions to construct graph"
        if self.verbose:
            logging.info("Starting construction")
        self.skel = self.motions[0].skel
        for m in self.motions:
            m.set_skeleton(self.skel)
        if self.verbose:
            logging.info("Creating nodes")
        # Create nodes
        ns = utils.run_parallel(
            create_nodes,
            list(range(len(self.motions))),
            motions=self.motions,
            num_cpus=num_workers,
            base_length=self.base_length,
            stride_length=self.stride_length,
            compare_length=self.compare_length,
            fps=self.fps,
        )
        ns = flatten(ns)
        if self.verbose:
            logging.info(f"Merging {len(ns)} nodes...")
        for motion_idx, frame_start, frame_end in tqdm.tqdm(ns):
            # for motion_idx, frame_start, frame_end in ns:
            self.graph.add_node(
                self.graph.number_of_nodes(),
                motion_idx=motion_idx,
                frame_start=frame_start,
                frame_end=frame_end,
            )
        if self.verbose:
            logging.info("Creating edges...")

        wes = utils.run_parallel(
            compare_and_connect_edge,
            list(range(self.graph.number_of_nodes())),
            nodes=self.graph.nodes,
            motions=self.motions,
            num_cpus=num_workers,
            frames_compare=self.frames_compare,
            w_joints=w_joints,
            w_joint_pos=w_joint_pos,
            w_joint_vel=w_joint_vel,
            w_root_pos=w_root_pos,
            w_root_vel=w_root_vel,
            w_ee_pos=w_ee_pos,
            w_ee_vel=w_ee_vel,
            w_trajectory=w_trajectory,
            diff_threshold=diff_threshold,
            num_comparison=num_comparison,
            verbose=self.verbose,
        )
        wes = flatten(wes)

        self.w_joints = w_joints
        self.w_joint_pos = w_joint_pos
        self.w_joint_vel = w_joint_vel
        self.w_root_pos = w_root_pos
        self.w_root_vel = w_root_vel
        self.w_ee_pos = w_ee_pos
        self.w_ee_vel = w_ee_vel
        self.w_trajectory = w_trajectory
        if self.verbose:
            logging.info(f"Merging {len(wes)} edges...")
        for w, e_i, e_j in tqdm.tqdm(wes):
            self.graph.add_edge(e_i, e_j, weights=w)
        if self.verbose:
            logging.info("MotionGraph was constructed")
            logging.info(f"NumNodes: {self.graph.number_of_nodes()}")
            logging.info(f"NumEdges: {self.graph.number_of_edges()}")

    def clear_visit_info(self):
        for n in self.graph.nodes:
            self.graph.nodes[n]["num_visit"] = 0
        for e in self.graph.edges:
            self.graph.edges[e]["num_visit"] = 0

    def create_motion_by_following(self, nodes):
        motion = velocity.MotionWithVelocity(skel=self.skel, fps=self.fps)
        for i in range(len(nodes) - 1):
            n1 = nodes[i]
            n2 = nodes[i + 1]
            nodes_inbetween = nx.shortest_path(self.graph, n1, n2)
            range_nodes = (
                range(len(nodes_inbetween))
                if i == len(nodes) - 2
                else range(len(nodes_inbetween) - 1)
            )
            for j in range_nodes:
                n = nodes_inbetween[j]
                motion_idx = self.graph.nodes[n]["motion_idx"]
                frame_start = self.graph.nodes[n]["frame_start"]
                frame_end = self.graph.nodes[n]["frame_end"]
                if self.motions[motion_idx] is None:
                    self.load_motion_at_idx(
                        motion_idx, self.motion_files[motion_idx]
                    )
                m = self.motions[motion_idx].detach(
                    frame_start, frame_end + self.frames_blend
                )
                motion = motion_ops.append_and_blend(
                    motion, m, blend_length=self.blend_length,
                )
        return motion

    def create_random_path(
        self,
        length,
        start_node=None,
        leave_visit_info=True,
        use_visit_info="edge",
    ):
        """
        This funtion generate a sequence of randomly visited nodes on the graph

        length - length of the generated motion
        start_node - we can specify a start node if necessary
        """

        t_processed = 0.0
        nodes = list(self.graph.nodes)
        if start_node is None or start_node not in self.graph.nodes:
            cur_node = random.choice(nodes)
        else:
            cur_node = start_node
        prev_node = cur_node
        visited_nodes = []
        while t_processed < length:
            # Record currently visiting node
            visited_nodes.append(cur_node)

            if leave_visit_info:
                self.graph.nodes[cur_node]["num_visit"] += 1
                if t_processed > 0.0:
                    self.graph.edges[(prev_node, cur_node)]["num_visit"] += 1

            # Append the selected motion to the current motion
            frame_start = self.graph.nodes[cur_node]["frame_start"]
            frame_end = self.graph.nodes[cur_node]["frame_end"]

            if self.verbose:
                logging.info(f"[{cur_node}] {self.graph.nodes[cur_node]}")

            t_processed += (frame_end - frame_start + 1.0) / self.fps
            # Jump to adjacent node (motion) randomly
            if self.graph.out_degree(cur_node) == 0:
                if self.verbose:
                    logging.info("Dead-end exists in the graph!")
                break
            prev_node = cur_node
            if use_visit_info == "node" or use_visit_info == "edge":
                successors = list(self.graph.successors(cur_node))
                if use_visit_info == "node":
                    num_visit = np.array(
                        [
                            self.graph.nodes[next_node]["num_visit"] + 0.001
                            for next_node in successors
                        ]
                    )
                else:
                    num_visit = np.array(
                        [
                            self.graph.edges[(cur_node, next_node)][
                                "num_visit"
                            ]
                            + 0.001
                            for next_node in successors
                        ]
                    )
                probability = 1.0 / num_visit
                probability = probability / np.sum(probability)
                cur_node = np.random.choice(successors, p=probability)
            else:
                cur_node = random.choice(list(self.graph.successors(cur_node)))
        return visited_nodes, t_processed

    def create_random_motion(self, length, start_node=None):
        """
        This funtion generate a motion from the given motion graph by randomly
        traversing the graph.

        length - length of the generated motion
        start_node - we can specify a start node if necessary
        """
        motion = velocity.MotionWithVelocity(skel=self.skel, fps=self.fps)
        t_processed = 0.0
        nodes = list(self.graph.nodes)
        if start_node is None or start_node not in self.graph.nodes:
            cur_node = random.choice(nodes)
        else:
            cur_node = start_node
        visited_nodes = []
        while t_processed < length:
            # Record currently visiting node
            visited_nodes.append(cur_node)

            # Append the selected motion to the current motion
            motion_idx = self.graph.nodes[cur_node]["motion_idx"]
            frame_start = self.graph.nodes[cur_node]["frame_start"]
            frame_end = self.graph.nodes[cur_node]["frame_end"]

            # Load the motion if it is not loaded in advance
            if self.motions[motion_idx] is None:
                self.load_motion_at_idx(
                    motion_idx, self.motion_files[motion_idx]
                )

            """
            We should detach with the extra (frames_blend)
            because motion.append affects the end of current motion
            """
            m = motion_ops.cut(
                self.motions[motion_idx],
                frame_start,
                frame_end + self.frames_blend,
            )

            if self.verbose:
                logging.info(f"[{cur_node}] {self.graph.nodes[cur_node]}")

            motion = motion_ops.append_and_blend(
                motion, m, blend_length=self.blend_length,
            )

            t_processed = motion.length()
            # Jump to adjacent node (motion) randomly
            if self.graph.out_degree(cur_node) == 0:
                if self.verbose:
                    logging.info("Dead-end exists in the graph!")
                break
            cur_node = random.choice(list(self.graph.successors(cur_node)))
        return motion, visited_nodes

    def reduce(self, method="scc", num_component=1):
        """
        scc : strongly connected component
        wcc : weakly connected component
        """
        if method == "scc":
            components = nx.strongly_connected_components(self.graph)
        elif method == "wcc":
            components = nx.weakly_connected_components(self.graph)
        else:
            raise NotImplementedError
        components = sorted(components, key=len, reverse=True)
        nodes = []
        for i in range(num_component):
            nodes += components[i]
        logging.info(f"Using reduced component with {len(nodes)} nodes")
        self.graph.remove_nodes_from(
            [n for n in self.graph.nodes if n not in nodes]
        )

    def load_motion_at_idx(self, idx, file):
        motion = velocity.MotionWithVelocity(skel=self.skel, fps=self.fps)
        motion = bvh.load(
            file=file,
            motion=motion,
            scale=self.scale,
            load_skel=self.skel is None,
            v_up_skel=self.skel.v_up,
            v_face_skel=self.skel.v_face,
            v_up_env=self.skel.v_up_env,
        )
        motion.resample(fps=self.fps)
        self.motions[idx] = motion
        if self.verbose:
            logging.info(f"Loaded: {file}")

    def load_motions(self, num_workers=8):
        self.motions = bvh.load_parallel(
            self.motion_files,
            scale=self.scale,
            v_up_skel=self.skel.v_up,
            v_face_skel=self.skel.v_face,
            v_up_env=self.skel.v_up_env,
        )
        self.motions = [motion.resample(self.fps) for motion in self.motions]
        total_length = np.sum([m.length() for m in self.motions])
        logging.info(f"Total {total_length:.2f} sec long motions were loaded")
        logging.info(
            f"FPS ~ {self.motions[0].num_frames() / self.motions[0].length():.2f}"
        )

    def save_graph(self, filename="temp_motion_graph.gzip"):
        with gzip.open(filename, "wb") as f:
            pickle.dump(self.graph, f)
            nn = self.graph.number_of_nodes()
            ne = self.graph.number_of_edges()
            logging.info(f"Saved {filename} ({nn} nodes / {ne} edges)")

    def load_graph(self, filename="temp_motion_graph.gzip"):
        with gzip.open(filename, "rb") as f:
            self.graph = pickle.load(f)
            nn = self.graph.number_of_nodes()
            ne = self.graph.number_of_edges()
            logging.info(f"Loaded {filename} ({nn} nodes / {ne} edges)")

    def draw(self):
        nx.draw(self.graph, with_labels=True)
        plt.show()
