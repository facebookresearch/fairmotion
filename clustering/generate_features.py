import argparse
import math
import numpy as np
import os

from enum import Enum
from ..pfnn import Quaternions, Animation, BVH


GENERIC_TO_PFNN_MAPPING = {
    "neck": "Neck",
    "lhip": "LeftUpLeg",
    "rhip": "RightUpLeg",
    "lwrist": "LeftHand",
    "rwrist": "RightHand",
    "belly": "Spine",
    "chest": "Spine1",
    "relbow": "RightForeArm",
    "rshoulder": "RightArm",
    "lelbow": "LeftForeArm",
    "lshoulder": "LeftArm",
    "lankle": "LeftFoot",
    "rankle": "RightFoot",
    "rtoes": "RightToeBase",
    "ltoes": "LeftToeBase",
}

class Skeleton(Enum):
    PFNN = 'pfnn'
    CMU = 'cmu'


def distance_between_points(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def side_of_plane(a, b, c, p):
    ba = np.array(b) - np.array(a)
    ca = np.array(c) - np.array(a)
    cross = np.cross(ba, ca)

    pa = np.array(p) - np.array(a)
    return np.dot(cross, pa) > 0


def side_of_plane_normal(n1, n2, a, p):
    normal = np.array(n1) - np.array(n2)
    pa = np.array(p) - np.array(a)
    return np.dot(normal, pa) > 0


def angle_within_range(j1, j2, k1, k2, range):
    j = np.array(j2) - np.array(j1)
    k = np.array(k2) - np.array(k1)

    angle = np.arccos(np.dot(j, k)/(np.linalg.norm(j) * np.linalg.norm(k)))
    angle = np.degrees(angle)

    if angle > range[0] and angle < range[1]:
        return True
    else:
        return False


def velocity_direction_above_threshold(j1, j1_prev, j2, j2_prev, p, p_prev, threshold, time_per_frame=1/120):
    velocity = np.array(p) - np.array(j1) - (np.array(p_prev) - np.array(j1_prev))
    direction = np.array(j2) - np.array(j1)

    velocity_along_direction = np.dot(velocity, direction)/np.linalg.norm(direction)
    velocity_along_direction = velocity_along_direction/time_per_frame
    return velocity_along_direction > threshold


def velocity_direction_above_threshold_normal(j1, j1_prev, j2, j3, p, p_prev, threshold, time_per_frame=1/120):
    velocity = np.array(p) - np.array(j1) - (np.array(p_prev) - np.array(j1_prev))
    j31 = np.array(j3) - np.array(j1)
    j21 = np.array(j2) - np.array(j1)
    direction = np.cross(j31, j21)

    velocity_along_direction = np.dot(velocity, direction)/np.linalg.norm(direction)
    velocity_along_direction = velocity_along_direction/time_per_frame
    return velocity_along_direction > threshold


def velocity_above_threshold(p, p_prev, threshold, time_per_frame=1/120):
    velocity = np.linalg.norm(np.array(p) - np.array(p_prev))/time_per_frame
    return velocity > threshold


class Features:
    def __init__(self, anim, joints, frame_time, skeleton=Skeleton.PFNN):
        self.global_positions = Animation.positions_global(anim)
        self.joints = joints
        self.frame_time = frame_time
        self.frame_num = 1
        # humerus length
        self.hl = 0
        # shoulder width
        self.sw = 0
        # hip width
        self.hw = 0
        if skeleton == Skeleton.PFNN:
            self.joint_mapping = GENERIC_TO_PFNN_MAPPING
        else:
            self.joint_mapping = GENERIC_TO_CMU_MAPPING

    def next_frame():
        self.frame_num += 1


    def transform_and_fetch_position(j):
        return self.global_positions[self.frame_num][self.joints.index(self.joint_mapping(j))]

    def transform_and_fetch_prev_position(j):
        return self.global_positions[self.frame_num-1][self.joints.index(self.joint_mapping(j))]

    def f_move(j1, j2, j3, j4, range):
        j1, j2, j3, j4 = [self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]]
        j1_prev, j2_prev, j3_prev, j4_prev = [self.transform_and_fetch_prev_position(j) for j in [j1, j2, j3, j4]]
        return velocity_direction_above_threshold(j1, j1_prev, j2, j2_prev, j3, j3_prev, range)

    def f_nmove(j1, j2, j3, j4, range):
        j1, j2, j3, j4 = [self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]]
        j1_prev, j2_prev, j3_prev, j4_prev = [self.transform_and_fetch_prev_position(j) for j in [j1, j2, j3, j4]]
        return velocity_direction_above_threshold_normal(j1, j1_prev, j2, j3, j4, j4_prev, range)

    def f_plane(j1, j2, j3, j4):
        j1, j2, j3, j4 = [self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]]
        return side_of_plane(j1, j2, j3, j4)

    def f_nplane(j1, j2, j3, j4):
        j1, j2, j3, j4 = [self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]]
        return side_of_plane_normal(j1, j2, j3, j4)

    def f_angle(j1, j2, j3, j4, range):
        j1, j2, j3, j4 = [self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]]
        return angle_within_range(j1, j2, j3, j4, range)

    def f_fast(j1, threshold):
        j1 = self.transform_and_fetch_position(j1)
        j1_prev = self.transform_and_fetch_prev_position(j1)
        return velocity_above_threshold(j1, j1_prev, threshold)


def extract_features(filepath):
    anim, joints, time_per_frame = BVH.load(filepath)
    features = []
    f = Features(anim, joints, time_per_frame, Skeleton.PFNN)
    for i in range(1, len(anim), 30):
        pose_features = []
        pose_features.append(f.f_nmove("neck", "rhip", "lhip", "rwrist", 1.8*f.hl))
        pose_features.append(f.f_nmove("neck", "lhip", "rhip", "lwrist", 1.8*f.hl))
        pose_features.append(f.f_nplane("chest", "neck", "neck", "rwrist", 0.2*f.hl))
        pose_features.append(f.f_nplane("chest", "neck", "neck", "lwrist", 0.2*f.hl))
        pose_features.append(f.f_move("belly", "chest", "chest", "rwrist", 1.8*f.hl))
        pose_features.append(f.f_move("belly", "chest", "chest", "lwrist", 1.8*f.hl))
        pose_features.append(f.f_angle("relbow", "rshoulder", "relbow", "rwrist", [0, 110]))
        pose_features.append(f.f_angle("lelbow", "lshoulder", "lelbow", "lwrist", [0, 110]))
        pose_features.append(f.f_nplane("lshoulder", "rshoulder", "lwrist", "rwrist", 2.5*f.sw))
        pose_features.append(f.f_move("lwrist", "rwrist", "rwrist", "lwrist", 1.4*f.hl))
        pose_features.append(f.f_move("rwrist", "root", "lwrist", "root", 1.4*f.hl))
        pose_features.append(f.f_move("lwrist", "root", "rwrist", "root", 1.4*f.hl))
        pose_features.append(f.f_fast("rwrist", 2.5*f.hl))
        pose_features.append(f.f_fast("lwrist", 2.5*f.hl))
        pose_features.append(f.f_plane("root", "lhip", "ltoes", "rankle", 0.38*f.hl))
        pose_features.append(f.f_plane("root", "rhip", "rtoes", "lankle", 0.38*f.hl))
        pose_features.append(f.f_fast("root", 2.3*f.hl))
        f.next_frame()
    return pose_features


def main(args):
    features = extract_features(args.file)
    np.save(args.output_file, features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from BVH files")
    parser.add_argument("--file", type=str, help="File to extract features from")
    parser.add_argument("--output-file", type=str)
    args = parser.parse_args()
    main(args)
