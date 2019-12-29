"""
Example command:
python mocap_processing/tasks/clustering/generate_features.py --type kinetic --folder ~/data/clustering_pfnn/pfnn_bvh_split --output-folder ~/data/clustering_pfnn/pfnn_bvh_split_kinetic_joint_position_features
"""

import argparse
import numpy as np
import os
import tqdm
from collections import defaultdict
from enum import Enum
from multiprocessing import Pool
from mocap_processing.motion.pfnn import Animation, BVH


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
    "rknee": "RightLeg",
    "lknee": "LeftLeg",
    "root": "LowerBack",
}

thresholds = None

GENERIC_TO_CMU_MAPPING = {}


class Skeleton(Enum):
    PFNN = "pfnn"
    CMU = "cmu"


def distance_between_points(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def distance_from_plane(a, b, c, p, threshold):
    ba = np.array(b) - np.array(a)
    ca = np.array(c) - np.array(a)
    cross = np.cross(ca, ba)

    pa = np.array(p) - np.array(a)
    return np.dot(cross, pa) / np.linalg.norm(cross) > threshold


def distance_from_plane_normal(n1, n2, a, p, threshold):
    normal = np.array(n2) - np.array(n1)
    pa = np.array(p) - np.array(a)
    return np.dot(normal, pa) / np.linalg.norm(normal) > threshold


def angle_within_range(j1, j2, k1, k2, range):
    j = np.array(j2) - np.array(j1)
    k = np.array(k2) - np.array(k1)

    angle = np.arccos(np.dot(j, k) / (np.linalg.norm(j) * np.linalg.norm(k)))
    angle = np.degrees(angle)

    if angle > range[0] and angle < range[1]:
        return True
    else:
        return False


def velocity_direction_above_threshold(
    j1, j1_prev, j2, j2_prev, p, p_prev, threshold, time_per_frame=1 / 120
):
    velocity = np.array(p) - np.array(j1) - (np.array(p_prev) - np.array(j1_prev))
    direction = np.array(j2) - np.array(j1)

    velocity_along_direction = np.dot(velocity, direction) / np.linalg.norm(direction)
    velocity_along_direction = velocity_along_direction / time_per_frame
    return velocity_along_direction > threshold


def velocity_direction_above_threshold_normal(
    j1, j1_prev, j2, j3, p, p_prev, threshold, time_per_frame=1 / 120
):
    velocity = np.array(p) - np.array(j1) - (np.array(p_prev) - np.array(j1_prev))
    j31 = np.array(j3) - np.array(j1)
    j21 = np.array(j2) - np.array(j1)
    direction = np.cross(j31, j21)

    velocity_along_direction = np.dot(velocity, direction) / np.linalg.norm(direction)
    velocity_along_direction = velocity_along_direction / time_per_frame
    return velocity_along_direction > threshold


def velocity_above_threshold(p, p_prev, threshold, time_per_frame=1 / 120):
    velocity = np.linalg.norm(np.array(p) - np.array(p_prev)) / time_per_frame
    return velocity > threshold


def calc_average_velocity(positions, i, joint_idx, sliding_window, frame_time):
    current_window = 0
    average_velocity = np.zeros(len(positions[0][joint_idx]))
    for j in range(-sliding_window, sliding_window + 1):
        if i + j - 1  < 0 or i + j >= len(positions):
            continue
        average_velocity += positions[i+j][joint_idx] - positions[i+j-1][joint_idx]
        current_window += 1
    return np.linalg.norm(average_velocity / (current_window * frame_time))
    


class P95Thresholds:
    def __init__(self, root_folder, sliding_window=2):
        print("Calculating velocity clipping values")
        self.velocities = defaultdict(list)
        self.sliding_window = sliding_window
        for root, _, files in os.walk(root_folder, topdown=False):
            for filename in tqdm.tqdm(files):
                anim, joints, time_per_frame = BVH.load(os.path.join(root, filename))
                self.joints = joints
                self._update_velocities(Animation.positions_global(anim), time_per_frame)
        self.thresholds = self._compute_p95_thresholds()


    def _update_velocities(self, positions, frame_time):
        for i in range(1, len(positions)):
            for joint_idx in range(len(self.joints)):
                velocity = calc_average_velocity(positions, i, joint_idx, self.sliding_window, frame_time)
                self.velocities[self.joints[joint_idx]].append(velocity)


    def _compute_p95_thresholds(self):
        return {joint: np.percentile(self.velocities[joint], 95) for joint in self.joints}

    
    def get_threshold(self, joint_idx):
        return self.thresholds[self.joints[joint_idx]]
    

    def __str__(self):
        print_str = ""
        for key, value in self.thresholds.items():
            print_str += key + ":" + str(value) + "\n"
        return print_str


class KineticFeatures:
    def __init__(self, anim, joints, frame_time, thresholds, sliding_window=2):
        self.local_positions = anim.positions
        global_positions = Animation.positions_global(anim)
        for joint in range(len(joints) - 1, 0, -1):
            global_positions[:,joint] =  global_positions[:,joint] - global_positions[:,anim.parents[joint]]
        self.positions = global_positions 
        self.joints = joints
        self.frame_time = frame_time
        self.thresholds = thresholds
        self.sliding_window = sliding_window


    def average_kinetic_energy(self, joint):
        average_kinetic_energy = 0
        for i in range(1, len(self.positions)):
            average_velocity = calc_average_velocity(self.positions, i, joint, self.sliding_window, self.frame_time)
            average_velocity = np.clip(average_velocity, a_min=0, a_max=self.thresholds.get_threshold(joint))
            average_kinetic_energy += average_velocity ** 2
        average_kinetic_energy = average_kinetic_energy / (
            len(self.positions) - 1.0
        )
        return average_kinetic_energy 


    def local_position_stats(self, joint):
        return np.mean(self.local_positions[:, joint], axis=0), np.std(self.local_positions[:, joint], axis=0)


class ManualFeatures:
    def __init__(self, anim, joints, frame_time, skeleton=Skeleton.PFNN):
        self.global_positions = Animation.positions_global(anim)
        self.joints = joints
        self.frame_time = frame_time
        self.frame_num = 1
        self.offsets = anim.offsets
        if skeleton == Skeleton.PFNN:
            self.joint_mapping = GENERIC_TO_PFNN_MAPPING
        else:
            self.joint_mapping = GENERIC_TO_CMU_MAPPING

        # humerus length
        self.hl = distance_between_points(
            self.transform_and_fetch_offset("lshoulder"),
            self.transform_and_fetch_offset("lelbow"),
        )
        # shoulder width
        self.sw = distance_between_points(
            self.transform_and_fetch_offset("lshoulder"),
            self.transform_and_fetch_offset("rshoulder"),
        )
        # hip width
        self.hw = distance_between_points(
            self.transform_and_fetch_offset("lhip"),
            self.transform_and_fetch_offset("rhip"),
        )

    def next_frame(self):
        self.frame_num += 1

    def transform_and_fetch_position(self, j):
        if j == "y_unit":
            return [0, 1, 0]
        elif j == "minus_y_unit":
            return [0, -1, 0]
        elif j == "zero":
            return [0, 0, 0]
        elif j == "y_min":
            return [
                0,
                min([y for (_, y, _) in self.global_positions[self.frame_num]]),
                0,
            ]
        return self.global_positions[self.frame_num][
            self.joints.index(self.joint_mapping[j])
        ]

    def transform_and_fetch_prev_position(self, j):
        return self.global_positions[self.frame_num - 1][
            self.joints.index(self.joint_mapping[j])
        ]

    def transform_and_fetch_offset(self, j):
        return self.offsets[self.joints.index(self.joint_mapping[j])]

    def f_move(self, j1, j2, j3, j4, range):
        j1_prev, j2_prev, j3_prev, j4_prev = [
            self.transform_and_fetch_prev_position(j) for j in [j1, j2, j3, j4]
        ]
        j1, j2, j3, j4 = [
            self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]
        ]
        return velocity_direction_above_threshold(
            j1, j1_prev, j2, j2_prev, j3, j3_prev, range
        )

    def f_nmove(self, j1, j2, j3, j4, range):
        j1_prev, j2_prev, j3_prev, j4_prev = [
            self.transform_and_fetch_prev_position(j) for j in [j1, j2, j3, j4]
        ]
        j1, j2, j3, j4 = [
            self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]
        ]
        return velocity_direction_above_threshold_normal(
            j1, j1_prev, j2, j3, j4, j4_prev, range
        )

    def f_plane(self, j1, j2, j3, j4, threshold):
        j1, j2, j3, j4 = [
            self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]
        ]
        return distance_from_plane(j1, j2, j3, j4, threshold)

    def f_nplane(self, j1, j2, j3, j4, threshold):
        j1, j2, j3, j4 = [
            self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]
        ]
        return distance_from_plane_normal(j1, j2, j3, j4, threshold)

    def f_angle(self, j1, j2, j3, j4, range):
        j1, j2, j3, j4 = [
            self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]
        ]
        return angle_within_range(j1, j2, j3, j4, range)

    def f_fast(self, j1, threshold):
        j1_prev = self.transform_and_fetch_prev_position(j1)
        j1 = self.transform_and_fetch_position(j1)
        return velocity_above_threshold(j1, j1_prev, threshold)


def extract_manual_features(anim, joints, time_per_frame):
    features = []
    f = ManualFeatures(anim, joints, time_per_frame, Skeleton.PFNN)
    for i in range(1, len(anim), 30):
        pose_features = []
        pose_features.append(f.f_nmove("neck", "rhip", "lhip", "rwrist", 1.8 * f.hl))
        pose_features.append(f.f_nmove("neck", "lhip", "rhip", "lwrist", 1.8 * f.hl))
        pose_features.append(f.f_nplane("chest", "neck", "neck", "rwrist", 0.2 * f.hl))
        pose_features.append(f.f_nplane("chest", "neck", "neck", "lwrist", 0.2 * f.hl))
        pose_features.append(f.f_move("belly", "chest", "chest", "rwrist", 1.8 * f.hl))
        pose_features.append(f.f_move("belly", "chest", "chest", "lwrist", 1.8 * f.hl))
        pose_features.append(
            f.f_angle("relbow", "rshoulder", "relbow", "rwrist", [0, 110])
        )
        pose_features.append(
            f.f_angle("lelbow", "lshoulder", "lelbow", "lwrist", [0, 110])
        )
        pose_features.append(
            f.f_nplane("lshoulder", "rshoulder", "lwrist", "rwrist", 2.5 * f.sw)
        )
        pose_features.append(
            f.f_move("lwrist", "rwrist", "rwrist", "lwrist", 1.4 * f.hl)
        )
        pose_features.append(f.f_move("rwrist", "root", "lwrist", "root", 1.4 * f.hl))
        pose_features.append(f.f_move("lwrist", "root", "rwrist", "root", 1.4 * f.hl))
        pose_features.append(f.f_fast("rwrist", 2.5 * f.hl))
        pose_features.append(f.f_fast("lwrist", 2.5 * f.hl))
        pose_features.append(f.f_plane("root", "lhip", "ltoes", "rankle", 0.38 * f.hl))
        pose_features.append(f.f_plane("root", "rhip", "rtoes", "lankle", 0.38 * f.hl))
        pose_features.append(
            f.f_nplane("zero", "y_unit", "y_min", "rankle", 1.2 * f.hl)
        )
        pose_features.append(
            f.f_nplane("zero", "y_unit", "y_min", "lankle", 1.2 * f.hl)
        )
        pose_features.append(f.f_nplane("lhip", "rhip", "lankle", "rankle", 2.1 * f.hw))
        pose_features.append(f.f_angle("rknee", "rhip", "rknee", "rankle", [0, 110]))
        pose_features.append(f.f_angle("lknee", "lhip", "lknee", "lankle", [0, 110]))
        pose_features.append(f.f_fast("rankle", 2.5 * f.hl))
        pose_features.append(f.f_fast("lankle", 2.5 * f.hl))
        pose_features.append(
            f.f_angle("neck", "root", "rshoulder", "relbow", [25, 180])
        )
        pose_features.append(
            f.f_angle("neck", "root", "lshoulder", "lelbow", [25, 180])
        )
        pose_features.append(f.f_angle("neck", "root", "rhip", "rknee", [50, 180]))
        pose_features.append(f.f_angle("neck", "root", "lhip", "lknee", [50, 180]))
        pose_features.append(f.f_plane("rankle", "neck", "lankle", "root", 0.5 * f.hl))
        pose_features.append(f.f_angle("neck", "root", "zero", "y_unit", [70, 110]))
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


def extract_kinetic_features(anim, joints, time_per_frame, thresholds):
    features = KineticFeatures(anim, joints, time_per_frame, thresholds)
    kinetic_feature_vector = []
    for i in range(len(joints)):
        positions_mean, positions_stddev = features.local_position_stats(i)
        feature_vector = [features.average_kinetic_energy(i)]
        feature_vector.extend(positions_mean)
        feature_vector.extend(positions_stddev)
        kinetic_feature_vector.extend(feature_vector)
    return kinetic_feature_vector


def extract_features(filepath, feature_type, thresholds=None):
    anim, joints, time_per_frame = BVH.load(filepath)
    if feature_type == "manual":
        return extract_manual_features(anim, joints, time_per_frame)
    else:
        return extract_kinetic_features(anim, joints, time_per_frame, thresholds)



def wrapper_extract_features(inputs):
    global thresholds
    filepath = inputs[0]
    feature_type = inputs[1]
    output_filepath  =inputs[2]

    print("Started " + filepath)
    print(thresholds.get_threshold(2), thresholds.get_threshold(3), thresholds.get_threshold(5), thresholds.get_threshold(10))
    features = extract_features(filepath, feature_type, thresholds)
    np.save(output_filepath, features)
    filename = filepath.split("/")[-1]
    with open(os.path.join(args.output_folder, "features.tsv"), "a") as all_features:
        if args.type == "manual":
            all_features.write(
                filename + ":"
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
    print("Finished " + filepath)

def main(args):
    global thresholds
    thresholds = P95Thresholds(args.folder)
    print(thresholds.get_threshold(2), thresholds.get_threshold(3), thresholds.get_threshold(5), thresholds.get_threshold(10))
    print(thresholds)
    pool = Pool(40)
    for root, _, files in os.walk(args.folder, topdown=False):
        pool.map(wrapper_extract_features, [(os.path.join(root, filename), args.type, os.path.join(args.output_folder, filename)) for filename in files])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from BVH files")
    parser.add_argument(
        "--type", nargs="?", choices=["manual", "kinetic", "distribution"], default="manual"
    )
    parser.add_argument(
        "--folder", type=str, help="Folder with files to extract features from", required=True,
    )
    parser.add_argument("--output-folder", type=str, required=True)
    args = parser.parse_args()
    main(args)
