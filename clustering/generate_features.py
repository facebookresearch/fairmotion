import argparse
import os
from ..pfnn import Quaternions, Animation, BVH


class Features:

    def __init__(self, anim, joints, frame_time):
        self.anim  = anim
        self.joints = joints
        self.frame_time = frame_time
        self.frame_num = 1
        # humerus length
        self.hl = 
        # shoulder width
        self.sw = 
        # hip width
        self.hw = 

    def next_frame():
        self.frame_num += 1

    def f_move(j1, j2, j3, j4):
        return 0

    def f_nmove(j1, j2, j3, j4):
        return 0

    def f_plane(j1, j2, j3, j4):
        return 0

    def f_nplane(j1, j2, j3, j4):
        return 0

    def f_angle(j1, j2, j3, j4):
        return 0

    def f_fast(j1, ):
        curr_pos = self.anim.positions[self.frame_num][self.joints.index(j1)]
        prev_pos = self.anim.positions[self.frame_num-1][self.joints.index(j1)]
        return np.linalg.norm(curr_pos - prev_pos)/self.frame_time


def extract_features(filepath):
    anim, joints, time_per_frame = BVH.load(filepath)
    features = []
    f = Features(anim, joints, time_per_frame)
    for i in range(1, len(anim)):
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


def main(args):
    features = extract_features(args.file)
    np.save(args.output_file, features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from BVH files")
    parser.add_argument("--file", type=str, help="File to extract features from")
    parser.add_argument("--output-file", type=str)
    args = parser.parse_args()
    main(args)

