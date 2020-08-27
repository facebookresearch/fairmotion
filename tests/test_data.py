# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import tempfile
import unittest
from fairmotion.data import bvh, asfamc
from fairmotion.ops import conversions
from fairmotion.utils import utils


TEST_SINUSOIDAL_FILE = "tests/data/sinusoidal.bvh"
TEST_SINUSOIDAL_2_FILE = "tests/data/sinusoidal_2.bvh"


class TestBVH(unittest.TestCase):
    def test_load_motion(self):
        # Load file
        motion = bvh.load(file=TEST_SINUSOIDAL_FILE)

        # Read pose data from all frames (in Euler angle)
        with open(TEST_SINUSOIDAL_FILE) as f:
            file_content = f.readlines()
        for frame_num, line in enumerate(file_content[-motion.num_frames() :]):
            # Skip first 3 entries that store translation data, and read the
            # Euler angle data of joints
            angle_data = np.array(
                list(map(float, line.strip().split()))[3:]
            ).reshape(-1, 3)
            for T, true_E in zip(motion.poses[frame_num].data, angle_data):
                R, _ = conversions.T2Rp(T)
                E = conversions.R2E(R, order="XYZ", degrees=True)
                np.testing.assert_almost_equal(E, true_E)

    def test_load_parallel(self):
        # Load files
        motions = bvh.load_parallel(
            files=[TEST_SINUSOIDAL_FILE, TEST_SINUSOIDAL_2_FILE],
        )
        self.assertEqual(len(motions), 2)
        # Test if the loaded motion objects are not same in the first frame
        self.assertNotEqual(motions[0].poses[0], motions[1].poses[0])

        # Use kwargs
        v_up_skel = utils.str_to_axis("y")
        v_face_skel = utils.str_to_axis("z")
        v_up_env = utils.str_to_axis("x")
        motions = bvh.load_parallel(
            files=[TEST_SINUSOIDAL_FILE, TEST_SINUSOIDAL_2_FILE],
            scale=0.1,
            v_up_skel=v_up_skel,
            v_face_skel=v_face_skel,
            v_up_env=v_up_env,
        )
        np.testing.assert_equal(motions[0].skel.v_up, v_up_skel)
        np.testing.assert_equal(motions[0].skel.v_face, v_face_skel)
        np.testing.assert_equal(motions[0].skel.v_up_env, v_up_env)

    def test_save_motion(self):
        # Load file
        motion = bvh.load(file=TEST_SINUSOIDAL_FILE)

        with tempfile.NamedTemporaryFile() as fp:
            # Save loaded file
            bvh.save(motion, fp.name, rot_order="XYZ")
            # Reload saved file and test if it is same as reference file
            # Read pose data from all frames (in Euler angle)
            with open(TEST_SINUSOIDAL_FILE) as f:
                orig_file = f.readlines()
            with open(fp.name) as f:
                saved_file = f.readlines()
            for orig_line, saved_line in zip(
                orig_file[-motion.num_frames() :],
                saved_file[-motion.num_frames() :],
            ):
                # Skip first 3 entries that store translation data, and read
                # the Euler angle data of joints
                orig_data, saved_data = [
                    np.array(list(map(float, line.strip().split())))
                    for line in [orig_line, saved_line]
                ]
                np.testing.assert_almost_equal(orig_data, saved_data)

            # Reload saved file and test if it has the same data as original
            # motion object
            with open(TEST_SINUSOIDAL_FILE) as f:
                file_content = f.readlines()
            for frame_num, line in enumerate(
                file_content[-motion.num_frames() :]
            ):
                # Skip first 3 entries that store translation data, and read
                # the Euler angle data of joints
                angle_data = np.array(
                    list(map(float, line.strip().split()))[3:]
                ).reshape(-1, 3)
                for T, true_E in zip(motion.poses[frame_num].data, angle_data):
                    R, _ = conversions.T2Rp(T)
                    E = conversions.R2E(R, order="XYZ", degrees=True)
                    np.testing.assert_almost_equal(E, true_E)


class TestASFAMC(unittest.TestCase):
    def test_load_motion(self):
        # Load file
        motion = asfamc.load(
            file="tests/data/01.asf", motion="tests/data/01_01.amc"
        )
        motion_bvh = bvh.load(file="tests/data/01_01.bvh")
        for i in range(1, motion.num_frames()):
            pose = motion.get_pose_by_frame(i)
            pose_bvh = motion_bvh.get_pose_by_frame(i)
            """
            for k, j in zip(pose.skel.joints, pose_bvh.skel.joints):
                print(k.name, j.name)
            root Hips
            lhipjoint LHipJoint
            lfemur LeftUpLeg
            ltibia LeftLeg
            lfoot LeftFoot
            ltoes LeftToeBase
            rhipjoint RHipJoint
            rfemur RightUpLeg
            rtibia RightLeg
            rfoot RightFoot
            rtoes RightToeBase
            lowerback LowerBack
            upperback Spine
            thorax Spine1
            lowerneck Neck
            upperneck Neck1
            head Head
            lclavicle LeftShoulder
            lhumerus LeftArm
            lradius LeftForeArm
            lwrist LeftHand
            lhand LeftFingerBase
            lfingers LFingers
            lthumb LThumb
            rclavicle RightShoulder
            rhumerus RightArm
            rradius RightForeArm
            rwrist RightHand
            rhand RightFingerBase
            rfingers RFingers
            rthumb RThumb
            """
            for joint_idx, (k, j) in enumerate(zip(pose.data, pose_bvh.data)):
                # As asfamc and bvh are from different sources, we are not strictly comparing them.
                # We require no more than two different elements.
                failures = 0
                if joint_idx == 0:
                    compare_rotation = 0
                else:
                    compare_rotation = 1
                for kk, jj in zip(
                    np.nditer(k[:, : 3 - compare_rotation]),
                    np.nditer(j[:, : 3 - compare_rotation]),
                ):
                    if abs(kk - jj) > 0.2:
                        failures += 1
                assert failures <= 2, failures


if __name__ == "__main__":
    unittest.main()
