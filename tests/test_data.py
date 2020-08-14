import numpy as np
import tempfile
import unittest
from mocap_processing.data import bvh, asfamc
from mocap_processing.utils import conversions


class TestBVH(unittest.TestCase):
    def test_load_motion(self):
        # Load file
        motion = bvh.load(file="tests/data/sinusoidal.bvh")

        # Read pose data from all frames (in Euler angle)
        with open("tests/data/sinusoidal.bvh") as f:
            file_content = f.readlines()
        for frame_num, line in enumerate(file_content[-motion.num_frames():]):
            # Skip first 3 entries that store translation data, and read the
            # Euler angle data of joints
            angle_data = np.array(
                list(map(float, line.strip().split()))[3:]
            ).reshape(-1, 3)
            for T, true_E in zip(motion.poses[frame_num].data, angle_data):
                R, _ = conversions.T2Rp(T)
                E = conversions.rad2deg(conversions.R2E(R.transpose()))
                np.testing.assert_almost_equal(E, true_E)

    def test_save_motion(self):
        # Load file
        motion = bvh.load(file="tests/data/sinusoidal.bvh")

        with tempfile.NamedTemporaryFile() as fp:
            # Save loaded file
            bvh.save(motion, fp.name)

            # Reload saved file and test if it is same as reference file
            # Read pose data from all frames (in Euler angle)
            with open("tests/data/sinusoidal.bvh") as f:
                orig_file = f.readlines()
            with open(fp.name) as f:
                saved_file = f.readlines()
            for orig_line, saved_line in zip(
                orig_file[-motion.num_frames():],
                saved_file[-motion.num_frames():]
            ):
                # Skip first 3 entries that store translation data, and read
                # the Euler angle data of joints
                orig_data, saved_data = [
                    np.array(list(map(float, line.strip().split())))
                    for line in [orig_line, saved_line]
                ]
                # Saved angles are in 'Z Y X' order, convert to 'X Y Z' since
                # test/data/sinusoidal.bvh is in that order
                saved_data = np.flip(
                    saved_data.reshape(-1, 3), axis=1
                ).reshape(-1)
                # Flip translation data back to 'X Y Z'
                saved_data[0], saved_data[2] = saved_data[2], saved_data[0]
                np.testing.assert_almost_equal(orig_data, saved_data)

            # Reload saved file and test if it has the same data as original
            # motion object
            with open("tests/data/sinusoidal.bvh") as f:
                file_content = f.readlines()
            for frame_num, line in enumerate(
                file_content[-motion.num_frames():]
            ):
                # Skip first 3 entries that store translation data, and read
                # the Euler angle data of joints
                angle_data = np.array(
                    list(map(float, line.strip().split()))[3:]
                ).reshape(-1, 3)
                for T, true_E in zip(motion.poses[frame_num].data, angle_data):
                    R, _ = conversions.T2Rp(T)
                    E = conversions.rad2deg(conversions.R2E(R.transpose()))
                    np.testing.assert_almost_equal(E, true_E)

class TestASFAMC(unittest.TestCase):
    def test_load_motion(self):
        # Load file
        motion = asfamc.load(file="tests/data/01.asf", motion='tests/data/01_01.amc')
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
                for kk, jj in zip(np.nditer(k[:, :3 - compare_rotation]), np.nditer(j[:, :3 - compare_rotation])):
                    if abs(kk - jj) > 0.2:
                        failures += 1
                assert failures <= 2, failures
if __name__ == "__main__":
    unittest.main()
