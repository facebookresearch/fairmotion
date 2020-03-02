import tempfile
import unittest
import mocap_processing.motion.kinematics as kinematics
from basecode.math import mmMath


class TestMotion(unittest.TestCase):
    def test_motion(self):
        motion = kinematics.Motion(file="tests/data/sinusoidal.bvh")
        # Inspect 0th frame, root joint
        T = motion.get_pose_by_frame(0).get_transform(0, local=False)
        _, p = mmMath.T2Rp(T)
        self.assertListEqual(list(p), [-3, 6, 5])
        # Inspect 100th frame, root joint
        T = motion.get_pose_by_frame(100).get_transform(0, local=False)
        _, p = mmMath.T2Rp(T)
        self.assertListEqual(list(p), [-3, 6, 5])
        # Inspect 100th frame, "child2" joint
        T = motion.get_pose_by_frame(100).get_transform("child2", local=False)
        _, p = mmMath.T2Rp(T)
        self.assertListEqual(list(p), [-8, 11, 5])

    def test_save_motion(self):
        # Load file
        motion = kinematics.Motion(file="tests/data/sinusoidal.bvh")
        with tempfile.NamedTemporaryFile() as fp:
            # Save loaded file
            motion.save_bvh(fp.name, fps=60.0)
            # Reload saved file and test if it is same as reference file
            ref_motion = kinematics.Motion(file="tests/data/sinusoidal.bvh")
            test_motion = kinematics.Motion(file=fp.name)
            for frame_idx in range(ref_motion.num_frame()):
                for joint in ref_motion.skel.joints:
                    self.assertListEqual(
                        ref_motion.get_pose_by_frame(frame_idx).get_transform(
                            joint.name, local=True
                        ).tolist(),
                        test_motion.get_pose_by_frame(frame_idx).get_transform(
                            joint.name, local=True
                        ).tolist(),
                    )


if __name__ == '__main__':
    unittest.main()
