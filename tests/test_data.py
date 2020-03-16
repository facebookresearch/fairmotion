import tempfile
import unittest
from mocap_processing.data import bvh
from mocap_processing.motion import kinematics


class TestBVH(unittest.TestCase):
    def test_load_motion(self):
        # Load file
        motion = bvh.load(file="tests/data/sinusoidal.bvh")

        ref_motion = kinematics.Motion(file="tests/data/sinusoidal.bvh")
        for frame_idx in range(ref_motion.num_frames()):
            for joint in ref_motion.skel.joints:
                self.assertListEqual(
                    ref_motion.get_pose_by_frame(frame_idx)
                    .get_transform(joint.name, local=True)
                    .tolist(),
                    motion.get_pose_by_frame(frame_idx)
                    .get_transform(joint.name, local=True)
                    .tolist(),
                )

    def test_save_motion(self):
        # Load file
        motion = bvh.load(file="tests/data/sinusoidal.bvh")

        with tempfile.NamedTemporaryFile() as fp:
            # Save loaded file
            bvh.save(motion, fp.name)
            # Reload saved file and test if it is same as reference file
            ref_motion = kinematics.Motion(file="tests/data/sinusoidal.bvh")
            test_motion = bvh.load(file=fp.name)
            for frame_idx in range(ref_motion.num_frames()):
                for joint in ref_motion.skel.joints:
                    self.assertListEqual(
                        ref_motion.get_pose_by_frame(frame_idx)
                        .get_transform(joint.name, local=True)
                        .tolist(),
                        test_motion.get_pose_by_frame(frame_idx)
                        .get_transform(joint.name, local=True)
                        .tolist(),
                    )

    def test_parallel_read(self):
        files = ["tests/data/sinusoidal.bvh"] * 5
        motions = bvh.read_motions_parallel(files, num_workers=3)
        self.assertEqual(len(motions), 5)
        for pose1, pose2 in zip(motions[0].poses[0].data, motions[-1].poses[0].data):
            self.assertListEqual(pose1.flatten().tolist(), pose2.flatten().tolist())


if __name__ == "__main__":
    unittest.main()
