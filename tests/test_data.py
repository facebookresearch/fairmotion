import numpy as np
import tempfile
import unittest
from mocap_processing.data import bvh
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
                E = conversions.rad2deg(conversions.R2E(R))
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
                    E = conversions.R2E(R, degrees=True)
                    np.testing.assert_almost_equal(E, true_E)


if __name__ == "__main__":
    unittest.main()
