from mocap_processing.tasks.clustering import generate_features
import unittest


class Test3DGeometryUtils(unittest.TestCase):
    def test_velocity_above_threshold(self):
        self.assertEqual(
            generate_features.velocity_above_threshold(
                [0, 0, 0], [1, 1, 1], 40, time_per_frame=1.0 / 120
            ),
            True,
        )
        self.assertEqual(
            generate_features.velocity_above_threshold(
                [0, 0, 0], [1, 1, 1], 240, time_per_frame=1.0 / 120
            ),
            False,
        )

    def test_angle_between_range(self):
        self.assertEqual(
            generate_features.angle_within_range(
                [0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 1, 0], (87, 93)
            ),
            True,
        )

    def test_distance_from_plane_normal(self):
        self.assertEqual(
            generate_features.distance_from_plane_normal(
                [0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 1, 2], 3,
            ),
            False,
        )
        self.assertEqual(
            generate_features.distance_from_plane_normal(
                [0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 1, 2], 1
            ),
            True,
        )

    def test_distance_from_plane(self):
        self.assertEqual(
            generate_features.distance_from_plane(
                [0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], 2,
            ),
            False,
        )
        self.assertEqual(
            generate_features.distance_from_plane(
                [0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], 0,
            ),
            True,
        )


if __name__ == "__main__":
    unittest.main()
