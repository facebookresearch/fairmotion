from . import generate_features
import unittest

class Test3DGeometryUtils(unittest.TestCase):

    def test_velocity_above_threshold(self):
        self.assertEqual(
            generate_features.velocity_above_threshold([0, 0, 0], [1, 1, 1], 1),
            True,
        )

if __name__ == '__main__':
    unittest.main()
