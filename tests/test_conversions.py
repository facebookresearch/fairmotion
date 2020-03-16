import numpy as np
import unittest
from basecode.math import mmMath
from tests import utils as test_utils
from mocap_processing.utils import conversions


class TestConversions(unittest.TestCase):
    def test_R2E(self):
        R = test_utils.get_random_R()
        E = conversions.R2E(np.array([R]))
        E_mmMath = mmMath.R2ZYX(R)
        for test, ref in zip(reversed(E[0]), E_mmMath):
            self.assertAlmostEqual(test, ref)

    def test_R2Q(self):
        Q = test_utils.get_random_Q()
        R = conversions.Q2R(Q)
        Q_test = conversions.R2Q(R)
        for q, q_test in zip(Q, Q_test):
            self.assertAlmostEqual(q, q_test)

    def test_R2A(self):
        R = test_utils.get_random_R()
        A = conversions.R2A(np.array([R]))
        R_test = conversions.A2R(A[0])
        for r, r_test in zip(R.flatten(), R_test.flatten()):
            self.assertAlmostEqual(r, r_test)

    def test_Rp2T(self):
        T = test_utils.get_random_T()
        R, p = conversions.T2Rp(T)
        T_test = conversions.Rp2T(R, p)
        for t, t_test in zip(T.flatten(), T_test.flatten()):
            self.assertAlmostEqual(t, t_test)


if __name__ == "__main__":
    unittest.main()
