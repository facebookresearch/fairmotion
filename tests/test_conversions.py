# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import unittest
from tests import utils as test_utils
from fairmotion.ops import conversions


class TestConversions(unittest.TestCase):
    def test_R2E(self):
        R = test_utils.get_random_R()
        np.testing.assert_almost_equal(
            conversions.R2E(np.array([R]))[0], conversions.R2E(R)
        )

    def test_R2Q(self):
        Q = test_utils.get_random_Q()
        R = conversions.Q2R(Q)

        np.testing.assert_almost_equal(
            conversions.R2Q(np.array([R]))[0], conversions.R2Q(R)
        )
        np.testing.assert_almost_equal(
            conversions.Q2R(np.array([Q]))[0], conversions.Q2R(Q)
        )

        Q_test = conversions.R2Q(R)
        for q, q_test in zip(Q, Q_test):
            self.assertAlmostEqual(q, q_test)

        # Test if batched input conversion is same as single input
        R_batched = np.array([test_utils.get_random_R() for _ in range(2)])
        Q_batched = conversions.R2Q(R_batched)
        for R, Q in zip(R_batched, Q_batched):
            np.testing.assert_array_equal(conversions.R2Q(R), Q)

    def test_R2A(self):
        R = test_utils.get_random_R()
        A = conversions.R2A(np.array([R]))

        np.testing.assert_almost_equal(
            conversions.R2A(np.array([R]))[0], conversions.R2A(R)
        )
        np.testing.assert_almost_equal(
            conversions.A2R(np.array([A]))[0], conversions.A2R(A)
        )

        R_test = conversions.A2R(A[0])
        for r, r_test in zip(R.flatten(), R_test.flatten()):
            self.assertAlmostEqual(r, r_test)

    def test_Rp2T(self):
        T = test_utils.get_random_T()
        R, p = conversions.T2Rp(T)

        np.testing.assert_almost_equal(
            conversions.Rp2T(np.array([R]), np.array([p]))[0],
            conversions.Rp2T(R, p),
        )

        T_test = conversions.Rp2T(R, p)
        for t, t_test in zip(T.flatten(), T_test.flatten()):
            self.assertAlmostEqual(t, t_test)

    def test_p2T(self):
        T = test_utils.get_random_T()
        _, p = conversions.T2Rp(T)

        np.testing.assert_almost_equal(
            conversions.p2T(np.array([p]))[0], conversions.p2T(p),
        )

    def test_R2T(self):
        T = test_utils.get_random_T()
        R, _ = conversions.T2Rp(T)

        np.testing.assert_almost_equal(
            conversions.R2T(np.array([R]))[0], conversions.R2T(R),
        )

    def test_T2Rp(self):
        T = test_utils.get_random_T()

        np.testing.assert_almost_equal(
            conversions.T2Rp(np.array([T]))[0][0], conversions.T2Rp(T)[0],
        )


if __name__ == "__main__":
    unittest.main()
