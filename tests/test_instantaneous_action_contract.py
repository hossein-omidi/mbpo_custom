from __future__ import annotations

import unittest
import numpy as np

from mbpo.geometry10 import (
    ACTION_SEMANTICS,
    action_for_target_orientation,
    build,
    hold_action,
    pose_transition,
    target_orientation_from_action,
)


class InstantaneousTargetActionTests(unittest.TestCase):
    def test_action_contract_identifier(self):
        self.assertEqual(
            ACTION_SEMANTICS,
            "instantaneous_target_tilt_absolute_azimuth_relative_v1",
        )

    def test_zero_is_not_a_hold_action(self):
        tilt, azimuth, dtilt, dazimuth = target_orientation_from_action(
            30.0, 359.0, [0.0, 0.0]
        )
        self.assertAlmostEqual(tilt, 45.0)
        self.assertAlmostEqual(azimuth, 359.0)
        self.assertAlmostEqual(dtilt, 15.0)
        self.assertAlmostEqual(dazimuth, 0.0)

    def test_hold_action_preserves_pose(self):
        action = hold_action(30.0)
        tilt, azimuth, dtilt, dazimuth = target_orientation_from_action(
            30.0, 180.0, action
        )
        self.assertAlmostEqual(tilt, 30.0, places=5)
        self.assertAlmostEqual(azimuth, 180.0, places=5)
        self.assertAlmostEqual(dtilt, 0.0, places=5)
        self.assertAlmostEqual(dazimuth, 0.0, places=5)

    def test_azimuth_wrap_is_circular(self):
        tilt, azimuth, dtilt, dazimuth = target_orientation_from_action(
            30.0, 359.0, [0.5, 0.01]
        )
        self.assertAlmostEqual(tilt, 67.5)
        self.assertAlmostEqual(azimuth, 0.8, places=6)
        self.assertAlmostEqual(dtilt, 37.5)
        self.assertAlmostEqual(dazimuth, 1.8, places=6)

    def test_target_encoder_is_inverse_of_decoder(self):
        action = action_for_target_orientation(30.0, 359.0, 90.0, 1.0)
        tilt, azimuth, _, _ = target_orientation_from_action(30.0, 359.0, action)
        self.assertAlmostEqual(tilt, 90.0, places=5)
        self.assertAlmostEqual(azimuth, 1.0, places=5)
        self.assertTrue(np.all(np.abs(action) <= 1.0))

    def test_geometry10_pose_transition_uses_same_mapping(self):
        state = build(500.0, 100.0, 100.0, 30.0, 359.0, 40.0, 100.0, 1.0)
        result = pose_transition(state[None, :], [[0.0, 0.0]], 195)
        self.assertAlmostEqual(float(result[0][0]), 45.0 / 90.0, places=6)
        self.assertAlmostEqual(float(result[2][0]), 194.0 / 195.0, places=8)
        self.assertFalse(bool(result[5][0]))

    def test_terminal_clock_is_unchanged(self):
        state = build(500.0, 100.0, 100.0, 30.0, 359.0, 40.0, 100.0, 1.0 / 195.0)
        result = pose_transition(state[None, :], [[-1.0, 1.0]], 195)
        self.assertTrue(bool(result[5][0]))
        self.assertAlmostEqual(float(result[2][0]), 0.0)

    def test_invalid_actions_are_rejected_not_clipped(self):
        for action in ([1.0001, 0.0], [0.0, float("nan")]):
            with self.assertRaises(ValueError):
                target_orientation_from_action(30.0, 180.0, action)

    def test_irradiance_over_normalization_range_is_rejected(self):
        with self.assertRaises(ValueError):
            build(2001.0, 100.0, 100.0, 30.0, 180.0, 40.0, 100.0, 1.0)


if __name__ == "__main__":
    unittest.main()
