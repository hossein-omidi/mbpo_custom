import unittest
import numpy as np

from MBPO_workspace.src.replay import ScenarioReplayBuffer


class ScenarioReplayBufferTests(unittest.TestCase):
    def test_persistent_whole_day_split_and_checkpoint_roundtrip(self):
        buffer = ScenarioReplayBuffer(capacity=100, seed=17, validation_fraction=0.25)
        for day in range(8):
            sid = f"2024-{(day % 12) + 1:02d}-{(day % 20) + 1:02d}"
            for step in range(5):
                state = np.asarray([0.1, 0.2, 0.3, 0.4, 0.0, 1.0 - step * 0.1], dtype=np.float32)
                nxt = state.copy(); nxt[5] = max(0.0, state[5] - 0.01)
                buffer.add(state, np.zeros(2, dtype=np.float32), 0.001, nxt, False, sid)
        train, holdout = buffer.scenario_split(0.25, seed=999)
        train_ids = set(buffer.scenario_ids[train].tolist())
        holdout_ids = set(buffer.scenario_ids[holdout].tolist())
        self.assertFalse(train_ids & holdout_ids)
        self.assertGreaterEqual(buffer.active_holdout_scenario_count, 1)
        self.assertGreaterEqual(buffer.active_train_scenario_count, 1)

        state = buffer.state_dict()
        restored = ScenarioReplayBuffer(capacity=100, seed=17, validation_fraction=0.25)
        restored.load_state_dict(state)
        self.assertEqual(restored.active_holdout_scenario_count,
                         buffer.active_holdout_scenario_count)
        self.assertEqual(restored.active_train_scenario_count,
                         buffer.active_train_scenario_count)
        self.assertEqual(restored.validation_fraction, 0.25)

    def test_fraction_mismatch_is_rejected(self):
        buffer = ScenarioReplayBuffer(capacity=10, seed=1, validation_fraction=0.2)
        with self.assertRaises(ValueError):
            buffer.scenario_split(0.25, seed=1)


if __name__ == "__main__":
    unittest.main()
