"""Regression tests for config extraction without importing the TF/Ray stack."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from SAC_workspace.src.config import config as reference


class ReferenceResolutionTests(unittest.TestCase):
    def test_evaluation_builder_is_independent_of_training_override(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / '_nsrdb_base.py'
            compact = root / 'conf_compact.py'
            skeleton = root / '0.py'
            base.write_text('''
NSRDB_MANIFEST = 'data/weather.json'
NSRDB_VALIDATION_SCENARIO_IDS = ['2018-02-15']
EPISODE_TZ = 'UTC'
EPISODE_PERIODS = 118
EPISODE_START_TIME = '12:00'
EPISODE_FREQ = '5min'
def nsrdb_environment_kwargs(**overrides):
    kw = {'observation_mode': 'physical', 'start_time': EPISODE_START_TIME,
          'periods': EPISODE_PERIODS, 'freq': EPISODE_FREQ, 'tz': EPISODE_TZ,
          'randomize_initial_orientation': True, 'randomize_scenario': True}
    kw.update(overrides)
    return kw
def nsrdb_evaluation_environment_kwargs(**overrides):
    kw = nsrdb_environment_kwargs(randomize_initial_orientation=False,
                                  randomize_scenario=False,
                                  fixed_eval_scenarios=list(NSRDB_VALIDATION_SCENARIO_IDS))
    kw.update(overrides)
    kw.pop('fixed_eval_dates', None)
    return kw
def default_nsrdb_algo_kwargs(**overrides):
    kw = dict(_base.params['kwargs'])
    kw.update({'discount': 1., 'reward_scale':100., 'min_alpha':0.001,
               'n_epochs':2000, 'n_initial_exploration_steps':8000, 'n_train_repeat':2})
    kw.update(overrides)
    return kw
''', encoding='utf-8')
            compact.write_text("""params = build_nsrdb_params('x', 'y',
    algo_kwargs={'discount':1.},
    policy_params_kwargs={'hidden_layer_sizes':(64,64)},
    q_params_kwargs={'hidden_layer_sizes':(64,64)},
    train_env_kwargs={'observation_mode':'compact', 'randomize_initial_orientation':True},
    eval_env_kwargs={'observation_mode':'compact', 'fixed_eval_scenarios':None},
    replay_pool_params_kwargs={'kwargs':{'max_size':50000}})
""", encoding='utf-8')
            skeleton.write_text("params={'kwargs': {'discount': 0.995, 'tau':0.005, 'reward_scale':1.}}",encoding='utf-8')
            with (patch.object(reference, 'ROOT', root),
                  patch.object(reference, '_BASE', base),
                  patch.object(reference, '_COMPACT', compact),
                  patch.object(reference, '_SKELETON', skeleton)):
                cfg = reference.load_config(require_data=False)
            self.assertTrue(cfg.train_env['randomize_initial_orientation'])
            self.assertFalse(cfg.eval_env['randomize_initial_orientation'])
            self.assertTrue(cfg.train_env['randomize_scenario'])
            self.assertFalse(cfg.eval_env['randomize_scenario'])
            self.assertIsNone(cfg.eval_env['fixed_eval_scenarios'])
            self.assertEqual(cfg.algo['min_alpha'], 0.001)
            self.assertEqual(cfg.algo['reward_scale'], 100.)
            self.assertEqual(cfg.algo['replay_size'], 50000)


if __name__ == '__main__':
    unittest.main()
