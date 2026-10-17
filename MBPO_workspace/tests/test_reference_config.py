import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from SAC_workspace.src.config.config import SACConfig
from MBPO_workspace.src import config as mbconfig


class ReferenceConfigTests(unittest.TestCase):
    def test_reads_model_knobs_from_original_modules_without_importing_ray(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            skeleton=root/'0.py'
            skeleton.write_text("params = {'kwargs': {'real_ratio': 0.1, 'num_networks': 7, 'num_elites': 5, 'hidden_dim': 200, 'max_model_rollout_length': 5, 'model_train_freq': 117, 'rollout_batch_size': 400, 'model_retain_epochs': 5}}")
            base=root/'_nsrdb_base.py'
            base.write_text("def default_nsrdb_algo_kwargs(**overrides):\n"
                            "    kw = dict()\n"
                            "    kw.update({'real_ratio': 0.75, 'max_model_rollout_length': 1})\n"
                            "    kw.update(overrides)\n"
                            "    return kw\n")
            compact=root/'conf_compact.py'
            compact.write_text("params = build_nsrdb_params('ver', 'compact', "
                               "algo_kwargs={'real_ratio': 0.85, 'num_networks': 5, "
                               "'num_elites': 3, 'hidden_dim': 64, "
                               "'model_retain_epochs': 4})\n")
            sac=SACConfig(train_env={},eval_env={},algo={},manifest=root/'manifest',config_paths=tuple())
            with patch.object(mbconfig,'_SKELETON',skeleton),patch.object(mbconfig,'_BASE',base),\
                 patch.object(mbconfig,'_COMPACT',compact),patch.object(mbconfig,'load_sac_config',return_value=sac):
                cfg=mbconfig.load_config(require_data=False)
            self.assertEqual(cfg.model.rollout_length,1)
            self.assertEqual(cfg.model.num_models,5)
            self.assertEqual(cfg.model.num_elites,3)
            self.assertEqual(cfg.model.rollout_batch_size,400)
            self.assertEqual(cfg.model.model_retain_epochs,4)
            self.assertAlmostEqual(cfg.model.real_ratio,.85)

if __name__=='__main__':unittest.main()
