"""Unit tests for collector; Python standard library only."""
import importlib.util
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

FILE=Path(__file__).resolve().parents[1]/'sac_mbpo_collect.py'
spec=importlib.util.spec_from_file_location('sac_mbpo_collect',FILE)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class CollectorTests(unittest.TestCase):
    def test_finite_and_nonfinite(self):
        self.assertIsNone(mod.finite_num('nan'))
        self.assertIsNone(mod.finite_num('bad'))
        self.assertEqual(mod.finite_num('0.015'),0.015)
    def test_pair_matching_and_bad_alpha(self):
        c={'status':'READ','metadata':{'epoch':100,'env_steps':11700,'agent_updates':42,
                                    'agent_alpha_effective':0.015}}
        h={'last':{'epoch':'100','env_steps':'11700','updates':'42','alpha':'0.015'}}
        self.assertEqual(mod.compare_pair(c,h)['status'],'MATCH_ON_CHECKED_FIELDS')
        h['last']['alpha']='0.03'
        self.assertEqual(mod.compare_pair(c,h)['status'],'MISMATCH')
    def test_incomplete_checkpoint_is_not_pass(self):
        c={'status':'NOT_DESERIALIZED','note':'disabled','metadata':{}}
        self.assertEqual(mod.compare_pair(c,{'last':{}})['status'],'CHECKPOINT_NOT_READ')
    def test_stage4_mismatch_only_if_checksum_matched(self):
        c={'sha256':'other'}
        self.assertEqual(mod.compare_to_stage4('sac',c,{'last':{'alpha':'0.4'}})['status'],
                         'DIFFERENT_CHECKPOINT_THAN_STAGE4')
        c={'sha256':mod.STAGE4['sac']['sha256']}
        h={'last':{'alpha':'0.022418'},'last100_means':{}}
        self.assertEqual(mod.compare_to_stage4('sac',c,h)['status'],'STAGE4_CHECKPOINT_HISTORY_MISMATCH')
    def test_csv_summary_and_detect_bad_episodes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'episodes.csv'
            p.write_text('epoch,env_steps,updates,scenario_id,alpha,return_kwh,critic_loss\n'
                         '1,117,0,2022-07-03,1.0,0.3,nan\n'
                         '2,234,1,2022-07-04,0.6,0.4,5.2\n')
            result=mod.load_csv_summary(p)
            self.assertEqual(result['rows'],2)
            self.assertEqual(result['distinct_scenario_ids'],2)
            self.assertEqual(result['last100_means']['critic_loss'],5.2)
            self.assertEqual(result['anomalies'],[])
            p.write_text('epoch,env_steps,updates,scenario_id,alpha,return_kwh\n'
                         '1,117,0,2022-01-01,1.0,0.3\n'
                         '3,234,1,2022-01-02,1.0,0.4\n')
            self.assertTrue(mod.load_csv_summary(p)['anomalies'])
    def test_no_source_skips_private_code(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'SAC_workspace/runs/sac_seed0').mkdir(parents=True)
            (root/'MBPO_workspace/runs').mkdir(parents=True)
            (root/'SAC_workspace/src/agents').mkdir(parents=True)
            (root/'SAC_workspace/src/agents/sac_agent.py').write_text('private = True')
            report,z=mod.run(root,no_load_checkpoints=True,include_source=False)
            from zipfile import ZipFile
            with ZipFile(z) as zz:
                self.assertFalse(any(x.startswith('source/') for x in zz.namelist()))
            self.assertEqual(report['privacy']['source_included'],False)

if __name__=='__main__':unittest.main()
