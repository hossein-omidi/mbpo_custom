"""Offline fail-closed tests: no NSRDB, pvlib, or external checkpoint required."""
import copy
import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

HERE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('sac_checkpoint_config_audit',HERE/'scripts/sac_checkpoint_config_audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

class ModelAuditTests(unittest.TestCase):
    def setUp(self):
        candidate=Path('MBPO_workspace/evaluation/pvwatts_seed0_heldout1')
        self.src=candidate if (candidate/'episodes.csv').is_file() else Path('/mnt/data/pv_analysis/pvwatts_seed0_heldout1')

    def test_replay_rewards_pass_on_valid_physical_rows(self):
        n=30;next_states=np.zeros((n,6),dtype=np.float32);next_states[:,2]=.07
        next_states[:,5]=np.arange(n,0,-1,dtype=np.float32)/117
        next_states[-1,5]=0
        replay=dict(capacity=64,count=n,head=n,states=np.zeros((n,6)),
          next_states=next_states,actions=np.zeros((n,2)),rewards=(next_states[:,2]*2000/12000).reshape(-1,1),
          dones=(next_states[:,5]==0).reshape(-1,1).astype(float))
        result=audit._validate_replay(replay,2000,5/60,0.)
        self.assertEqual(result['count'],30)
        broken=copy.deepcopy(replay);broken['rewards'][-1,0]+=.003
        with self.assertRaisesRegex(AssertionError,'Replay reward'):
            audit._validate_replay(broken,2000,5/60,0.)

    def test_replay_terminal_flag_mismatch_rejected(self):
        nxt=np.zeros((1,6));nxt[0,5]=1/117
        replay=dict(capacity=2,count=1,head=1,next_states=nxt,
                    actions=np.zeros((1,2)),rewards=np.zeros((1,1)),dones=np.ones((1,1)))
        with self.assertRaisesRegex(AssertionError,'terminal'):
            audit._validate_replay(replay,2000,5/60,0.)

    def test_replay_saturated_power_does_not_falsely_fail(self):
        nxt=np.zeros((1,6));nxt[0,2]=1.;nxt[0,5]=1/117
        replay=dict(capacity=2,count=1,head=1,next_states=nxt,
                    actions=np.zeros((1,2)),rewards=np.array([[.25]]),dones=np.zeros((1,1)))
        stats=audit._validate_replay(replay,2000,5/60,0.)
        self.assertEqual(stats['unsaturated_power_rows'],0)

    def test_tampered_daily_energy_rejected(self):
        episodes=[dict(scenario_id='2022-01-01',seed='42',method=m,
                gross_energy_kwh=str(1.0 if m!='poa_instant_ideal' else 2.0),
                movement_cost_kwh=('' if m=='poa_instant_ideal' else '0'),
                net_energy_kwh=('' if m=='poa_instant_ideal' else '1'),
                initial_tilt_deg='30',initial_azimuth_deg='180') for m in audit.REQUIRED]
        summary={'methods':{m:{'mean_gross_kwh':1.0,'mean_net_kwh':1.0,'mean_movement_cost_kwh':0.0} for m in audit.REQUIRED},
                 'paired_net_kwh':{f'{a}_minus_{b}':{'mean_difference_kwh':0.0,'ratio_of_weighted_means_percent':0.0} for a,b in [('sac','fixed_no_motion'),('mbpo','fixed_no_motion'),('mbpo','sac')]}}
        summary['methods']['poa_instant_ideal']['mean_gross_kwh']=2.0
        self.assertEqual(audit.sample_weighting(episodes,None,summary)['evaluated_days'],1)
        summary['methods']['sac']['mean_net_kwh']=1.1
        with self.assertRaisesRegex(AssertionError,'Summary sac.mean_net'):
            audit.sample_weighting(episodes,None,summary)

    def test_tampered_percent_gain_rejected(self):
        if not (self.src/'episodes.csv').exists():self.skipTest('Real evaluation not available')
        episodes=audit.read_csv(self.src/'episodes.csv')
        plan=audit.read_json(self.src/'scenario_selection.json')
        summary=audit.read_json(self.src/'summary.json')
        summary['paired_net_kwh']['sac_minus_fixed_no_motion']['ratio_of_weighted_means_percent']+=1.0
        with self.assertRaisesRegex(AssertionError,'gain percentage'):
            audit.sample_weighting(episodes,plan,summary)

    def test_history_step_and_reward(self):
        rows=[dict(epoch=str(i),env_steps=str(117*i),scenario_id=f'2022-01-{i:02d}',
                   return_kwh='1.0',gross_energy_kwh='1.0',movement_cost_kwh='0.0',alpha='.2') for i in (1,2)]
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'history.csv'
            with p.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            self.assertTrue(audit.history_report(p,2,234)['checkpoint_matches_history_row'])
            rows[1]['env_steps']='220'
            with p.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            with self.assertRaisesRegex(AssertionError,'real interaction'):
                audit.history_report(p)

    def test_real_checkpoint_fields_and_sidecars(self):
        algo=dict(gamma=1.0,tau=.005,reward_scale=100.,target_entropy=-2.,min_alpha=.001)
        saved_config={'sac_config_version':1,'algo':algo,'hidden_sizes':(64,64)}
        mdp={'power_model':'legacy','observation_mode':'compact','horizon':117}
        nxt=np.zeros((1,6),dtype=np.float32);nxt[0,5]=0.0
        buffer=dict(capacity=3,count=1,head=1,actions=np.zeros((1,2),dtype=np.float32),
                next_states=nxt,rewards=np.zeros((1,1),dtype=np.float32),dones=np.ones((1,1),dtype=np.float32))
        ag=dict(actor={'weight':torch.ones(2)},critic={'weight':torch.ones(2)},
                log_alpha=torch.tensor([np.log(.2)],dtype=torch.float32), **algo)
        ck=dict(schema=1,hash=audit.canon_hash(saved_config),epoch=1,env_steps=117,
                seed=7,agent=ag,resolved_mdp=mdp,replay=buffer)
        config=dict(sac_config=saved_config,sac_algo=algo,sac_train_env={'movement_penalty':0.})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'latest.pt';torch.save(ck,p)
            (root/'config.json').write_text(json.dumps(saved_config))
            (root/'resolved_mdp.json').write_text(json.dumps(mdp))
            report=audit.inspect_checkpoint(p,'sac',{'power_model':'legacy'},config,2000.)
            self.assertEqual(report['checkpoint_real_steps'],117)
            self.assertTrue(report['sidecar_config.json_validated'])
            self.assertFalse(report['alpha_at_floor'])
            ck['hash']='corrupt';torch.save(ck,p)
            with self.assertRaisesRegex(AssertionError,'configuration'):
                audit.inspect_checkpoint(p,'sac',{'power_model':'legacy'},config,2000.)

    def test_real_eval_recalculation_when_available(self):
        if not (self.src/'episodes.csv').exists():self.skipTest('User evaluation ZIP unavailable in this run')
        result=audit.sample_weighting(audit.read_csv(self.src/'episodes.csv'),
          audit.read_json(self.src/'scenario_selection.json'),audit.read_json(self.src/'summary.json'))
        self.assertEqual(result['evaluated_days'],336)
        self.assertEqual(result['represented_days'],1167)
        self.assertAlmostEqual(result['paired_gains']['sac_minus_fixed_no_motion']['weighted_difference_kwh'],.003329200541939698,places=10)
        self.assertEqual(result['paired_gains']['sac_minus_fixed_no_motion']['winning_days'],136)

if __name__=='__main__':unittest.main()
