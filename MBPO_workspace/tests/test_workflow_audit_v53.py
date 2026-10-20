"""Protect the final evaluation report and independently audit saved energy data."""
from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import csv
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from MBPO_workspace.tests.test_advanced_suite_v5 import sample_paths, setup_dir
from MBPO_workspace.src.evaluation import write_outputs
from MBPO_workspace.src.advanced_suite import render_suite
audit = importlib.import_module("MBPO_workspace.workflows.07_audit_scientific_results").audit


class EvaluationWorkflowContractV53(unittest.TestCase):
    def test_last_report_does_not_expect_nonexistent_plots_key(self):
        workflow=importlib.import_module('MBPO_workspace.workflows.03_evaluate')
        sampling=importlib.import_module('SAC_workspace.src.evaluation.scenario_sampling')
        advanced=importlib.import_module('MBPO_workspace.src.advanced_suite')
        with tempfile.TemporaryDirectory() as td:
            profile=Path(td)/'profile.json'
            profile.write_text(json.dumps({'active_methods':[
                'mbpo','sac','fixed_no_motion','poa_instant_ideal']}))
            weather=types.ModuleType('mbpo.env.nsrdb_weather')
            weather.load_scenario_manifest=lambda _: {'scenarios':[]}
            mbpo=types.ModuleType('mbpo');mbpo.__path__=[]
            env=types.ModuleType('mbpo.env');env.__path__=[]
            cfg=types.SimpleNamespace(sac=types.SimpleNamespace(manifest='fake.json'))
            args=['03_evaluate','--mbpo-checkpoint','a.pt','--sac-checkpoint','b.pt',
                  '--scenarios','2020-01-02','--representation-config',str(profile),
                  '--outdir',str(Path(td)/'evaluation')]
            with patch.dict(sys.modules, {'mbpo':mbpo,'mbpo.env':env,
                                         'mbpo.env.nsrdb_weather':weather}), \
                 patch.object(sys,'argv',args), \
                 patch.object(workflow,'load_config',return_value=cfg), \
                 patch.object(workflow,'evaluate',return_value=[]), \
                 patch.object(workflow,'write_outputs',return_value={'methods':{}}), \
                 patch.object(advanced,'export_detail'), \
                 patch.object(advanced,'render_suite',return_value={
                     'figures':['a.png','b.png'],'skipped':[]}), \
                 patch.object(sampling,'resolve_cli_selection',return_value=(['2020-01-02'],None)):
                output=StringIO()
                with redirect_stdout(output):workflow.main()
                self.assertIn('advanced figures: 2',output.getvalue())
                with patch.object(sys,'argv',args+['--skip-advanced-suite']):
                    output=StringIO()
                    with redirect_stdout(output):workflow.main()
                    self.assertIn('advanced figures: 0',output.getvalue())

    def test_explicit_missing_training_history_errors_early(self):
        workflow=importlib.import_module('MBPO_workspace.workflows.03_evaluate')
        offline=importlib.import_module('MBPO_workspace.workflows.06_plot_suite')
        with tempfile.TemporaryDirectory() as td:
            missing=str(Path(td)/'never.csv')
            for module,args in ((workflow,['03_evaluate','--mbpo-checkpoint','m',
                             '--sac-checkpoint','s','--sac-training-csv',missing]),
                                (offline,['06_plot_suite','--eval-dir',td,
                             '--sac-training-csv',missing])):
                with patch.object(sys,'argv',args),redirect_stderr(StringIO()) as err:
                    with self.assertRaises(SystemExit) as problem:module.main()
                self.assertEqual(problem.exception.code,2)
                self.assertIn('file not found',err.getvalue())


class ReadOnlyNumericalAuditV53(unittest.TestCase):
    def test_audit_paired_weighted_episodes_and_exported_band(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)
            paths=sample_paths()
            setup_dir(folder,paths)
            plan=json.loads((folder/'scenario_selection.json').read_text())
            plan['selected_count']=len(plan['strata'])
            plan['selected_scenario_ids']=[s['selected_scenario_id'] for s in plan['strata']]
            (folder/'scenario_selection.json').write_text(json.dumps(plan))
            write_outputs(paths,folder,plan)
            render_suite(folder)
            result=audit(folder)
            self.assertEqual(result['sampled_days'],2)
            self.assertEqual(result['represented_manifest_days'],4)
            self.assertEqual(result['trajectory_paths'],8)
            self.assertGreater(result['tested_curve_points'],100)
            self.assertEqual(result['training_day_overlap'],{'mbpo':None,'sac':None})
            # Corrupt weighted summary: audit must fail instead of approving it.
            summary_path=folder/'summary.json'
            original=json.loads(summary_path.read_text())
            bad=json.loads(summary_path.read_text())
            bad['methods']['mbpo']['mean_gross_kwh']+=.02
            summary_path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,'Weighted summary gross'):
                audit(folder)
            summary_path.write_text(json.dumps(original))
            # Corrupt observed-day dispersion: cannot pass a styled, wrong band.
            csvpath=folder/'advanced_suite'/'curve_band_data.csv'
            with csvpath.open(newline='') as f:
                rows=list(csv.DictReader(f));fields=list(rows[0])
            target=next(x for x in rows if x['method']=='mbpo' and x['metric']=='power_w')
            target['observed_day_rms_sd']=str(float(target['observed_day_rms_sd'])+1)
            with csvpath.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
            with self.assertRaisesRegex(ValueError,'Plotted descriptive scatter'):
                audit(folder)


if __name__=='__main__':unittest.main()
