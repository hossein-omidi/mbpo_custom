"""Regression checks for design-weighted inference and honest reporting."""
import copy
import math
import unittest

from SAC_workspace.src.evaluation.statistical_inference import (
    stratified_point_estimate,build_valid_statistical_summary)
from SAC_workspace.src.evaluation.stratified_statistics import (
    _weighted,monthly_paired_weighted_gains)
from MBPO_workspace.src.evaluation import paired_summary


class StatisticalAuditV41(unittest.TestCase):
    def setUp(self):
        self.days=('2020-01-02','2020-01-20')
        self.plan={
            'protocol':'calendar_stratified_utc_daytime',
            'population_count':4,'selected_count':2,
            'strata':[
                {'selected_scenario_id':self.days[0],'population_days':1,'year':2020,'month':1},
                {'selected_scenario_id':self.days[1],'population_days':3,'year':2020,'month':1},
            ],
        }
        self.paths=[]
        for sid,net in zip(self.days,(1.,5.)):
            for method,gross,n in (('mbpo',net,net),('sac',.5,.5),
                                   ('fixed_no_motion',.2,.2),('poa_instant_ideal',7.,None)):
                self.paths.append({'scenario_id':sid,'seed':42,'method':method,
                                   'gross_energy_kwh':gross,'net_energy_kwh':n,
                                   'movement_cost_kwh':None if n is None else 0.})

    def test_unequal_population_weights(self):
        self.assertAlmostEqual(stratified_point_estimate(dict(zip(self.days,(1.,5.))),self.plan),4.)
        self.assertNotAlmostEqual(4.,(1.+5.)/2.)

    def test_main_summary_is_design_weighted_and_has_no_naive_ci(self):
        report=paired_summary(self.paths,plan=self.plan)
        self.assertAlmostEqual(report['methods']['mbpo']['mean_net_kwh'],4.)
        pair=report['paired_net_kwh']['mbpo_minus_sac']
        self.assertAlmostEqual(pair['mean_difference_kwh'],3.5)
        self.assertEqual(pair['bootstrap_ci_95pct_kwh'],[None,None])
        self.assertIsNone(pair['mean_percent_gain'])
        self.assertAlmostEqual(pair['ratio_of_weighted_means_percent'],700.)
        self.assertIn('not estimated',pair['ci_status'])

    def test_duplicate_and_missing_seed_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            build_valid_statistical_summary(self.paths+[dict(self.paths[0])],self.plan)
        shortened=self.paths[:-1]
        with self.assertRaisesRegex(ValueError,'Unmatched'):
            build_valid_statistical_summary(shortened,self.plan)
        mismatched=copy.deepcopy(self.paths)
        mismatched[0]['seed']=8
        with self.assertRaisesRegex(ValueError,'Unmatched'):
            build_valid_statistical_summary(mismatched,self.plan)

    def test_nonfinite_and_duplicate_strata_rejected(self):
        v=dict(zip(self.days,(1.,5.)))
        bad=copy.deepcopy(self.plan);bad['strata'][1]['selected_scenario_id']=self.days[0]
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            stratified_point_estimate(v,bad)
        with self.assertRaisesRegex(ValueError,'Invalid'):
            stratified_point_estimate({self.days[0]:1.,self.days[1]:float('nan')},self.plan)

    def test_missing_measurement_does_not_change_denominator(self):
        actual=_weighted({self.days[0]:1.},self.plan,n_boot=2)
        self.assertIsNone(actual)
        full=_weighted(dict(zip(self.days,(1.,5.))),self.plan,n_boot=2)
        self.assertAlmostEqual(full['weighted_mean'],4.)
        self.assertEqual(full['represented_days'],4)

    def test_monthly_paired_plot_uses_design_weights(self):
        rows=[]
        for p in self.paths:
            if p['net_energy_kwh'] is not None:
                rows.append({k:p[k] for k in ('method','scenario_id','net_energy_kwh')})
        gains=monthly_paired_weighted_gains(rows,self.plan,'mbpo','sac')
        self.assertAlmostEqual(gains[1],3.5)
        self.assertIsNone(gains[2])

    def test_exact_randomization_distribution_matches_variance_formula(self):
        # Exhaustively enumerate all 2 x 3 possible independent one-per-bin
        # samples. This checks the randomization expectation AND true variance;
        # the true variance still cannot be estimated from one realized plan.
        import itertools
        import numpy as np
        bins=[(0.,2.),(3.,6.,9.)]
        N=sum(len(b) for b in bins)
        possible=[]
        for y0,y1 in itertools.product(*bins):
            possible.append((len(bins[0])*y0+len(bins[1])*y1)/N)
        target=sum(sum(x) for x in bins)/N
        variance=sum((len(b)/N)**2*(1.-1./len(b))*np.var(b,ddof=1)
                     for b in bins)
        self.assertAlmostEqual(float(np.mean(possible)),target)
        self.assertAlmostEqual(float(np.var(possible,ddof=0)),variance)

    def test_exhaustive_census_has_exact_zero_sampling_variance(self):
        census=copy.deepcopy(self.plan)
        census['strata'][1]['population_days']=1
        census['population_count']=2
        result=build_valid_statistical_summary(self.paths,census)
        self.assertEqual(result['sampling_design_variance_status'],'exact_census_zero')
        self.assertEqual(result['methods']['mbpo']['design_sampling_variance_kwh2'],0.0)
        self.assertIsNone(result['methods']['mbpo']['confidence_interval_95'])

if __name__=='__main__': unittest.main()
