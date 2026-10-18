import unittest
import math
from SAC_workspace.src.evaluation.statistical_inference import (
    stratified_point_estimate,paired_day_differences,day_descriptives,
    replicated_sampling_ci,build_valid_statistical_summary)

class VerifiedInference(unittest.TestCase):
    def setUp(self):
        self.plan={'population_count':5,'selected_count':2,'strata':[
            {'selected_scenario_id':'2020-01-01','population_days':2},
            {'selected_scenario_id':'2020-01-12','population_days':3}]}
    def test_exact_expansion_and_paired(self):
        a={'2020-01-01':1,'2020-01-12':4};b={'2020-01-01':2,'2020-01-12':2}
        self.assertAlmostEqual(stratified_point_estimate(a,self.plan),2.8)
        self.assertAlmostEqual(stratified_point_estimate(paired_day_differences(a,b),self.plan),0.8)
        with self.assertRaises(ValueError):paired_day_differences(a,{'2020-01-01':1})
    def test_one_day_has_undefined_sd_not_zero(self):
        self.assertIsNone(day_descriptives([3])['sd_across_days'])
        with self.assertRaises(ValueError):replicated_sampling_ci([4.0])
    def test_repeated_selections_student_t(self):
        x=replicated_sampling_ci([2,4,6,8]);self.assertEqual(x['n_independent_selection_replicates'],4)
        self.assertAlmostEqual(x['mean_across_selection_replicates'],5)
        self.assertLess(x['ci_lower'],5);self.assertGreater(x['ci_upper'],5)
    def test_oracle_is_gross_only_and_no_fake_ci(self):
        ps=[]
        for sid in ['2020-01-01','2020-01-12']:
            for m in ['sac','poa_instant_ideal']:
                ps.append({'scenario_id':sid,'seed':42,'method':m,
                           'gross_energy_kwh':2 if m=='sac' else 4,
                           'net_energy_kwh':2 if m=='sac' else None})
        r=build_valid_statistical_summary(ps,self.plan)
        self.assertEqual(r['methods']['poa_instant_ideal']['metric'],'gross_energy_kwh')
        self.assertIsNone(r['methods']['sac']['confidence_interval_95'])
        self.assertAlmostEqual(r['paired_gaps']['sac_gross_over_poa_oracle']['ratio_of_weighted_means'],.5)

if __name__=='__main__':unittest.main()
