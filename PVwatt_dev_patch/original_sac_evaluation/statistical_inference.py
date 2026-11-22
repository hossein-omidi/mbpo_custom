"""Finite-population uncertainty for stratified NSRDB dates.

One date sampled per temporal bin does NOT identify within-bin variance.
Exact confidence intervals require independent sampling repetitions or a
replicated design with >=2 selections per stratum. All CIs describe the
specified eligible manifest, not new unseen years or model-training seeds.
"""
from collections import defaultdict
import math
import numpy as np


def day_descriptives(values):
    """Descriptive across-day variation, never a confidence interval."""
    v=np.asarray(list(values),float)
    if not len(v) or not np.isfinite(v).all():raise ValueError('Nonempty finite daily values required')
    return {'n_days':len(v),'mean':float(v.mean()),'sd_across_days':float(v.std(ddof=1)) if len(v)>1 else None,
            'min':float(v.min()),'median':float(np.median(v)),'max':float(v.max())}


def stratified_point_estimate(day_values,plan):
    """Horvitz-Thompson / post-stratum exact expansion for one uniform draw per bin."""
    strata=plan['strata'];ids=[s['selected_scenario_id'] for s in strata]
    if len(ids)!=len(set(ids)):
        raise ValueError('Duplicate selected day in the sampling plan')
    if set(ids)!=set(day_values):raise ValueError('All and only selected days must have values')
    w=np.asarray([float(s['population_days']) for s in strata],float)
    v=np.asarray([day_values[sid] for sid in ids],float)
    if not np.isfinite(v).all() or not np.isfinite(w).all() or np.any(w<1):
        raise ValueError('Invalid weights or finite daily data')
    if not math.isclose(float(w.sum()),float(plan['population_count'])):
        raise ValueError('Population weights mismatch')
    return float(np.dot(w,v)/w.sum())


def paired_day_differences(left,right):
    if set(left)!=set(right):raise ValueError('Pairing must match exact scenario IDs')
    return {sid:float(left[sid])-float(right[sid]) for sid in left}


def replicated_sampling_ci(repetitions, confidence=0.95):
    """CI based on independent selection-plan replicates, NOT repeated policy RNG.

    Repetitions must be independent *date selections* from the same population
    frame with the same frozen policies. Values are already design-weighted
    estimator outputs. Student t interval on repeated selection estimates.
    """
    v=np.asarray(repetitions,float)
    if len(v)<2 or not np.isfinite(v).all():raise ValueError('At least two independent selection experiments required')
    if not 0<confidence<1:raise ValueError('Invalid confidence')
    try:
        from scipy.stats import t
        crit=float(t.ppf((1+confidence)/2,len(v)-1))
    except ImportError as exc:
        raise RuntimeError('SciPy is necessary for Student-t confidence intervals') from exc
    sd=float(v.std(ddof=1));se=sd/math.sqrt(len(v));mu=float(v.mean())
    return {'mean_across_selection_replicates':mu,'sampling_sd':sd,'standard_error':se,
            'ci_lower':mu-crit*se,'ci_upper':mu+crit*se,'confidence':confidence,
            'n_independent_selection_replicates':int(len(v)),
            'interval_type':'Student t across independent selection plans; approximation'}


def day_summary_from_paths(paths,method,metric):
    """Average stochastic-policy seeds within each scenario before inference."""
    data=defaultdict(list)
    seen=set()
    for p in paths:
        if p['method']==method:
            sid=str(p['scenario_id'])
            key=(sid,int(p['seed']))
            if key in seen:
                raise ValueError('Duplicate method/day/seed evaluation: '+str(key))
            seen.add(key)
            value=p[metric]
            if value is None or not np.isfinite(float(value)):
                raise ValueError('Metric unavailable/nonfinite for method '+method)
            data[sid].append(float(value))
    return {sid:float(np.mean(x)) for sid,x in data.items()}


def build_valid_statistical_summary(paths,plan,method_names=('mbpo','sac','fixed_no_motion','poa_instant_ideal')):
    found={p['method'] for p in paths}; methods=[x for x in method_names if x in found]
    if not methods:raise ValueError('No visible methods')
    selected={s['selected_scenario_id'] for s in plan['strata']}
    # Each day and controller must have the SAME evaluation seed set.
    # More action seeds on one weather day are not more independent weather days.
    by_method_seed={}
    for p in paths:
        if p['method'] not in methods:continue
        sid=str(p['scenario_id']);seed=int(p['seed']);method=p['method']
        if sid not in selected:raise ValueError('Unexpected scenario in paired analysis: '+sid)
        key=(method,sid)
        if seed in by_method_seed.setdefault(key,set()):
            raise ValueError('Duplicate method/day/seed evaluation: '+str((method,sid,seed)))
        by_method_seed[key].add(seed)
    all_seed_sets=[by_method_seed.get((m,sid),set()) for m in methods for sid in selected]
    if not all_seed_sets or not all_seed_sets[0] or any(x!=all_seed_sets[0] for x in all_seed_sets):
        raise ValueError('Unmatched evaluation days or policy seeds')
    census=all(int(s['population_days'])==1 for s in plan['strata'])
    from SAC_workspace.src.evaluation.stratified_statistics import _time_grid_note
    report={'population':plan['population_count'],'selected_days':plan['selected_count'],
            'sampling_design_variance_status':('exact_census_zero' if census else 'not_estimable_one_day_per_nontrivial_bin'),
            'estimand':'Mean energy per ' + _time_grid_note(plan) + ' over the eligible finite manifest',
            'uncertainty':('Exhaustive eligible-manifest census: no finite-population sampling error.' if census else
                           'Sampling variance not estimable from one observed date per nontrivial temporal bin.'),
            'methods':{},'paired_gaps':{}}
    for m in methods:
        key='gross_energy_kwh' if m=='poa_instant_ideal' else 'net_energy_kwh'
        x=day_summary_from_paths(paths,m,key)
        report['methods'][m]={'metric':key,'weighted_mean':stratified_point_estimate(x,plan),
                              **day_descriptives(x.values()),'confidence_interval_95':None,
                              'design_sampling_variance_kwh2':0.0 if census else None}
    for a,b in (('mbpo','sac'),('mbpo','fixed_no_motion'),('sac','fixed_no_motion')):
        if a in methods and b in methods:
            x=day_summary_from_paths(paths,a,'net_energy_kwh');y=day_summary_from_paths(paths,b,'net_energy_kwh')
            d=paired_day_differences(x,y)
            report['paired_gaps'][a+'_minus_'+b]={'weighted_mean_difference_kwh':stratified_point_estimate(d,plan),
                                                  **day_descriptives(d.values()),'confidence_interval_95':None,
                                                  'design_sampling_variance_kwh2':0.0 if census else None}
    for a in ('mbpo','sac','fixed_no_motion'):
        if a in methods and 'poa_instant_ideal' in methods:
            x=day_summary_from_paths(paths,a,'gross_energy_kwh')
            y=day_summary_from_paths(paths,'poa_instant_ideal','gross_energy_kwh')
            mean_oracle=stratified_point_estimate(y,plan)
            report['paired_gaps'][a+'_gross_over_poa_oracle']={'ratio_of_weighted_means':stratified_point_estimate(x,plan)/mean_oracle if mean_oracle>0 else None,
                                                               'confidence_interval_95':None}
    return report
