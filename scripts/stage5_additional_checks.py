#!/usr/bin/env python3
"""Additional independent, read-only Stage 5 scientific evidence checks.

Only Python standard library. No Torch/checkpoint/simulator imports. These checks
strengthen *artifact validation*, not SAC training or independently reproduce
an optimizer step or full pvlib power calculations.
"""
from __future__ import annotations
import datetime as dt
import math
import re
import statistics
from collections import Counter, defaultdict

METHODS = ('sac', 'mbpo', 'fixed_no_motion')
ORACLE = 'poa_instant_ideal'
FIVE_MINUTES = dt.timedelta(minutes=5)
REFERENCE_START = dt.time(12, 5)
WEATHER_FIELDS = ('solar_zenith_deg','solar_azimuth_deg','dni_wm2','dhi_wm2',
                  'ghi_wm2','ambient_temperature_c','wind_speed_m_s')


def parse_utc(text, context='timestamp'):
    if not isinstance(text,str) or not text.strip():
        raise ValueError(f'{context}: missing UTC timestamp')
    try:
        parsed = dt.datetime.fromisoformat(text.strip().replace('Z','+00:00'))
    except ValueError as e:
        raise ValueError(f'{context}: invalid ISO timestamp {text!r}: {e}') from e
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise ValueError(f'{context}: timestamp must have explicit UTC offset: {text!r}')
    return parsed.astimezone(dt.timezone.utc)


def check_clock(rows, scenario_id, label, *, require_start=True):
    """Validate exactly 117 post-step times at five-minute intervals.

    `rows` must already be sorted by integer step. For the pinned 12:00 UTC,
    118-timestamp MDP, actions are at 12:05,12:10,...,21:45 UTC.
    """
    if len(rows) != 117:
        raise ValueError(f'{label}: expected 117 action steps, found {len(rows)}')
    times = [parse_utc(r.get('utc_iso'),f'{label}/step {i+1}') for i,r in enumerate(rows)]
    if require_start and times[0].timetz().replace(tzinfo=None) != REFERENCE_START:
        raise ValueError(f'{label}: expected first post-step UTC 12:05, found {times[0].isoformat()}')
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}',str(scenario_id)):
        try: day=dt.date.fromisoformat(str(scenario_id))
        except ValueError as e: raise ValueError(f'{label}: invalid scenario date') from e
        if any(t.date()!=day for t in times):
            raise ValueError(f'{label}: UTC day differs from scenario ID {scenario_id}')
    for i in range(1,len(times)):
        if times[i]-times[i-1]!=FIVE_MINUTES:
            raise ValueError(f'{label}: non-uniform timestep at step {i+1}: {times[i]-times[i-1]}')
    return tuple(times)


def physical_weather_parity(rows_by_key, paired_keys, *, methods=METHODS, allow_missing=False):
    """Compare solar geometry and weather at paired step indices across methods.

    This checks the *recorded inputs* only; not an independent PVWatts
    recomputation from raw weather data.
    """
    count=0
    for pair in paired_keys:
        rs=[rows_by_key.get((*pair,m)) for m in methods]
        if allow_missing and any(x is None for x in rs): continue
        if any(x is None for x in rs):raise ValueError(f'Missing recorded weather path for {pair}')
        if any(len(x)!=117 for x in rs):raise ValueError(f'Incomplete weather paths for {pair}')
        for step in range(117):
            for field in WEATHER_FIELDS:
                first=rs[0][step].get(field)
                if first is None or first=='':raise ValueError(f'Missing {field} for {pair} step {step+1}')
                value=float(first)
                if not math.isfinite(value):raise ValueError(f'Invalid {field} for {pair}')
                for other in rs[1:]:
                    raw=other[step].get(field)
                    if raw is None or raw=='':raise ValueError(f'Missing {field} for {pair}')
                    val=float(raw)
                    if not math.isfinite(val) or not math.isclose(value,val,abs_tol=1e-7,rel_tol=1e-10):
                        raise ValueError(f'Paired physical weather mismatch {field}: {pair} step {step+1}')
            count+=1
    return {'paired_weather_step_groups':count,'compared_fields':list(WEATHER_FIELDS),
            'interpretation':'Recorded-weather parity only; not independent physics verification'}


def validate_oracle(episodes_rows, detailed_rows, pairs, *, tol=3e-5):
    """Gross-only instantaneous POA oracle checks (separate from action contract).

    Verifies time/weather, power-to-energy, episode gross totals, and oracle
    instantaneous power >= all recorded feasible powers. Does NOT prove that
    the oracle is actuator feasible, or independently solve the POA optimum.
    """
    pair_keys={(p['scenario_id'],str(p['seed'])) for p in pairs}
    erows={(r['scenario_id'],str(r['seed'])):r for r in episodes_rows if r.get('method')==ORACLE}
    oracle_by=defaultdict(list); feasible_by=defaultdict(list)
    for r in detailed_rows:
        key=(r['scenario_id'],str(r['seed']))
        if r['method']==ORACLE:oracle_by[key].append(r)
        elif r['method'] in METHODS: feasible_by[(*key,r['method'])].append(r)
    if not erows and not oracle_by:
        return {'status':'NOT_SUPPLIED','note':'No POA data; no oracle conclusion'}
    if set(erows)!=pair_keys or set(oracle_by)!=pair_keys:
        raise ValueError('Oracle episodes and detailed trajectories must cover every paired day')
    sample_count=0
    for key in sorted(pair_keys):
        rows=sorted(oracle_by[key],key=lambda x:int(x['step']))
        if [int(r['step']) for r in rows]!=list(range(1,118)):
            raise ValueError(f'Oracle incomplete steps for {key}')
        clock=check_clock(rows,key[0],f'Oracle {key}')
        other_rows=[]
        for m in METHODS:
            candidates=sorted(feasible_by[(*key,m)],key=lambda x:int(x['step']))
            if len(candidates)!=117:raise ValueError(f'Oracle requires complete {m} day {key}')
            if clock!=check_clock(candidates,key[0],f'{m} {key}'):
                raise ValueError(f'Oracle clock mismatch for {key}/{m}')
            other_rows.append(candidates)
        gross=0.
        for i,r in enumerate(rows):
            if r.get('power_model')!='pvwatts_sapm':raise ValueError(f'Oracle wrong/missing PVWatts label {key}/{i}')
            power=float(r['power_w']);energy=float(r['energy_kwh'])
            if not all(math.isfinite(v) for v in (power,energy)):
                raise ValueError(f'Nonfinite oracle power/energy {key}/{i}')
            if not math.isclose(energy,power/12000.,abs_tol=1e-7,rel_tol=1e-6):
                raise ValueError(f'Oracle power-to-energy mismatch {key}/{i}')
            for m,rr in zip(METHODS,other_rows):
                for field in WEATHER_FIELDS:
                    a=float(r[field]);b=float(rr[i][field])
                    if not all(math.isfinite(v) for v in (a,b)) or not math.isclose(a,b,abs_tol=1e-7,rel_tol=1e-10):
                        raise ValueError(f'Oracle weather mismatch {field} {key}/{i}/{m}')
                if power + max(1e-5,1e-7*abs(float(rr[i]['power_w']))) < float(rr[i]['power_w']):
                    raise ValueError(f'Oracle below {m} instantaneous power at {key}/{i}')
            for field in ('action_tilt','action_azimuth','net_step_kwh'):
                if r.get(field) not in ('',None,'None','nan'):
                    raise ValueError(f'Gross-only oracle incorrectly has {field} {key}/{i}')
            gross+=energy;sample_count+=1
        reported=float(erows[key]['gross_energy_kwh'])
        if not math.isfinite(reported) or not math.isclose(gross,reported,abs_tol=tol,rel_tol=1e-6):
            raise ValueError(f'Oracle episode gross-total mismatch {key}')
        if erows[key].get('net_energy_kwh') not in (None,'','None','nan'):
            raise ValueError(f'Oracle must not report net energy {key}')
    return {'status':'PASS_RECORDED_CONSISTENCY_ONLY','oracle_trajectories':len(oracle_by),
            'oracle_steps':sample_count,'not_checked':'Recomputed PVWatts optimum, actuator feasibility, net energy'}


def weighted_crosscheck(pairs, selection, summary, *, episodes_rows=None, abs_tol=5e-8):
    """Reconstruct design-weighted point estimates from the published sampling plan.

    Each selected bin has one sampled scenario, with weight population_days.
    Not a confidence interval; one observation per bin cannot estimate within-bin
    sampling variance without extra modeling or assumptions.
    """
    strata=selection.get('strata')
    if not isinstance(strata,list) or not strata:raise ValueError('Missing selection strata')
    weights={};total=0
    for s in strata:
        sid=str(s['selected_scenario_id'])
        if sid in weights:raise ValueError('Duplicated selected scenario ID in sampling plan')
        n=int(s['population_days']);ns=int(s['sampled_days'])
        if n<1 or ns!=1:raise ValueError('Only one sampled date per stratum is supported')
        weights[sid]=n;total+=n
    if total!=int(selection['population_count']):
        raise ValueError('Population weights do not sum to population_count')
    per_day=defaultdict(list)
    for p in pairs:per_day[p['scenario_id']].append(p)
    if set(per_day)!=set(weights):raise ValueError('Selection-plan dates and episode IDs differ')
    result={'weight_sum':total,'selected_count':len(weights),'weighted_means_kwh':{},'weighted_differences_kwh':{}}
    for method in METHODS:
        key=f'{method}_net_kwh'
        mean=sum(weights[sid]*statistics.mean(r[key] for r in rr) for sid,rr in per_day.items())/total
        result['weighted_means_kwh'][method]=mean
        expected=float(summary['methods'][method]['mean_net_kwh'])
        if not math.isclose(mean,expected,abs_tol=abs_tol,rel_tol=1e-8):
            raise ValueError(f'Weighted {method} mean differs from official summary: {mean} vs {expected}')
    if episodes_rows is not None:
        oracle=defaultdict(list)
        for row in episodes_rows:
            if row.get('method')==ORACLE:
                oracle[row['scenario_id']].append(float(row['gross_energy_kwh']))
        if oracle:
            if set(oracle)!=set(weights):
                raise ValueError('Oracle energy day IDs differ from selection-plan IDs')
            oracle_mean=sum(weights[k]*statistics.mean(oracle[k]) for k in oracle)/total
            expected=float(summary['methods'][ORACLE]['mean_gross_kwh'])
            if not math.isclose(oracle_mean,expected,abs_tol=abs_tol,rel_tol=1e-8):
                raise ValueError('Weighted POA oracle gross mean differs from official summary')
            result['weighted_oracle_gross_kwh']=oracle_mean
    for a,b in (('mbpo','fixed_no_motion'),('mbpo','sac'),('sac','fixed_no_motion')):
        name=f'{a}_minus_{b}'
        val=result['weighted_means_kwh'][a]-result['weighted_means_kwh'][b]
        result['weighted_differences_kwh'][name]=val
        recorded=summary['paired_net_kwh'][name]['mean_difference_kwh']
        if not math.isclose(val,float(recorded),abs_tol=abs_tol,rel_tol=1e-8):
            raise ValueError(f'Weighted {name} paired difference differs from summary')
    return result


def training_provenance_check(history, checkpoint_info, stage4_history=None):
    """Compare CSV with Stage 4 reported checkpoint stats; not a checkpoint reload."""
    errs=[]
    if history.get('episodes')!=int(checkpoint_info['epoch']):errs.append('epoch mismatch')
    reported=float(history['final_alpha'])
    if stage4_history:
        expected=float(stage4_history['last_alpha'])
        if not math.isclose(reported,expected,rel_tol=1e-5,abs_tol=1e-7):
            errs.append(f'FINAL_ALPHA_MISMATCH: CSV {reported:.10f} vs Stage4 checkpoint {expected:.10f}')
    return {'status':'FAIL_CHECKPOINT_HISTORY_MISMATCH' if errs else 'MATCHES_REPORTED_STAGE4_SNAPSHOT',
            'issues':errs,'limitation':'Actual checkpoint byte/optimizer verification requires trusted local .pt and provenance script'}


def replay_recency_diagnostic(training_csv_rows, capacity=50000, horizon=117):
    """Describe end-of-run FIFO coverage, without asserting a causal replay bug."""
    keep=min(len(training_csv_rows),math.ceil(capacity/horizon))
    full=[r['scenario_id'] for r in training_csv_rows]
    recent=full[-keep:]
    def season(s):
        try: month=int(s[5:7])
        except Exception:return 'unknown'
        if month in (12,1,2):return 'winter'
        if month in (3,4,5):return 'spring'
        if month in (6,7,8):return 'summer'
        return 'autumn'
    a=Counter(map(season,full));b=Counter(map(season,recent))
    all_seasons=sorted(set(a)|set(b))
    tv=sum(abs(a[k]/len(full)-b[k]/len(recent)) for k in all_seasons)/2
    return {'replay_capacity':capacity,'at_most_recent_episodes_covered':keep,
            'approximate_distinct_recent_scenarios':len(set(recent)),
            'recent_episodes_by_season':dict(b),'all_episodes_by_season':dict(a),
            'recent_vs_full_season_total_variation':tv,
            'warning':'This is an approximate FIFO episode-window diagnostic, not proof of replay bias or overfitting'}


def train_eval_overlap(training_ids, evaluation_ids):
    """Exact recorded scenario-ID set intersection (no speculative date coercion)."""
    train={str(x).strip() for x in training_ids}
    eval_set={str(x).strip() for x in evaluation_ids}
    if not train or not eval_set or '' in train or '' in eval_set:
        raise ValueError('Missing training/evaluation scenario IDs')
    overlap=sorted(train & eval_set)
    return {'count':len(overlap),'sample_ids':overlap[:40],
            'limitation':'Exact ID overlap only; does not prove independent meteorological years or training-seed generality'}
