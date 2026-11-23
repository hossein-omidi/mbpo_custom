#!/usr/bin/env python3
"""Guarded local-only patch for audited PostProcessing Monte Carlo branch.

The installer verifies EXACT current GitHub blob hashes, checks every anchor,
validates syntax, and stages all files before modifying anything. It never
creates backups, commits, pushes, or rewrites SAC/MBPO training algorithms.
"""
from __future__ import annotations
import argparse,ast,difflib,hashlib,os,tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
NEW=ROOT/'new_files'
MARKER='# POSTPROCESSING_MC_CURVE_STATS_AND_VERIFY_FIX_V1'
EXPECTED={
 'MBPO_workspace/src/sample_complexity.py':'8a641848721e298aa477ad5419d1fef847d19ee6',
 'MBPO_workspace/workflows/09_monte_carlo_evaluate.py':'967fd40f534703c1e28d7472db6dc14bd062e864',
 'MBPO_workspace/src/plots.py':'87e01b08c19054c335e8665638a3fa13ecbf78c0',
 'PostProcessing_MC_patch/tools/verify_mc_environment.py':'68f28f2f49c603dd04cf54c5d374ab146149eddc',
}
# Do not modify these: verify paired trainer/split/physics versions before
# producing new paper figures, avoiding accidental mixing of code branches.
REQUIRED_UNCHANGED={
 'SAC_workspace/src/training/experiment_protocol.py':'caf11e2f6801e98ac6c8ed817117f3176ba0db8c',
 'SAC_workspace/src/training/trainer.py':'99ef8bde096738a3c9a457ef259440037c35bd35',
 'MBPO_workspace/src/trainer.py':'31611b94ad92a0761d58061ab2cb158419189e9f',
 'MBPO_workspace/src/dynamics.py':'edff92372c70637058138023e99eb3a3efdc90a1',
}


def blob_sha(b):return hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()


def once(s,original,replacement,where):
    n=s.count(original)
    if n!=1:
        raise ValueError(f'{where}: expected one exact source anchor; found {n}: {original[:90]!r}')
    return s.replace(original,replacement,1)


def patch_verify(s):
    # This is a program invoked as a script path from the repo root. CPython
    # sets sys.path[0] to its tools/ directory, not the repository root.
    # Correct the CLI import contract, not the PV model or dependency check.
    s=once(s,
'''    args=parser.parse_args(argv)
    from mbpo.env.trajectory_uncertainty import normalized_spec''',
'''    args=parser.parse_args(argv)
    # Direct-file execution does not automatically add the repo root to
    # sys.path. Validate the expected checkout layout before importing the
    # project. `python -m PostProcessing_MC_patch.tools.verify_mc_environment`
    # also works and does not need this adjustment.
    import sys
    repo_root=Path(__file__).resolve().parents[2]
    if not (repo_root/'mbpo'/'env').is_dir() or not (repo_root/'SAC_workspace'/'src').is_dir():
        raise RuntimeError(f'This verifier must be placed inside a complete PostProcessing checkout: {repo_root}')
    if str(repo_root) not in sys.path:
        sys.path.insert(0,str(repo_root))
    from mbpo.env.trajectory_uncertainty import normalized_spec''',
    'verifier script import path')
    if s.startswith('#!/'):
        head,rest=s.split('\n',1)
        return head+'\n'+MARKER+'\n'+rest
    return MARKER+'\n'+s


def patch_sample(s):
    # The statistical comparison is only valid if every recorded training
    # episode belongs to the frozen training split (never validation/test).
    s=once(s,
'''    episodes = _read_csv(root / "episodes.csv")
    if not aggregate or not daily or not references:''',
'''    episodes = _read_csv(root / "episodes.csv")
    allowed_training_days=set(map(str,split.get('train_ids',[])))
    if not allowed_training_days:
        raise ValueError(f'{root}: frozen split has no training scenarios')
    unexpected_training=[str(r.get('scenario_id','')) for r in episodes
                         if str(r.get('scenario_id','')) not in allowed_training_days]
    if unexpected_training:
        raise ValueError(f'{root}: training log includes non-training historical day IDs: '
                         f'{sorted(set(unexpected_training))[:10]}')
    if not aggregate or not daily or not references:''',
    'sample-complexity training split membership')
    s=once(s,
'''    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")''',
'''    # Additional publication-oriented descriptive ±1 SD curves. Separate
    # SD across independent training seeds from SD across historical days.
    # The existing two-way bootstrap 95% CIs remain unchanged and are NOT SD.
    from MBPO_workspace.src.curve_statistics import write_sample_complexity_mean_sd
    report['descriptive_mean_sd_reporting'] = write_sample_complexity_mean_sd(
        out, steps, sac_matrix, mbpo_matrix, target)
    report['bootstrap_interval_caveat'] = (
        'Original two-way bootstrap intervals resample independent training seeds and a '
        'deterministically selected month-stratified set of validation days. They are '
        'resampling sensitivity intervals, not design-validated 95% coverage for '
        'all NSRDB days, future years, or weather-error realizations.')
    report['periodic_validation_weather'] = (
        'One frozen weather realization_id=0 per historical validation day; '
        'descriptive seed/day SD excludes within-day Monte Carlo draws')
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")''',
    'multi-seed sample-complexity summary integration')
    s=once(s,'    ax.fill_between(steps, sac_lo, sac_hi, alpha=0.18)',
       '    ax.fill_between(steps, sac_lo, sac_hi, alpha=0.18, label="SAC 95% two-way bootstrap band")',
       'CI label for SAC')
    s=once(s,'    ax.fill_between(steps, mbpo_lo, mbpo_hi, alpha=0.18)',
       '    ax.fill_between(steps, mbpo_lo, mbpo_hi, alpha=0.18, label="MBPO 95% two-way bootstrap band")',
       'CI label for MBPO')
    s=once(s,'    ax.fill_between(steps, diff_lo, diff_hi, alpha=0.2)',
       '    ax.fill_between(steps, diff_lo, diff_hi, alpha=0.2, label="Paired 95% two-way bootstrap band")',
       'CI label for paired method')
    return MARKER+'\n'+s


def patch_mc_evaluator(s):
    s=once(s,
'''    write_csv(out/'daily_power_mean_std_by_utc_step.csv',clock_rows)''',
'''    write_csv(out/'daily_power_mean_std_by_utc_step.csv',clock_rows)
    # The original file gives mean/SD across M weather realizations for each
    # named historical day. This complementary output pools days and MC draws
    # without treating 195 sequential timestamps as independent observations.
    from MBPO_workspace.src.curve_statistics import aggregate_mc_power_profiles
    pooled_clock_rows=aggregate_mc_power_profiles(clock_rows,args.draws)
    write_csv(out/'mc_pooled_power_mean_sd_by_utc_step.csv',pooled_clock_rows)''',
    'nested MC pooled mean/SD output')
    s=once(s,
'''        'power_output':'daily_power_mean_std_by_utc_step.csv (every selected historical day)',''',
'''        'power_output':'daily_power_mean_std_by_utc_step.csv (every selected historical day)',
        'pooled_power_output':'mc_pooled_power_mean_sd_by_utc_step.csv (equal day and draw weights)',
        'pooled_power_sd_interpretation':(
            'Descriptive SD over tested historical day × assumed weather draw mixture; '
            'not a standard error or a confidence interval and not training-seed SD'),''',
    'nested MC provenance')
    s=once(s,
'''    if not args.skip_figures:
        plot_diagnostics(out,rows,first_day_clock_rows,paired)''',
'''    if not args.skip_figures:
        plot_diagnostics(out,rows,first_day_clock_rows,paired)
        from MBPO_workspace.src.curve_statistics import plot_pooled_mc_power_profiles
        plot_pooled_mc_power_profiles(out,pooled_clock_rows)''',
    'MC pooled plot creation')
    return MARKER+'\n'+s


def patch_plots(s):
    # Raw, heterogeneous sampled training returns are not independent
    # evaluation replicas, so do not claim their rolling average is an SD band.
    s=once(s,
"        ax.set_title('Learning curves (weather distributions may differ by sampled episode)')",
"        ax.set_title('Single-run training returns: rolling mean only (not mean ± SD)')",
    'training diagnostic title')
    return MARKER+'\n'+s


PATCHERS={
 'MBPO_workspace/src/sample_complexity.py':patch_sample,
 'MBPO_workspace/workflows/09_monte_carlo_evaluate.py':patch_mc_evaluator,
 'MBPO_workspace/src/plots.py':patch_plots,
 'PostProcessing_MC_patch/tools/verify_mc_environment.py':patch_verify,
}


def plan(repo):
    repo=Path(repo)
    for rel,expected in REQUIRED_UNCHANGED.items():
        p=repo/rel
        if not p.is_file():raise FileNotFoundError(f'Required reviewed source is absent: {p}')
        actual=blob_sha(p.read_bytes())
        if actual!=expected:
            raise ValueError(f'Unexpected {rel} version {actual}; expected {expected}. No modifications made.')
    changed={}
    for rel,fn in PATCHERS.items():
        p=repo/rel
        if not p.is_file():raise FileNotFoundError(p)
        old=p.read_text(encoding='utf-8')
        if old.startswith(MARKER+'\n') or old.startswith('#!/usr/bin/env python3\n'+MARKER+'\n'):
            continue  # Already applied; do not rewrite the user's source.
        actual=blob_sha(old.encode('utf-8'))
        if actual!=EXPECTED[rel]:
            raise ValueError(f'{rel} has unreviewed changes (Git blob SHA {actual}); expected {EXPECTED[rel]}')
        new=fn(old)
        ast.parse(new,filename=str(p))
        changed[rel]=(old,new)
    for p in sorted(NEW.rglob('*.py')):
        rel=p.relative_to(NEW).as_posix()
        target=repo/rel
        new=p.read_text(encoding='utf-8')
        ast.parse(new,filename=rel)
        if target.exists():
            if target.read_text(encoding='utf-8')==new:continue
            raise FileExistsError(f'{rel} exists with different content; refusing overwrite')
        changed[rel]=('',new)
    return changed


def diff(changes):
    result=[]
    for rel,(old,new) in sorted(changes.items()):
        d=f'diff --git a/{rel} b/{rel}\n'
        if not old:d+='new file mode 100644\n'
        d+=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
                                         fromfile='a/'+rel if old else '/dev/null',tofile='b/'+rel))
        result.append(d)
    return ''.join(result)


def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--root',required=True,help='Local PostProcessing repository root')
    pa.add_argument('--apply',action='store_true',help='Write files locally; never commit or create backups')
    pa.add_argument('--dry-run',action='store_true',help='Validate only; default behavior')
    pa.add_argument('--diff-file',default=None,help='Write reviewable Git patch for manual git apply')
    a=pa.parse_args()
    if a.dry_run and a.apply:pa.error('--apply and --dry-run cannot be combined')
    root=Path(a.root).expanduser().resolve()
    if not root.is_dir():raise FileNotFoundError(root)
    changes=plan(root)
    print('Reviewed pipeline/statistics modifications:',len(changes))
    for rel,(old,new) in sorted(changes.items()):print(' ', 'MOD' if old else 'NEW',rel)
    if a.diff_file:
        p=Path(a.diff_file);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(diff(changes),encoding='utf-8');print('Git-applicable patch:',p)
    if not a.apply:
        print('Dry run; no repository modifications. Review the diff before --apply.')
        return
    staged=[]
    try:
        for rel,(_old,new) in sorted(changes.items()):
            dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=dest.parent,
                    prefix='.pipeline_',suffix='.tmp',delete=False) as f:
                f.write(new);staged.append((Path(f.name),dest))
        for src,dest in staged:os.replace(src,dest)
    finally:
        for src,_ in staged:
            if src.exists():src.unlink()
    print('Patch applied locally; no backups created and no Git commands executed.')

if __name__=='__main__':main()
