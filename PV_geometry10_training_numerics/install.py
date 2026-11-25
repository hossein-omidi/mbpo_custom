#!/usr/bin/env python3
"""Install only the two numerical corrections against audited GitHub blobs.
No dependencies, no backups, no changes to data, output directories or Git index.
"""
from pathlib import Path
import argparse
import hashlib
import os
import tempfile

BASE = '3bd5873f01eefe0a7652c6e4444ca296994c75de'
CHANGES = {
    'SAC_workspace/src/agents/sac_agent.py': (
        'ddd43819722e0bae473c3ebcc3e153b54fb08471',
        '77eee8ee022ed81c9524d3e1c0a2402be76d97c6',
        '''        z = mean if deterministic else mean + std * torch.randn_like(mean)
        action = torch.tanh(z)
        if not with_logprob:
            return action, None
        normal_lp = (-0.5 * ((z - mean) / std).square() - log_std - 0.5 * math.log(2. * math.pi)).sum(-1, keepdim=True)''',
        '''        # Retain the sampled standard-normal residual: recovering it from
        # z - mean loses precision when std is small in float32.
        noise = torch.zeros_like(mean) if deterministic else torch.randn_like(mean)
        z = mean if deterministic else mean + std * noise
        action = torch.tanh(z)
        if not with_logprob:
            return action, None
        normal_lp = (-0.5 * noise.square() - log_std - 0.5 * math.log(2. * math.pi)).sum(-1, keepdim=True)'''),
    'MBPO_workspace/src/dynamics.py': (
        '9ab1fed6fbb45b455e5cdf44fc8a2eed69dadf25',
        '63d717f15c1caf086bbda9df67b74c7e44ac4a83',
        '''        skill_power = 1.0-rmse[2]/max(float(persist_rmse[2]),1e-8)
        skill_all = 1.0-np.linalg.norm(rmse)/max(float(np.linalg.norm(persist_rmse)),1e-8)''',
        '''        # Floor only the denominator, not the persistence error in the
        # numerator. A strictly positive score must mean strictly lower RMSE.
        p_power, e_power = float(persist_rmse[2]), float(rmse[2])
        p_all, e_all = float(np.linalg.norm(persist_rmse)), float(np.linalg.norm(rmse))
        skill_power = (p_power-e_power)/max(p_power,1e-8)
        skill_all = (p_all-e_all)/max(p_all,1e-8)'''),
    'MBPO_workspace/workflows/06_model_validation.py': (
        '863a093c3055cf3d763f2f01d540f9548805da9f',
        'f506f02525feb37fe72e1365a73fdeed87c08899',
        '''            "skill_vs_persistence": float(1.0 - rmse / max(persistence_rmse, 1e-12)),''',
        '''            # Same sign-preserving regularized RMSE improvement as the training gate.
            "skill_vs_persistence": float((persistence_rmse - rmse) / max(persistence_rmse, 1e-8)),'''),
    'PV_experiments/scientific_checks.py': (
        'c546b70915eaa719b03e60e7e1dcc0905dc29cbc',
        '4671277679331aea3c71af6b937b401f89f05949',
        '''            # Reproduce the pinned diagnostic convention, including its epsilon floor.
            same([1-calc[0]/max(calc[1],1e-12),calc[0]/max(calc[4],1e-12)],''',
        '''            # Reproduce the sign-preserving skill convention and SD-ratio floor.
            same([(calc[1]-calc[0])/max(calc[1],1e-8),calc[0]/max(calc[4],1e-12)],'''),
}
TEST = 'geometry10_tests/test_training_numerics.py'

def blob(data):
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()

def transform(path, data):
    before, after, old, new = CHANGES[path]
    if blob(data) == after:
        compile(data, path, 'exec')
        return data
    if blob(data) != before:
        raise ValueError(f'{path}: unexpected Git blob {blob(data)} (expected {before} or {after})')
    old, new = old.encode(), new.encode()
    if data.count(old) != 1:
        raise ValueError(f'{path}: expected exactly one replacement anchor')
    result = data.replace(old, new, 1)
    if blob(result) != after:
        raise ValueError(f'{path}: transformed file does not match verified Git blob')
    compile(result, path, 'exec')
    return result

def install(root, apply=False):
    root = Path(root).resolve()
    if not (root/'mbpo/geometry10.py').is_file() or not (root/'geometry10_tests').is_dir():
        raise ValueError('Expected complete Geometry10 checkout')
    test_src = Path(__file__).parent/'test_training_numerics.py'
    payload = test_src.read_bytes()
    compile(payload, TEST, 'exec')
    pending = []
    for name in CHANGES:
        p = root/name
        original = p.read_bytes()
        revised = transform(name, original)
        if original != revised:
            pending.append((p, original, revised))
        print(f'{name}: {"update" if original != revised else "already installed"}')
    target = root/TEST
    previous = target.read_bytes() if target.exists() else None
    if previous is not None and previous != payload:
        raise ValueError(f'{TEST}: unexpected existing file')
    if previous is None:
        pending.append((target, None, payload))
    # Validate every file and stage all writes before changing the checkout.
    if not apply:
        print(f'DRY RUN: {len(pending)} files would change; all four source blobs verified.')
        return
    staged = []
    try:
        for p, old, data in pending:
            fd, tmp = tempfile.mkstemp(prefix='.pv-numerics-', dir=p.parent)
            with os.fdopen(fd,'wb') as f:
                f.write(data); f.flush(); os.fsync(f.fileno())
            os.chmod(tmp, p.stat().st_mode & 0o777 if p.exists() else 0o644)
            staged.append((p, old, Path(tmp)))
        for p, old, _ in staged:
            if (p.read_bytes() if p.exists() else None) != old:
                raise RuntimeError(f'{p} changed concurrently; aborting')
        for p, _, tmp in staged:
            os.replace(tmp,p)
    finally:
        for _, _, tmp in staged:
            tmp.unlink(missing_ok=True)
    print(f'APPLIED: {len(pending)} files. Run the repository tests before training.')
    print('Training identity has changed; use a fresh experiment name for new training.')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='.')
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--dry-run', action='store_true')
    group.add_argument('--apply', action='store_true')
    args = p.parse_args()
    install(args.root, args.apply)
