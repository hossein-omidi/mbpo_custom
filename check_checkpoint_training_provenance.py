#!/usr/bin/env python3
"""Read-only provenance check for trusted local SAC/MBPO .pt checkpoints.

Only run on checkpoints you created. torch.load(weights_only=False) can execute
pickle contents from untrusted files. Nothing is changed or retrained.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def checksum(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fp:
        while True:
            chunk = fp.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def run(checkpoint, training_csv):
    import torch
    cp = Path(checkpoint)
    csv_path = Path(training_csv)
    with csv_path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('The training CSV has no episode rows')
    data = torch.load(cp, map_location='cpu', weights_only=False)
    a = data['agent']
    alpha = float(torch.exp(torch.as_tensor(a['log_alpha'])).clamp_min(float(a['min_alpha'])).item())
    last = rows[-1]
    checks = {
        'epoch': int(data['epoch']) == int(last['epoch']),
        'env_steps': int(data['env_steps']) == int(last['env_steps']),
        'updates': int(a['updates']) == int(last['updates']),
        'alpha': math.isclose(alpha, float(last['alpha']), rel_tol=1e-5, abs_tol=1e-7),
        'history_rows': int(data['epoch']) == len(rows),
    }
    return {
        'checkpoint_file': str(cp.resolve()),
        'checkpoint_sha256': checksum(cp),
        'training_csv_file': str(csv_path.resolve()),
        'training_csv_sha256': checksum(csv_path),
        'checkpoint_epoch': int(data['epoch']),
        'csv_final_epoch': int(last['epoch']),
        'checkpoint_env_steps': int(data['env_steps']),
        'csv_final_env_steps': int(last['env_steps']),
        'checkpoint_updates': int(a['updates']),
        'csv_final_updates': int(last['updates']),
        'checkpoint_alpha': alpha,
        'csv_final_alpha': float(last['alpha']),
        'checks': checks,
        'status': 'PASS' if all(checks.values()) else 'FAIL_DIFFERENT_RUN_OR_INCONSISTENT_LOG',
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--training-csv', required=True)
    p.add_argument('--output', default=None)
    args = p.parse_args()
    report = run(args.checkpoint, args.training_csv)
    result = json.dumps(report, indent=2)
    print(result)
    if args.output:
        Path(args.output).write_text(result + '\n', encoding='utf-8')
    if report['status'] != 'PASS':
        raise SystemExit(2)

if __name__ == '__main__':
    main()
