#!/usr/bin/env python3
"""Extract policy_weights.pkl from an existing checkpoint.pkl (one-time helper)."""

import argparse
import os
import pickle

from softlearning.utils.keras import _apply_keras_hdf5_compat_patches


def main():
    parser = argparse.ArgumentParser(
        description='Export policy_weights.pkl from checkpoint.pkl')
    parser.add_argument(
        'checkpoint_dir',
        help='Checkpoint directory containing checkpoint.pkl')
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir.rstrip('/')
    src = os.path.join(checkpoint_dir, 'checkpoint.pkl')
    dst = os.path.join(checkpoint_dir, 'policy_weights.pkl')

    if not os.path.exists(src):
        raise FileNotFoundError(src)

    _apply_keras_hdf5_compat_patches()
    with open(src, 'rb') as f:
        picklable = pickle.load(f)
    weights = picklable['policy_weights']

    with open(dst, 'wb') as f:
        pickle.dump(weights, f)

    print('Wrote %s' % dst)


if __name__ == '__main__':
    main()
