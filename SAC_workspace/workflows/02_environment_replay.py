"""Read-only simulator smoke test. Replay buffer is introduced in Part 2."""
import numpy as np
from SAC_workspace.src.config.config import load_config
from SAC_workspace.src.environment.environment import SACEnvironment


def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--power-model', choices=('legacy','pvwatts_sapm'), default='legacy')
    p.add_argument('--wind-reference-height-m', type=float, default=2.0)
    p.add_argument('--gamma-pdc', type=float, default=-0.004)
    p.add_argument('--sapm-mount', default='open_rack_glass_polymer')
    p.add_argument('--wind-powerlaw-exponent', type=float, default=0.14)
    args=p.parse_args()
    kwargs=dict(power_model=args.power_model,
        wind_reference_height_m=args.wind_reference_height_m,
        gamma_pdc=args.gamma_pdc, sapm_mount=args.sapm_mount,
        wind_powerlaw_exponent=args.wind_powerlaw_exponent)
    config = load_config(**kwargs)
    env = SACEnvironment(config.train_env)
    try:
        env.seed(42)
        obs = env.reset()
        total = 0.0
        for i in range(env.horizon):
            obs, reward, done, info = env.step(np.zeros(2, dtype=np.float32))
            total += reward
            assert done == (i == env.horizon - 1)
        print(f'PASS: {env.horizon} real simulator transitions; zero-action net energy={total:.6f} kWh')
        print(f'Last scenario={info.get("scenario_id")} | observation size={obs.size}')
    finally:
        env.close()

if __name__ == '__main__':
    main()
