"""Read-only simulator smoke test. Replay buffer is introduced in Part 2."""
import numpy as np
from SAC_workspace.src.config.config import load_config
from SAC_workspace.src.environment.environment import SACEnvironment


def main():
    config = load_config()
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
