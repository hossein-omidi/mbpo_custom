import gym
import numpy as np

# Ensure the custom MBPO environment registry is loaded.
import softlearning.environments.adapters.gym_adapter  # noqa: F401


def main():
    env = gym.make('PVTracking-v0')
    obs = env.reset()
    print('reset obs shape:', np.asarray(obs).shape)
    print('action space:', env.action_space)
    print('obs space:', env.observation_space)

    for step in range(10):
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        print(
            f'step={step}',
            f'action={action}',
            f'reward={reward:.4f}',
            f'done={done}',
            f'obs_shape={np.asarray(next_obs).shape}',
            f'info_keys={list(info.keys())}',
        )
        if done:
            print('done at step', step)
            break

    env.close()
    print('PVTracking environment checker completed successfully.')


if __name__ == '__main__':
    main()
