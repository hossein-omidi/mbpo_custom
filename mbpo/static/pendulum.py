import numpy as np

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
        
        # For Pendulum, we only terminate if observations become non-finite
        # Pendulum doesn't have explicit terminal states, so we just check validity
        done = ~np.isfinite(next_obs).all(axis=-1)
        done = done[:, None]
        return done
