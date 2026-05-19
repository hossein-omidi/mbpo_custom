import numpy as np


class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
        notdone = np.isfinite(next_obs).all(axis=-1)
        done = ~notdone
        return done[:, None]
