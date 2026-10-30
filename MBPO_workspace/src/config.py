"""Ray-free MBPO configuration. The MDP and SAC hyperparameters come from
SAC_workspace's audited reader of the existing conf_compact.py source.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from SAC_workspace.src.config.config import (
    load_config as load_sac_config, _SKELETON, _BASE, _COMPACT,
    _literal_mapping_fields, _updated_dict, _call_kwargs,
)


@dataclass(frozen=True)
class ModelConfig:
    real_ratio: float
    num_models: int
    num_elites: int
    hidden_dim: int
    rollout_length: int
    model_train_freq: int
    rollout_batch_size: int
    model_pool_capacity: int = 10000
    model_retain_epochs: int = 4
    require_model_skill: bool = True
    model_lr: float = 0.001
    model_epochs: int = 5
    model_patience: int = 5
    model_batch_size: int = 512
    model_max_train_samples: int = 6000
    model_holdout_fraction: float = 0.2
    min_holdout_scenarios: int = 2
    uncertainty_max_std: float | None = None

    def validate(self):
        if not 0.0 < self.real_ratio < 1.0:
            raise ValueError('Model-based training requires real_ratio strictly between 0 and 1')
        if not (2 <= self.num_elites <= self.num_models):
            raise ValueError('Need at least two ensemble members and valid elite count')
        if self.rollout_length != 1:
            raise ValueError('Only the audited 1-step compact-state MBPO rollout is supported')
        for n in (self.hidden_dim, self.model_train_freq, self.rollout_batch_size,
                  self.model_pool_capacity, self.model_retain_epochs, self.model_epochs, self.model_patience,
                  self.model_batch_size, self.model_max_train_samples):
            if n <= 0: raise ValueError('Nonpositive ensemble training parameter')
        if not 0 < self.model_holdout_fraction < 0.5: raise ValueError('Invalid holdout fraction')
        if self.min_holdout_scenarios < 2: raise ValueError('Need at least two scenarios to hold out a full day')
        if self.uncertainty_max_std is not None and self.uncertainty_max_std <= 0:
            raise ValueError('uncertainty_max_std must be positive')


@dataclass(frozen=True)
class MBPOConfig:
    sac: object
    model: ModelConfig

    def to_dict(self):
        return {'sac': self.sac.to_dict(), 'model': asdict(self.model),
                'schema': 'mbpo-torch-structured-v1'}


def load_config(require_data=True, model_overrides=None, *, power_model='legacy',
                wind_reference_height_m=2.0, gamma_pdc=-0.004,
                sapm_mount='open_rack_glass_polymer', wind_powerlaw_exponent=0.14) -> MBPOConfig:
    sac = load_sac_config(require_data=require_data, power_model=power_model,
        wind_reference_height_m=wind_reference_height_m, gamma_pdc=gamma_pdc,
        sapm_mount=sapm_mount, wind_powerlaw_exponent=wind_powerlaw_exponent)
    keys = {'real_ratio','num_networks','num_elites','hidden_dim',
            'max_model_rollout_length','model_train_freq','rollout_batch_size','model_retain_epochs'}
    orig = _literal_mapping_fields(_SKELETON, 'params', keys)
    orig.update({k:v for k,v in _updated_dict(_BASE,'default_nsrdb_algo_kwargs').items() if k in keys})
    orig.update({k:v for k,v in _call_kwargs(_COMPACT,'build_nsrdb_params','algo_kwargs').items() if k in keys})
    mandatory = {'real_ratio','num_networks','num_elites','hidden_dim','max_model_rollout_length','model_train_freq','rollout_batch_size','model_retain_epochs'}
    missing = mandatory - set(orig)
    if missing: raise ValueError(f'MBPO reference model configuration is incomplete: {sorted(missing)}')
    mc = dict(real_ratio=float(orig['real_ratio']), num_models=int(orig['num_networks']),
              num_elites=int(orig['num_elites']),hidden_dim=int(orig['hidden_dim']),
              rollout_length=int(orig['max_model_rollout_length']),
              model_train_freq=int(orig['model_train_freq']),
              rollout_batch_size=int(orig['rollout_batch_size']),model_retain_epochs=int(orig['model_retain_epochs']))
    mc.update(model_overrides or {})
    model = ModelConfig(**mc)
    model.validate()
    return MBPOConfig(sac, model)


def config_digest(cfg: MBPOConfig) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(cfg.to_dict(),sort_keys=True).encode()).hexdigest()
