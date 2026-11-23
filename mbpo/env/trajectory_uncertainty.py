"""Correlated, episode-level NSRDB uncertainty sensitivity model.

This is a *hypothetical* error distribution, NOT calibrated NSRDB accuracy.
The historical observed five-minute input remains the immutable reference.
One RNG draw creates a complete weather trajectory at reset; conditional on
that draw, every subsequent physical environment step is deterministic.

Errors have a Gaussian day component plus stationary correlated AR(1) errors.
The irradiance errors use mean-one lognormal multiplicative factors. GHI is
reconstructed by preserving the observed DNI/DHI/GHI closure *residual*;
thus zero uncertainty is exactly the original NSRDB dataset, while changes
in DNI/DHI produce consistent changes in GHI. Large original closure errors
are not silently eliminated by this sensitivity experiment.

The parameter values must be supported by validation or declared explicitly
as sensitivity assumptions; they are never described as measurements.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math
import numpy as np

CHANNELS=('dni','dhi','temperature','wind_speed','surface_albedo')
DEFAULTS={
    'schema':'correlated_weather_assumption_v1',
    'enabled':False,
    'seed':2026,
    'dni_cv':0.0,
    'dhi_cv':0.0,
    'temperature_sd_c':0.0,
    'wind_cv':0.0,
    'albedo_sd':0.0,
    'day_fraction':0.50,
    'ar1':0.90,
    'latent_correlation':[
        [1.0,0.2,0.,0.,0.],
        [0.2,1.0,0.,0.,0.],
        [0.,0.,1.,0.,0.],
        [0.,0.,0.,1.,0.],
        [0.,0.,0.,0.,1.],
    ],
}
ERROR_KEYS=('dni_cv','dhi_cv','temperature_sd_c','wind_cv','albedo_sd')


def normalized_spec(raw=None):
    """Strict JSON-ready protocol for reproducible training/evaluation."""
    if raw is None:raw={}
    if not isinstance(raw,dict):raise TypeError('weather_uncertainty must be a JSON object')
    extra=set(raw)-set(DEFAULTS)
    if extra:raise ValueError(f'Unexpected uncertainty fields: {sorted(extra)}')
    spec={**DEFAULTS,**raw}
    if spec['schema']!=DEFAULTS['schema'] or type(spec['enabled']) is not bool:
        raise ValueError('Invalid uncertainty schema or enabled flag')
    if type(spec['seed']) is not int or not 0<=spec['seed']<2**63:
        raise ValueError('Uncertainty seed must be a nonnegative integer under 2**63')
    for k in ERROR_KEYS:
        v=float(spec[k])
        upper=20.0 if k=='temperature_sd_c' else .5
        if not math.isfinite(v) or v<0 or v>upper:
            raise ValueError(f'{k} must be finite and between 0 and {upper}')
        spec[k]=v
    day=float(spec['day_fraction']);ar=float(spec['ar1'])
    if not math.isfinite(day) or not 0<=day<=1:
        raise ValueError('day_fraction must be in [0,1]')
    if not math.isfinite(ar) or not -0.999<ar<0.999:
        raise ValueError('ar1 must be in (-0.999,0.999)')
    spec['day_fraction']=day;spec['ar1']=ar
    matrix=np.asarray(spec['latent_correlation'],dtype=float)
    if (matrix.shape!=(5,5) or not np.isfinite(matrix).all()
            or not np.allclose(matrix,matrix.T,rtol=0,atol=1e-12)
            or not np.allclose(np.diag(matrix),1.,rtol=0,atol=1e-12)
            or np.min(np.linalg.eigvalsh(matrix))<-1e-10):
        raise ValueError('latent_correlation must be a symmetric 5x5 PSD correlation matrix')
    # Force positive definite for Cholesky; ill-conditioned singular matrices
    # cannot be reliably sampled by the numerical implementation.
    if np.min(np.linalg.eigvalsh(matrix))<=1e-9:
        raise ValueError('latent_correlation must be numerically positive definite')
    spec['latent_correlation']=matrix.tolist()
    if spec['enabled'] and not any(spec[k]>0 for k in ERROR_KEYS):
        raise ValueError('Uncertainty enabled but every uncertainty magnitude is zero')
    return spec


def spec_fingerprint(spec):
    data=json.dumps(normalized_spec(spec),sort_keys=True,separators=(',',':'),allow_nan=False)
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def realization_seed(spec,scenario_id,realization_id):
    """Hash-based CRN seed, invariant to Python hash randomization and call order."""
    s=normalized_spec(spec)
    if not isinstance(scenario_id,str) or not scenario_id:
        raise ValueError('A concrete historical scenario_id is required')
    if isinstance(realization_id,bool) or not isinstance(realization_id,(int,np.integer)) or realization_id<0:
        raise ValueError('realization_id must be a nonnegative integer')
    blob=f'{s["schema"]}|{s["seed"]}|{scenario_id}|{int(realization_id)}'.encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:8],'little')


def _latent_path(spec,n,rng):
    """Stationary Gaussian time series, covariance C at every t."""
    C=np.asarray(spec['latent_correlation'],dtype=float)
    chol=np.linalg.cholesky(C)
    day=(chol@rng.standard_normal(5)).reshape(1,5)
    phi=spec['ar1'];factor=math.sqrt(1-phi**2)
    z=np.empty((n,5),dtype=float)
    z[0]=chol@rng.standard_normal(5)
    for t in range(1,n):
        z[t]=phi*z[t-1]+factor*(chol@rng.standard_normal(5))
    q=spec['day_fraction']
    return math.sqrt(q)*day+math.sqrt(1-q)*z


def generate_weather_profile(frame, cos_zenith, spec, *, seed):
    """Return an independent perturbed copy of a complete historical profile.

    frame: a pandas DataFrame (or DataFrame-compatible data) with fields
      dni,dhi,ghi,temperature,wind_speed and optional surface_albedo.
    cos_zenith: clip(cos(solar zenith), 0, 1) for each original UTC timestamp.
    This function NEVER modifies frame and never accesses the panel action.
    """
    s=normalized_spec(spec)
    out=frame.copy(deep=True)
    n=len(out)
    if n<2:raise ValueError('Weather episode needs at least two timestamps')
    cosz=np.asarray(cos_zenith,dtype=float)
    if cosz.shape!=(n,) or not np.isfinite(cosz).all() or (cosz<-1e-8).any() or (cosz>1+1e-8).any():
        raise ValueError('cos_zenith must be a valid n-timestamp solar geometry vector')
    cosz=np.clip(cosz,0.,1.)
    required=('dni','dhi','ghi','temperature','wind_speed')
    for col in required:
        if col not in out:raise ValueError(f'Missing weather column {col}')
        v=np.asarray(frame[col],dtype=float)
        if not np.isfinite(v).all():raise ValueError(f'Nonfinite NSRDB {col}')
        if col in ('dni','dhi','ghi','wind_speed') and np.min(v)<0:
            raise ValueError(f'Negative source NSRDB {col}')
    alb_col=next((c for c in ('surface_albedo','albedo','Surface Albedo','Surface_Albedo') if c in out),None)
    if s['enabled'] and s['albedo_sd']>0 and alb_col is None:
        raise ValueError('Albedo uncertainty requested without recorded albedo')
    if alb_col is not None:
        alb=np.asarray(frame[alb_col],dtype=float)
        if not np.isfinite(alb).all() or np.any((alb<0)|(alb>1)):
            raise ValueError('Source albedo must be in [0,1]')
        # Do not silently harmonize conflicting aliases. The unmodified
        # Stage-A physical evaluator rejects disagreement; augmentation must
        # respect the same data-integrity contract.
        aliases=[c for c in ('surface_albedo','albedo','Surface Albedo','Surface_Albedo')
                 if c in frame]
        for other in aliases[1:]:
            compare=np.asarray(frame[other],dtype=float)
            if not np.isfinite(compare).all() or not np.allclose(
                    compare,alb,rtol=0.,atol=1e-8):
                raise ValueError('Conflicting NSRDB surface-albedo column aliases')
    if not s['enabled']:
        return out,{'enabled':False,'seed':None,'clip_count':0,'max_baseline_closure_residual_wm2':float(np.max(np.abs(
             np.asarray(frame['ghi'],float)-np.asarray(frame['dhi'],float)-np.asarray(frame['dni'],float)*cosz)))}
    rng=np.random.default_rng(seed)
    shocks=_latent_path(s,n,rng)
    def multiplier(cv,z):
        if cv==0:return np.ones_like(z)
        sigma=math.sqrt(math.log1p(cv*cv))
        return np.exp(np.clip(sigma*z-.5*sigma*sigma,-30.,30.))
    dni=np.asarray(frame['dni'],float);dhi=np.asarray(frame['dhi'],float)
    ghi=np.asarray(frame['ghi'],float)
    dninew=dni*multiplier(s['dni_cv'],shocks[:,0]);dhinew=dhi*multiplier(s['dhi_cv'],shocks[:,1])
    ghinew=ghi+(dninew-dni)*cosz+(dhinew-dhi)
    # Preserve the source GHI/DHI/DNI closure residual where feasible.
    # Physically, nonnegative beam-horizontal irradiance implies GHI >= DHI.
    # Preserve source values as reference; project only the stochastic output
    # when the sampled residual would imply negative beam-horizontal energy.
    # Every projection is counted because closure may cease to be preserved.
    projected=ghinew<dhinew
    ghinew=np.maximum(ghinew,dhinew)
    out['dni']=dninew;out['dhi']=dhinew;out['ghi']=ghinew
    out['temperature']=np.asarray(frame['temperature'],float)+s['temperature_sd_c']*shocks[:,2]
    out['wind_speed']=np.asarray(frame['wind_speed'],float)*multiplier(s['wind_cv'],shocks[:,3])
    albedo_clips=0
    if alb_col is not None and s['albedo_sd']>0:
        candidate=np.asarray(frame[alb_col],float)+s['albedo_sd']*shocks[:,4]
        albedo_clips=int(np.count_nonzero((candidate<0)|(candidate>1)))
        new_albedo=np.clip(candidate,0.,1.)
        for col in ('surface_albedo','albedo','Surface Albedo','Surface_Albedo'):
            if col in out:out[col]=new_albedo
    diagnostics={
        'enabled':True,'seed':int(seed),
        'clip_count':int(np.count_nonzero(projected))+albedo_clips,
        'ghi_dhi_projection_count':int(np.count_nonzero(projected)),
        'albedo_projection_count':int(albedo_clips),
        'max_baseline_closure_residual_wm2':float(np.max(np.abs(ghi-dhi-dni*cosz))),
        'max_post_closure_residual_wm2':float(np.max(np.abs(ghinew-dhinew-dninew*cosz))),
        'max_ghi_below_dhi_wm2':float(np.max(np.maximum(dhinew-ghinew,0.))),
        'uncertainty_spec_sha256':spec_fingerprint(s),
        'interpretation':'hypothetical correlated trajectory uncertainty, not calibrated NSRDB error',
    }
    if not (np.isfinite(out[list(required)]).to_numpy(dtype=float)).all():
        raise FloatingPointError('Nonfinite weather values after uncertainty propagation')
    return out,diagnostics
