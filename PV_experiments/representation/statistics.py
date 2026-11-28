"""Explicit estimands for crossed training seeds and nested weather draws."""
from __future__ import annotations
import numpy as np
import pandas as pd

METHODS=('fixed_no_motion','sun_tracking','sac','mbpo')
ORACLE='poa_instant_ideal'
LABELS=dict(fixed_no_motion='Fixed (30° south)',sun_tracking='Astronomical (instantaneous)',sac='SAC',mbpo='MBPO',poa_instant_ideal='Instantaneous DC upper reference')
COLORS=dict(fixed_no_motion='#666666',sun_tracking='#D55E00',sac='#0072B2',mbpo='#009E73',poa_instant_ideal='#555555')
LABELS.update(mbpo_minus_sac='MBPO − SAC',mbpo_minus_sun_tracking='MBPO − astronomical')
COLORS.update(mbpo_minus_sac='#009E73',mbpo_minus_sun_tracking='#D55E00')
PAIRS=(('mbpo','sac'),('mbpo','sun_tracking'),('sac','sun_tracking'),('mbpo','fixed_no_motion'),('sac','fixed_no_motion'))


def finite(x):
    a=np.asarray(x,dtype=float)
    if a.size==0 or not np.isfinite(a).all():raise ValueError('Empty or nonfinite scientific array')
    return a


def equal_mixture(x):
    """x[training seed, historical day, weather draw, ...]. Equal empirical weights.

    Descriptive population variance decomposes (ordered conditioning on seed,
    then day). Crossed seed/day interactions are NOT independent random effects.
    Sample SD of seed means is a separate statistic (undefined for one seed).
    """
    x=finite(x)
    if x.ndim<3:raise ValueError('Expected seed/day/draw axes')
    mu=x.mean(axis=(0,1,2)); sm=x.mean(axis=(1,2)); dm=x.mean(axis=2)
    vs=np.mean((sm-mu)**2,axis=0)
    vd=np.mean((dm-sm[:,None])**2,axis=(0,1))
    vw=np.mean((x-dm[:,:,None])**2,axis=(0,1,2))
    total=np.mean((x-mu)**2,axis=(0,1,2))
    if not np.allclose(total,vs+vd+vw,rtol=1e-9,atol=1e-12):raise ArithmeticError('Variance identity failed')
    return dict(mean=mu,sd_total=np.sqrt(total),sd_seed_means=sm.std(axis=0,ddof=1) if x.shape[0]>1 else np.full_like(mu,np.nan),
                var_seed=vs,var_day_given_seed=vd,var_draw_given_seed_day=vw,
                q05=np.quantile(x,.05,axis=(0,1,2)),q95=np.quantile(x,.95,axis=(0,1,2)))


def from_day_moments(mu,sd,draws):
    """Exact empirical pooled moment from per-day sample SD across M draws.
    mu/sd axes [seed, day, step]. Not a paired difference or cumulative SD.
    """
    mu,sd=finite(mu),finite(sd)
    if mu.shape!=sd.shape or mu.ndim!=3 or draws<2 or (sd<0).any():raise ValueError('Invalid conditional moments')
    mean=mu.mean(axis=(0,1)); sm=mu.mean(axis=1)
    vs=np.mean((sm-mean)**2,axis=0); vd=np.mean((mu-sm[:,None])**2,axis=(0,1)); vw=(draws-1)/draws*np.mean(sd**2,axis=(0,1))
    return dict(mean=mean,sd_total=np.sqrt(vs+vd+vw),sd_seed_means=sm.std(axis=0,ddof=1) if len(sm)>1 else np.full_like(mean,np.nan),var_seed=vs,var_day_given_seed=vd,var_draw_given_seed_day=vw)


def circular_stats(degrees,axis=0):
    """Circular mean in degrees and resultant length. No linear azimuth SD."""
    a=np.deg2rad(finite(degrees)); z=np.mean(np.exp(1j*a),axis=axis); r=np.abs(z)
    mean=np.where(r>1e-8,np.mod(np.rad2deg(np.angle(z)),360),np.nan)
    # Floating-point modulo can round an infinitesimal negative angle to 360.
    mean=np.where(mean==360.,0.,mean)
    return mean,r


def season(month):
    return ('DJF' if month in (12,1,2) else 'MAM' if month in (3,4,5) else 'JJA' if month in (6,7,8) else 'SON')


def clock_grid(sid,start='08:45',horizon=195):
    return pd.date_range(f'{pd.Timestamp(sid).date()} {start}',periods=horizon+1,freq='5min',tz='UTC')[1:]


def check_close(a,b,what,atol=1e-8,rtol=1e-8):
    a,b=finite(a),finite(b)
    if a.shape!=b.shape:raise ValueError(f'{what}: shape mismatch {a.shape} != {b.shape}')
    if not np.allclose(a,b,atol=atol,rtol=rtol):raise ValueError(f'{what}: numerical mismatch')


def seed_summary(values):
    a=finite(values);return float(a.mean()), float(a.std(ddof=1)) if a.size>1 else None


def daily_spread(x):
    """Scalar energy array [seed, day, draw]. Distinct descriptive estimands.

    Historical-day sample SD uses marginal day means (averaged over seed/draw).
    RMS weather SD uses conditional sample variances (within seed/day).
    Neither is an independent random-effects estimate or a CI.
    """
    x=finite(x)
    if x.ndim!=3 or x.shape[2]<2:raise ValueError('Expected seed/day/draw energy array with >=2 draws')
    day=x.mean(axis=(0,2))
    return dict(sd_day_means=float(day.std(ddof=1)) if len(day)>1 else np.nan,
                rms_weather_draw_sd=float(np.sqrt(np.mean(x.var(axis=2,ddof=1)))))


def trajectory_clock(quantity,horizon=195):
    """Unwrapped UTC hours. Commands at decision time; resulting quantities at response time."""
    decision=quantity in ('action_tilt','action_azimuth','abs_action_tilt','abs_action_azimuth')
    return 8.75+(np.arange(horizon)+(0 if decision else 1))/12., ('decision' if decision else 'response')


def cumulative_plot_series(clock,mean,sd):
    """Prepend deterministic zero ONLY for rendering integrated quantities."""
    clock,mean,sd=finite(clock),finite(mean),finite(sd)
    if clock.ndim!=1 or clock.shape!=mean.shape or clock.shape!=sd.shape:raise ValueError('Cumulative plot shape mismatch')
    return np.r_[clock[0]-1/12.,clock],np.r_[0.,mean],np.r_[0.,sd]
