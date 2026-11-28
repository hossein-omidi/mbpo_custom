"""Pure numerical geometry contract tests (no gym/pvlib required)."""
import numpy as np
import pytest
from mbpo.geometry10 import (MODE, STATE_DIM, TAU, PANEL_AZ, SOLAR, WEATHER_DIM,
    POWER_DIM, FEATURE_NAMES, LOW, HIGH, build, check_batch, azimuth, circular,
    solar_features, weather_features, pose_transition, SolarContext, angular_rmse_deg)


def state(az=359., tilt=30., tau=1., zenith=45., sun_az=90.):
    return build(600.,120.,150.,tilt,az,zenith,sun_az,tau)


def test_feature_order_and_bounds():
    x=state()
    assert MODE=='geometry10' and len(x)==STATE_DIM==len(FEATURE_NAMES)==10
    assert np.all(x>=LOW) and np.all(x<=HIGH)
    assert np.allclose(x[:4],[.3,.06,.075,1/3])
    assert np.allclose(x[SOLAR],[.25,1,0],atol=1e-7)
    assert x[TAU]==1


def test_wrap_continuity_and_round_trip():
    x,y=state(359.),state(1.)
    assert np.linalg.norm(x-y)<.04
    assert np.isclose(azimuth(x[PANEL_AZ]),359,atol=1e-4)
    assert np.isclose(angular_rmse_deg(x[PANEL_AZ],y[PANEL_AZ]),2.,atol=1e-4)


def test_invalid_state_rejected():
    for x in (np.zeros(6),np.zeros(10),state()*np.nan):
        with pytest.raises(ValueError):check_batch(x)
    x=state();x[0]=-1
    with pytest.raises(ValueError):check_batch(x)


def test_night_and_terminal():
    x=build(0,0,0,30,180,120,360,0)
    assert np.all(x[:3]==0) and x[TAU]==0
    assert np.isclose(x[6],2/3) and np.isclose(x[8],1)
    with pytest.raises(ValueError):pose_transition(x[None,:],np.zeros((1,2)),195)


def test_target_extremes_and_wrap():
    x=np.stack([state(359,89,1/195),state(1,1,2/195)])
    a=np.array([[1,1],[-1,-1]],dtype=float)
    tilt,az,tau,dt,da,done=pose_transition(x,a,195)
    assert np.allclose(tilt,[1,0])
    assert np.allclose(azimuth(az),[179,181],atol=1e-4)
    assert np.allclose(dt,[1,-1],atol=1e-5)
    assert np.allclose(da,[180,-180],atol=1e-5)
    assert np.allclose(tau,[0,1/195]) and done.tolist()==[True,False]


def test_exogenous_inputs_do_not_depend_on_action_or_panel_pose():
    x=state(359,30);y=state(180,60)
    y[2]=.99  # Power must not enter weather model either.
    assert np.array_equal(weather_features(x),weather_features(y))
    assert WEATHER_DIM==6 and POWER_DIM==14
    z=state(359,30,zenith=70,sun_az=120)
    assert not np.array_equal(weather_features(x),weather_features(z))


def test_solar_context_selects_correct_day_and_time():
    # Test only the scenario/time lookup; the installed integration tests check
    # the cache against real pvlib ephemeris and actual env.step.
    from types import SimpleNamespace
    fake=SimpleNamespace(unwrapped=object(),horizon=195)
    ctx=SolarContext(fake)
    grid1=solar_features(np.linspace(100,120,196),np.linspace(40,100,196))
    grid2=solar_features(np.linspace(80,100,196),np.linspace(90,160,196))
    ctx.cache={'a':grid1,'b':grid2}
    x=np.stack([state(tau=1),state(tau=1/195),state(tau=100/195)])
    x[0,SOLAR]=grid1[0];x[1,SOLAR]=grid2[194];x[2,SOLAR]=grid1[95]
    assert np.allclose(ctx.next(x,['a','b','a']),[grid1[1],grid2[195],grid1[96]])
    with pytest.raises(ValueError):ctx.next(x,['b','b','a'])
    with pytest.raises(ValueError):ctx.next(x,None)


def test_invalid_off_grid_tau_rejected():
    x=state(tau=.12345)[None,:]
    with pytest.raises(ValueError):pose_transition(x,np.zeros((1,2)),195)


def test_power_normalization_is_invertible_not_saturated():
    for power in (0., 75., 2000.):
        obs=build(500,100,power,30,180,60,90,1)
        assert np.isclose(float(obs[2])*2000,power,atol=1e-4)
    for power in (-.1,2000.01,np.inf):
        with pytest.raises(ValueError):build(500,100,power,30,180,60,90,1)


def test_invalid_action_is_rejected():
    x=state(az=120,tilt=30,tau=1)[None,:]
    a=np.array([[1.000001,-1.000001]])
    with pytest.raises(ValueError):pose_transition(x,a,195)
    with pytest.raises(ValueError):pose_transition(x,[[1.01,0]],195)


def test_solar_context_rejects_wrong_metadata_shape():
    from types import SimpleNamespace
    ctx=SolarContext(SimpleNamespace(unwrapped=object(),horizon=195))
    x=state()[None,:]
    with pytest.raises(ValueError):ctx.next(x,[['a']])
