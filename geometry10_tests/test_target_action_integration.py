"""Regression coverage for action semantics across checkpoints and physical evaluations."""
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from mbpo.geometry10 import action_for_target_orientation, target_orientation_from_action


def test_inverse_and_domain():
    action = action_for_target_orientation(30., 359., 72., 1.)
    tilt, az, dt, da = target_orientation_from_action(30., 359., action)
    np.testing.assert_allclose([tilt, az, dt, da], [72., 1., 42., 2.], atol=2e-6)
    for invalid in ([0., np.nan], [1.001, 0.], [0.]):
        with pytest.raises(ValueError):
            target_orientation_from_action(30., 359., invalid)


def test_old_agent_and_replay_are_rejected():
    from SAC_workspace.src.agents.sac_agent import SACAgent
    from SAC_workspace.src.training.replay_buffer import ReplayBuffer

    for component in (SACAgent(), ReplayBuffer(8)):
        saved = component.state_dict()
        component.load_state_dict(saved)
        del saved["action_semantics"]
        with pytest.raises(ValueError, match="action semantics"):
            component.load_state_dict(saved)


def test_mc_periodic_and_recorder_share_baselines():
    from PV_experiments.representation.trajectories import Recorder
    from SAC_workspace.src.config.config import load_config
    from SAC_workspace.src.environment.environment import SACEnvironment
    from SAC_workspace.src.training.experiment_protocol import PeriodicEvaluator, build_scenario_protocol

    mc = importlib.import_module("MBPO_workspace.workflows.09_monte_carlo_evaluate")
    root = Path(__file__).resolve().parents[1]
    spec = json.loads((root / "PostProcessing_MC_patch/examples/hypothetical_uncertainty_5pct.json").read_text())
    cfg = load_config(require_data=False, weather_uncertainty=spec)
    if not cfg.manifest.is_file():
        pytest.skip("Prepare the repository NSRDB catalog for physical baseline parity")
    scenario = build_scenario_protocol(cfg.manifest, cfg.protocol)["train_ids"][0]
    env = SACEnvironment(cfg.eval_env)
    evaluator = PeriodicEvaluator.__new__(PeriodicEvaluator)
    evaluator.env = env
    try:
        for method, controller in [("fixed_no_motion", "fixed"), ("sun_tracking", "dual_axis_instantaneous")]:
            daily = mc.run_one(Recorder(env, method), method, scenario, 0)
            periodic = evaluator._run_episode(scenario, controller=controller)
            for key, other in [("gross_kwh", "gross_energy_kwh"), ("net_kwh", "net_energy_kwh"),
                               ("movement_kwh", "movement_cost_kwh")]:
                assert daily[key] == pytest.approx(periodic[other], abs=1e-12, rel=0.)
            if method == "fixed_no_motion":
                assert daily["movement_kwh"] == 0.
            else:
                assert daily["movement_kwh"] > 0.
    finally:
        env.close()
