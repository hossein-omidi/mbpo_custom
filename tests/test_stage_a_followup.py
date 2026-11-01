from __future__ import annotations
import ast
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_POWER_MODULES = [
    "pv_power_backend", "pvlib_physics", "pv_tracking",
    "stage_a_pv_tracking", "stage_a_data_contract",
]


def _find_module_tuple(source: str):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Tuple):
            names = [x.id for x in node.elts if isinstance(x, ast.Name)]
            if names == EXPECTED_POWER_MODULES:
                return names
    return None


def test_diagnostic_hashes_same_five_sources_as_environment():
    env_source = (ROOT / "SAC_workspace/src/environment/environment.py").read_text()
    diagnostic_source = (ROOT / "scripts/sac_mbpo_deep_diagnostic.py").read_text()
    assert _find_module_tuple(env_source) == EXPECTED_POWER_MODULES
    assert _find_module_tuple(diagnostic_source) == EXPECTED_POWER_MODULES


def test_fingerprint_helper_hashes_ordered_source_bytes_and_reports_components(tmp_path):
    from scripts.sac_mbpo_deep_diagnostic import fingerprint_modules
    p1, p2 = tmp_path / "first.py", tmp_path / "second.py"
    p1.write_bytes(b"first-source")
    p2.write_bytes(b"second-source")
    modules = (
        SimpleNamespace(__name__="first", __file__=str(p1)),
        SimpleNamespace(__name__="second", __file__=str(p2)),
    )
    digest, components = fingerprint_modules(modules)
    assert digest == hashlib.sha256(b"first-sourcesecond-source").hexdigest()
    assert components["first"] == hashlib.sha256(b"first-source").hexdigest()
    assert components["second"] == hashlib.sha256(b"second-source").hexdigest()


def test_sac_minimum_alpha_is_projected_and_can_recover():
    from SAC_workspace.src.agents.sac_agent import SACAgent
    agent = SACAgent(min_alpha=1e-3, lr=3e-4)
    floor = math.log(agent.min_alpha)
    agent.log_alpha.data.fill_(floor - 2.0)

    agent.alpha_opt.zero_grad(set_to_none=True)
    agent.log_alpha.sum().backward()  # gradient pushes alpha below the floor
    agent.alpha_opt.step()
    agent._project_alpha_floor()

    assert agent.log_alpha.item() == pytest.approx(floor, abs=1e-7)
    state = agent.alpha_opt.state[agent.log_alpha]
    assert torch.count_nonzero(state["exp_avg"]).item() == 0
    assert agent.alpha == pytest.approx(agent.min_alpha)

    agent.alpha_opt.zero_grad(set_to_none=True)
    (-agent.log_alpha.sum()).backward()  # later gradient requests larger alpha
    agent.alpha_opt.step()
    agent._project_alpha_floor()
    assert agent.log_alpha.item() > floor


def test_loading_old_checkpoint_projects_alpha_floor():
    from SAC_workspace.src.agents.sac_agent import SACAgent
    old = SACAgent(min_alpha=1e-3)
    floor = math.log(old.min_alpha)
    old.log_alpha.data.fill_(floor - 5.0)
    old.alpha_opt.zero_grad(set_to_none=True)
    old.log_alpha.sum().backward()
    old.alpha_opt.step()
    state = old.state_dict()

    restored = SACAgent(min_alpha=1e-3)
    restored.load_state_dict(state)
    assert restored.log_alpha.item() == pytest.approx(floor, abs=1e-7)
    assert restored.alpha == pytest.approx(restored.min_alpha)
    assert torch.count_nonzero(restored.alpha_opt.state[restored.log_alpha]["exp_avg"]).item() == 0


def test_sac_dual_axis_reference_passes_recorded_albedo():
    source = (ROOT / "SAC_workspace/src/evaluation/evaluation.py").read_text()
    tree = ast.parse(source)
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "compute_pv_power":
                calls.append({kw.arg for kw in node.keywords})
    assert calls
    assert any("surface_albedo" in kwargs for kwargs in calls)
    assert "NSRDB dual-axis reference is missing recorded surface_albedo" in source


def test_sac_dual_axis_reference_reuses_recorded_nsrdb_albedo():
    pytest.importorskip("pvlib")
    pytest.importorskip("gym")
    import numpy as np
    from SAC_workspace.src.config.config import load_config
    from SAC_workspace.src.environment.environment import SACEnvironment
    from SAC_workspace.src.evaluation.evaluation import (
        run_episode, ideal_instantaneous_dual_axis,
    )
    from mbpo.env.pv_power_backend import compute_pv_power

    cfg = load_config(power_model="pvwatts_sapm")
    env = SACEnvironment(cfg.eval_env)
    try:
        fixed = run_episode(cfg, "fixed_no_motion", "2024-06-21", 42, env=env)
        dual = ideal_instantaneous_dual_axis(cfg, fixed, env=env)
        physical = getattr(env, "unwrapped", env)
        assert len(dual["power"]) == len(fixed["power"]) == 117

        for i in range(117):
            info = fixed["infos"][i]
            assert dual["infos"][i]["surface_albedo"] == pytest.approx(
                info["surface_albedo"], abs=1e-12
            )
            expected = compute_pv_power(
                dual["tilt"][i], dual["azimuth"][i],
                info["solar_zenith_deg"], info["solar_azimuth_deg"],
                info["dni_wm2"], info["ghi_wm2"], info["dhi_wm2"],
                area=physical.area, efficiency=physical.efficiency,
                power_model=physical.power_model,
                temp_air_c=info.get("temperature_c"),
                wind_speed_ms=info.get("wind_speed_m_s"),
                wind_reference_height_m=physical.wind_reference_height_m,
                gamma_pdc=physical.gamma_pdc,
                sapm_mount=physical.sapm_mount,
                wind_powerlaw_exponent=physical.wind_powerlaw_exponent,
                surface_albedo=info["surface_albedo"],
            )
            assert dual["power"][i] == pytest.approx(expected.power_w, rel=1e-9, abs=1e-7)
    finally:
        env.close()
