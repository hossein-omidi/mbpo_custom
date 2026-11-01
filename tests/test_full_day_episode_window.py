from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def literal_assignment(path, name):
    text = (ROOT / path).read_text(encoding='utf-8')
    match = re.search(r'^\s*' + re.escape(name) + r"\s*=\s*(['\"])(.*?)\1\s*$|^\s*" + re.escape(name) + r"\s*=\s*(\d+)\s*$", text, re.MULTILINE)
    if not match:
        raise AssertionError(f'{name} literal assignment not found in {path}')
    if match.group(1):
        return match.group(2)
    return int(match.group(3))


def test_shared_reference_window_is_full_daylight_envelope():
    p = 'examples/config/pv_tracking/_nsrdb_base.py'
    assert literal_assignment(p, 'EPISODE_START_TIME') == '08:45'
    assert literal_assignment(p, 'EPISODE_PERIODS') == 196
    assert literal_assignment(p, 'EPISODE_FREQ') == '5min'
    assert 'EPISODE_ACTION_STEPS = EPISODE_PERIODS - 1  # 195' in (ROOT / p).read_text(encoding='utf-8')


def test_environment_defaults_match_shared_window():
    assert literal_assignment('mbpo/env/pv_tracking.py', 'DEFAULT_START_TIME') == '08:45'
    assert literal_assignment('mbpo/env/pv_tracking.py', 'DEFAULT_PERIODS') == 196
    assert 'DEFAULT_EPISODE_STEPS = DEFAULT_PERIODS - 1  # 195 transitions per episode' in (ROOT / 'mbpo/env/pv_tracking.py').read_text(encoding='utf-8')
    assert literal_assignment('mbpo/env/nsrdb_weather.py', 'DEFAULT_EPISODE_START_TIME') == '08:45'
    assert literal_assignment('mbpo/env/nsrdb_weather.py', 'DEFAULT_EPISODE_PERIODS') == 196


def test_window_math():
    # 08:45 -> 01:00 next date = 16h15 = 975 min = 195 transitions.
    start = 8 * 60 + 45
    end = 24 * 60 + 60
    assert end - start == 975
    assert (end - start) // 5 == 195
    assert 195 + 1 == 196


def test_evaluation_pipeline_has_no_fixed_117_step_assumptions():
    p = (ROOT / 'MBPO_workspace/src/advanced_suite.py').read_text(encoding='utf-8')
    assert 'len(rr)!=117' not in p
    assert 'np.arange(1,118)' not in p
    assert '(len(seedrr)*117.)' not in p
    assert '117-step window' not in p
