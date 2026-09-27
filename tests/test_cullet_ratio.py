"""碎玻璃掺配决策：回收料定比例、窑温校核、自动收回与配比联动。"""

from __future__ import annotations

import copy
import os

import pytest

from glassmelt.app import CycleInputs
from glassmelt.config import (
    PlatformConfig,
    config_to_mapping,
    default_config,
    load_config,
)
from glassmelt.console.api import ConsoleAPI
from glassmelt.errors import ConfigError, InterlockError

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "glassmelt.yaml"
)


def _supply(**overrides) -> dict:
    payload = {
        "inventory_t": 3000.0,
        "incoming_tpd": 200.0,
        "grade": "A",
        "moisture": 0.01,
        "contamination": 0.01,
    }
    payload.update(overrides)
    return payload


def test_abundant_supply_rolls_back_with_reason(platform):
    result = platform.plan_cullet(_supply())
    decision = result["decision"]
    # 回收料充足：物料侧想按 A 级上限 80% 执行
    assert decision["desired_pct"] == pytest.approx(80.0)
    assert decision["binding"] == "质量等级A级"
    # 窑温兜不住 80%：热负荷需求跌破最低稳定燃烧，自动收回到可行窗口内
    assert decision["ratio_pct"] == pytest.approx(71.826, abs=1e-2)
    assert len(decision["adjustments"]) == 1
    adjustment = decision["adjustments"][0]
    assert adjustment["constraint"] == "最低稳定燃烧负荷"
    assert "压不住" in adjustment["reason"]
    assert adjustment["from_pct"] == pytest.approx(80.0)
    assert adjustment["to_pct"] == pytest.approx(decision["ratio_pct"])
    assert decision["violations"] == []
    # 收回后的比例进了配料计算
    assert result["mix"]["cullet_ratio_pct"] == pytest.approx(decision["ratio_pct"])
    assert result["mix"]["cullet_kg"] > 0.0


def test_rollback_is_announced_on_the_event_bus(platform):
    seen: list[str] = []
    platform.bus.subscribe("batch", lambda event: seen.append(event.action))
    platform.plan_cullet(_supply())
    assert "cullet_ratio_decided" in seen
    assert "cullet_ratio_adjusted" in seen


def test_short_supply_follows_availability(platform):
    result = platform.plan_cullet(
        _supply(inventory_t=60.0, incoming_tpd=20.0, grade="B", contamination=0.02)
    )
    decision = result["decision"]
    assert decision["ratio_pct"] == pytest.approx(5.775, abs=1e-2)
    assert decision["binding"] == "回收料供应"
    assert decision["adjustments"] == []
    assert any("工艺下限" in alarm for alarm in decision["alarms"])


def test_normal_supply_runs_without_adjustment(platform):
    result = platform.plan_cullet(_supply(inventory_t=800.0, incoming_tpd=150.0, grade="B"))
    decision = result["decision"]
    assert decision["ratio_pct"] == pytest.approx(50.0)
    assert decision["adjustments"] == []
    assert decision["alarms"] == []
    assert decision["violations"] == []
    assert result["mix"]["cullet_kg"] == pytest.approx(640.0)


def test_ratio_flows_into_cycle_mix_plans(platform):
    platform.start()
    platform.plan_cullet(_supply(inventory_t=800.0, incoming_tpd=150.0, grade="B"))
    cycle = platform.cycle(CycleInputs(load=0.6, pull_rate=0.8, gob_mass_g=186.0, dt=1.0))
    assert cycle["mix"]["cullet_ratio_pct"] == pytest.approx(50.0)
    assert cycle["mix"]["cullet_kg"] == pytest.approx(640.0)
    # 比例变了，按老比例出的配比计划必须作废
    stale = platform.mix.plan(platform.scale.baseline(), cullet_ratio_pct=30.0)
    assert platform.mix.matches(stale, platform.scale.baseline(), cullet_ratio_pct=50.0) is False
    assert platform.mix.matches(stale, platform.scale.baseline(), cullet_ratio_pct=30.0) is True


def test_feasible_window_is_analytic(platform):
    window = platform.cullet.feasible_window_pct()
    assert window["hi_binding"] == "最低稳定燃烧负荷"
    assert window["hi_pct"] == pytest.approx(72.826, abs=1e-2)
    assert window["lo_pct"] == pytest.approx(0.0)


def test_heat_and_temp_estimates(platform):
    planner = platform.cullet
    assert planner.heat_demand_gj_h(0.0) == pytest.approx(115.2)
    assert planner.heat_demand_gj_h(80.0) == pytest.approx(71.04)
    assert planner.check(80.0)[0].direction == "reduce"
    assert planner.check(50.0) == []


def test_supply_payload_is_validated(platform):
    with pytest.raises(InterlockError):
        platform.plan_cullet({"incoming_tpd": 200.0})
    with pytest.raises(InterlockError):
        platform.plan_cullet(_supply(inventory_t=-1.0))


def test_unknown_grade_is_rejected(platform):
    with pytest.raises(ConfigError):
        platform.plan_cullet(_supply(grade="X"))


def test_console_endpoint_plans_cullet(platform):
    api = ConsoleAPI(platform)
    ok = api.plan_cullet(_supply())
    assert ok["ok"] is True
    assert ok["result"]["decision"]["ratio_pct"] == pytest.approx(71.826, abs=1e-2)
    bad = api.plan_cullet({"grade": "A"})
    assert bad["ok"] is False


def test_cullet_config_validation():
    mapping = config_to_mapping(default_config())
    broken = copy.deepcopy(mapping)
    del broken["limits"]["cullet"]
    with pytest.raises(ConfigError):
        PlatformConfig.from_mapping(broken)
    inverted = copy.deepcopy(mapping)
    inverted["limits"]["cullet"]["ratio_min_pct"] = 90.0
    with pytest.raises(ConfigError):
        PlatformConfig.from_mapping(inverted)
    weird = copy.deepcopy(mapping)
    weird["limits"]["cullet"]["q_cullet_gj_t"] = 9.9
    with pytest.raises(ConfigError):
        PlatformConfig.from_mapping(weird)


def test_yaml_cullet_section_matches_defaults():
    loaded = load_config(CONFIG_PATH)
    assert loaded.cullet == default_config().cullet


def test_status_exposes_cullet_snapshot(platform):
    platform.plan_cullet(_supply())
    status = platform.status()
    assert status["cullet"]["ratio_pct"] == pytest.approx(71.826, abs=1e-2)
    assert status["cullet"]["last"]["binding"] == "质量等级A级"
    assert "/api/cullet/plan" in status["endpoints"]["api"]
