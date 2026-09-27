"""碎玻璃掺配：库存定期望、热态定窗口、越限自动收回并说明原因。"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pytest

from glassmelt.app import CycleInputs
from glassmelt.config import PlatformConfig, config_to_mapping, default_config, load_config
from glassmelt.console.server import ConsoleServer
from glassmelt.errors import ConfigError, InterlockError

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "glassmelt.yaml"
)


def _post(url: str, payload: dict) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=5.0) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def test_rollback_to_thermal_ceiling_when_cullet_is_over_supplied(platform):
    payload = platform.decide_cullet(pull_tpd=60.0, yard_t=1600.0, inbound_tpd=40.0)
    decision = payload["decision"]
    assert decision["rolled_back"] is True
    assert decision["desired_ratio"] > decision["ratio"]
    assert decision["ratio"] == pytest.approx(decision["window_max"])
    assert decision["reasons"]
    assert "窑温压不住" in decision["reasons"][0]
    assert "最低稳定负荷" in decision["reasons"][0]
    assert decision["alarms"] == []
    assert payload["check"]["ok"] is True


def test_lift_to_thermal_floor_when_cullet_is_short(platform):
    payload = platform.decide_cullet(pull_tpd=72.0, yard_t=800.0, inbound_tpd=12.0)
    decision = payload["decision"]
    assert decision["rolled_back"] is True
    assert decision["ratio"] == pytest.approx(decision["window_min"])
    assert decision["ratio"] > decision["desired_ratio"]
    assert "化料跟不上" in decision["reasons"][0]
    assert decision["alarms"] == []


def test_ratio_inside_window_passes_through(platform):
    payload = platform.decide_cullet(pull_tpd=72.0, yard_t=1500.0, inbound_tpd=25.0)
    decision = payload["decision"]
    assert decision["rolled_back"] is False
    assert decision["ratio"] == pytest.approx(decision["desired_ratio"])
    assert decision["reasons"] == []
    assert decision["alarms"] == []
    assert payload["check"]["ok"] is True


def test_quality_cap_binds_when_thermal_window_is_wider(platform):
    payload = platform.decide_cullet(pull_tpd=72.0, yard_t=2400.0, inbound_tpd=60.0)
    decision = payload["decision"]
    assert decision["rolled_back"] is True
    assert decision["ratio"] == pytest.approx(platform.config.cullet.ratio_quality_max)
    assert decision["window_max"] > platform.config.cullet.ratio_quality_max
    assert "质量上限" in decision["reasons"][0]


def test_conflicting_limits_are_executed_on_thermal_side_with_alarm(platform):
    payload = platform.decide_cullet(pull_tpd=84.0, yard_t=1500.0, inbound_tpd=25.0)
    decision = payload["decision"]
    assert decision["ratio"] > platform.config.cullet.ratio_quality_max
    assert decision["alarms"]
    assert "质量上限被突破" in decision["alarms"][0]


def test_yard_runout_alarm_when_burning_down_stock(platform):
    payload = platform.decide_cullet(pull_tpd=72.0, yard_t=550.0, inbound_tpd=5.0)
    decision = payload["decision"]
    assert any("安全库存" in item for item in decision["alarms"])


def test_yard_overflow_alarm_when_stock_keeps_climbing(platform):
    payload = platform.decide_cullet(pull_tpd=60.0, yard_t=2900.0, inbound_tpd=45.0)
    decision = payload["decision"]
    assert any("逼近容量" in item for item in decision["alarms"])


def test_furnace_check_flags_unholdable_ratios(platform):
    high = platform.cullet.check(60.0, 0.8)
    assert high["ok"] is False
    assert "窑温压不住" in high["detail"]
    low = platform.cullet.check(72.0, 0.1)
    assert low["ok"] is False
    assert "化料跟不上" in low["detail"]
    inside = platform.cullet.check(72.0, 0.35)
    assert inside["ok"] is True
    assert "兜得住" in inside["detail"]


def test_decide_rejects_bad_inputs(platform):
    with pytest.raises(InterlockError):
        platform.cullet.decide(0.0, 1500.0, 25.0)
    with pytest.raises(InterlockError):
        platform.cullet.decide(72.0, -1.0, 25.0)
    with pytest.raises(InterlockError):
        platform.cullet.thermal_window(0.0)


def test_decision_event_is_published_with_reasons(platform):
    seen = []
    platform.bus.subscribe("batch", lambda event: seen.append(event))
    platform.decide_cullet(pull_tpd=60.0, yard_t=1600.0, inbound_tpd=40.0)
    decided = [event for event in seen if event.action == "cullet_ratio_decided"]
    assert decided
    event = decided[-1]
    assert event.payload["rolled_back"] is True
    assert event.payload["severity"] == "notice"
    assert event.payload["reasons"]


def test_ratio_change_enters_the_mix_plan(platform):
    platform.decide_cullet(pull_tpd=72.0, yard_t=1500.0, inbound_tpd=25.0)
    platform.scale.recalibrate(2.0, platform.clock.now())
    current = platform.cullet.current()
    plan = platform.mix.plan(platform.scale.baseline(), current)
    assert plan.cullet_ratio == pytest.approx(current.ratio)
    assert plan.cullet_epoch == current.epoch
    assert plan.cullet_kg == pytest.approx(plan.mix_total_kg * plan.cullet_ratio)
    assert plan.raw_kg == pytest.approx(plan.mix_total_kg - plan.cullet_kg)
    assert platform.mix.matches(
        plan, platform.scale.baseline(), platform.cullet.current()
    ) is True
    platform.decide_cullet(pull_tpd=60.0, yard_t=1600.0, inbound_tpd=40.0)
    assert platform.mix.matches(
        plan, platform.scale.baseline(), platform.cullet.current()
    ) is False


def test_startup_decides_cullet_before_mix_plan(platform):
    report = platform.start()
    steps = report["steps"]
    assert steps.index("cullet_decide") < steps.index("mix_plan")
    assert report["violations"] == []
    assert platform.cullet.current() is not None


def test_cycle_carries_cullet_decision_into_mix(running):
    result = running.cycle(
        CycleInputs(load=0.8, pull_rate=1.0, gob_mass_g=186.0, dt=1.0)
    )
    assert result["cullet"]["epoch"] == result["mix"]["cullet_epoch"]
    assert result["cullet"]["ratio"] == pytest.approx(result["mix"]["cullet_ratio"])
    assert "cullet_ratio" in running.trend.series_names()
    status = running.status()
    assert status["cullet"]["current"]["epoch"] == result["cullet"]["epoch"]
    assert running.readiness_report()["cullet_decided"] is True


def test_console_decide_endpoint(platform):
    server = ConsoleServer(platform)
    base = server.start()
    try:
        status, payload = _post(
            f"{base}/api/cullet/decide",
            {"pull_tpd": 60.0, "yard_t": 1600.0, "inbound_tpd": 40.0},
        )
        assert status == 200
        assert payload["ok"] is True
        assert payload["result"]["decision"]["rolled_back"] is True
        assert payload["result"]["check"]["ok"] is True
        status, payload = _post(f"{base}/api/cullet/decide", {})
        assert status == 200 and payload["ok"] is True
    finally:
        server.stop()


def test_loaded_yaml_matches_default_cullet():
    loaded = load_config(CONFIG_PATH)
    assert loaded.cullet == default_config().cullet


def test_cullet_config_rejects_inverted_energy_curve():
    mapping = config_to_mapping(default_config())
    mapping["limits"]["cullet"]["se_cullet_kj_kg"] = (
        mapping["limits"]["cullet"]["se_batch_kj_kg"]
    )
    with pytest.raises(ConfigError):
        PlatformConfig.from_mapping(mapping)


def test_cullet_config_rejects_bad_yard_band():
    mapping = config_to_mapping(default_config())
    mapping["limits"]["cullet"]["yard_min_t"] = mapping["limits"]["cullet"]["yard_capacity_t"]
    with pytest.raises(ConfigError):
        PlatformConfig.from_mapping(mapping)
