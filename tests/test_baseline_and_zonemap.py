"""投料秤基线与分区映射：标定与大修之后必须按新版本执行。"""

from __future__ import annotations

import pytest

from glassmelt.app import build_platform
from glassmelt.config import default_config
from glassmelt.errors import InterlockError, SequenceError


def _cullet(platform):
    """按额定工况出一版掺配决策，供配比计算使用。"""

    return platform.cullet.decide(
        platform.config.cullet.pull_rated_tpd,
        platform.config.cullet.yard_target_t,
        platform.config.cullet.inbound_default_tpd,
    )


def test_scale_baseline_is_persisted(platform):
    assert platform.scale.is_durable() is False
    baseline = platform.scale.recalibrate(5.0, platform.clock.now())
    assert baseline.epoch == 1
    assert platform.scale.is_durable() is True
    restored = platform.scale.restore()
    assert restored is True
    assert platform.scale.offset_kg == pytest.approx(5.0)
    assert platform.scale.epoch == 1


def test_scale_offset_beyond_limit_is_rejected(platform):
    limit = platform.config.batch.scale_offset_max_kg
    with pytest.raises(InterlockError):
        platform.scale.recalibrate(limit + 1.0, platform.clock.now())
    assert platform.scale.epoch == 0


def test_mix_plan_uses_the_current_baseline(platform):
    cullet = _cullet(platform)
    old = platform.scale.baseline()
    epoch_before = platform.scale.epoch
    stale_plan = platform.mix.plan(old, cullet)
    assert platform.mix.matches(stale_plan, platform.scale.baseline(), cullet) is True
    platform.recalibrate_scale(6.0)
    fresh = platform.scale.baseline()
    fresh_plan = platform.mix.plan(fresh, platform.cullet.current())
    assert fresh_plan.baseline_epoch == epoch_before + 1
    assert platform.mix.matches(stale_plan, fresh, platform.cullet.current()) is False
    assert platform.mix.matches(fresh_plan, fresh, platform.cullet.current()) is True
    assert fresh_plan.mix_total_kg == pytest.approx(
        platform.config.batch.charge_batch_kg + 6.0
    )


def test_recalibration_probe_reports_the_fresh_plan(platform):
    payload = platform.recalibrate_scale(3.5)
    assert payload["fresh"] is True
    assert payload["baseline"]["offset_kg"] == pytest.approx(3.5)
    assert payload["plan"]["baseline_epoch"] == payload["baseline"]["epoch"]


def test_mix_planner_tolerance_flag(platform):
    cullet = _cullet(platform)
    plan = platform.mix.plan(platform.scale.baseline(), cullet)
    assert platform.mix.tolerance_ok(plan) is True
    platform.scale.recalibrate(platform.config.batch.scale_offset_max_kg, platform.clock.now())
    wide = platform.mix.plan(platform.scale.baseline(), platform.cullet.current())
    assert wide.deviation_pct > platform.config.batch.mix_tolerance_pct
    assert platform.mix.tolerance_ok(wide) is False


def test_restore_from_disk_marks_baseline_restored(clock, tmp_path):
    root = str(tmp_path / "state")
    first = build_platform(default_config(), clock=clock, storage_root=root)
    first.scale.recalibrate(4.0, clock.now())
    second = build_platform(default_config(), clock=clock, storage_root=root)
    assert second.scale.restore() is True
    assert second.scale.offset_kg == pytest.approx(4.0)
    assert second.scale.is_durable() is True


def test_zone_map_rebuild_is_required_to_cover_all_zones(platform):
    with pytest.raises(InterlockError):
        platform.rebuild_zone_map({"zone-1": "burner-a"}, "缺分区")
    payload = platform.rebuild_zone_map(
        {
            zone: f"burner-{index + 20}"
            for index, zone in enumerate(platform.zones.zones())
        },
        "窑体大修",
    )
    assert payload["revision"] == 2
    assert all(item["revision"] == 2 for item in payload["zones"])


def test_trip_uses_the_current_zone_map(platform):
    platform.rebuild_zone_map(
        {
            zone: f"burner-{index + 40}"
            for index, zone in enumerate(platform.zones.zones())
        },
        "窑体大修",
    )
    outcome = platform.trip_burner("zone-2", "窑温越限")
    assert outcome["revision"] == 2
    assert outcome["burner_id"] == "burner-41"
    assert platform.trip.is_tripped() is True
    with pytest.raises(SequenceError):
        platform.trip_burner("zone-2", "重复跳闸")
    reset = platform.reset_trip()
    assert reset["was_tripped"] is True
    assert platform.trip.is_tripped() is False


def test_trip_records_a_stale_map_as_not_fresh(platform):
    platform.zones.rebuild(
        {
            zone: f"burner-{index + 60}"
            for index, zone in enumerate(platform.zones.zones())
        },
        "窑体大修",
    )
    outcome = platform.trip.trip("zone-1", "窑温越限")
    assert platform.zones.is_current(outcome.revision) is True
    assert platform.trip.zone_map_revision == outcome.revision
