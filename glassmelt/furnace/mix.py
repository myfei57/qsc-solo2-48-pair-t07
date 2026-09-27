"""配比计算。

生料配比按投料秤的当前标定基线折算：秤的标定偏移一变，配比必须跟着变，
否则料方长期按老偏移配料，窑内玻璃成分漂移、析晶发白。熟料比同理：碎玻璃
掺配比例一进配比计划就参与称量（碎玻璃按入炉掺配比单独称量，生料仍按秤
基线折算），比例变了还按老计划配料就是这份数据里要防的第二个错。所以配比
计划里要带上它用的基线代次与熟料比，装配与控制台都按它们核对"这一批是不
是按最新基线、最新比例算的"。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from ..config import BatchLimits
from ..errors import InterlockError
from ..eventbus import Event, EventBus


@dataclass(frozen=True)
class MixPlan:
    """一份配比计划。"""

    batch_kg: float
    offset_kg: float
    baseline_epoch: int
    mix_total_kg: float
    deviation_pct: float
    cullet_ratio_pct: float
    cullet_kg: float
    feed_total_kg: float

    def as_dict(self) -> dict:
        return {
            "batch_kg": round(self.batch_kg, 3),
            "offset_kg": round(self.offset_kg, 3),
            "baseline_epoch": self.baseline_epoch,
            "mix_total_kg": round(self.mix_total_kg, 3),
            "deviation_pct": round(self.deviation_pct, 4),
            "cullet_ratio_pct": round(self.cullet_ratio_pct, 3),
            "cullet_kg": round(self.cullet_kg, 3),
            "feed_total_kg": round(self.feed_total_kg, 3),
        }


class MixPlanner:
    """按投料秤基线与熟料比折算配比。"""

    def __init__(self, bus: EventBus, limits: BatchLimits) -> None:
        self._bus = bus
        self._limits = limits
        self._plans = 0
        self._lock = threading.RLock()

    def plan(self, baseline, cullet_ratio_pct: float = 0.0) -> MixPlan:
        """按给定基线与熟料比折算一批配比。"""

        ratio = float(cullet_ratio_pct)
        if not 0.0 <= ratio < 100.0:
            raise InterlockError(
                f"熟料比 {ratio:.1f}% 非法，禁止折算配比", component="furnace"
            )
        batch_kg = self._limits.charge_batch_kg
        offset_kg = float(baseline.offset_kg)
        epoch = int(baseline.epoch)
        mix_total_kg = batch_kg + offset_kg
        deviation_pct = (offset_kg / batch_kg) * 100.0
        cullet_kg = mix_total_kg * ratio / (100.0 - ratio)
        result = MixPlan(
            batch_kg=batch_kg,
            offset_kg=offset_kg,
            baseline_epoch=epoch,
            mix_total_kg=mix_total_kg,
            deviation_pct=deviation_pct,
            cullet_ratio_pct=ratio,
            cullet_kg=cullet_kg,
            feed_total_kg=mix_total_kg + cullet_kg,
        )
        with self._lock:
            self._plans += 1
        self._bus.publish(
            Event.of(
                "furnace",
                "furnace",
                "mix_planned",
                subject="mix_ratio",
                severity="notice" if not self._limits.mix_ok(deviation_pct) else "info",
                baseline_epoch=epoch,
                offset_kg=round(offset_kg, 3),
                deviation_pct=round(deviation_pct, 4),
                cullet_ratio_pct=round(ratio, 3),
            )
        )
        return result

    def matches(self, plan: MixPlan, baseline, cullet_ratio_pct: float = 0.0) -> bool:
        """核对配比计划是否用的就是当前基线与当前熟料比。"""

        return (
            plan.baseline_epoch == int(baseline.epoch)
            and plan.offset_kg == float(baseline.offset_kg)
            and plan.cullet_ratio_pct == float(cullet_ratio_pct)
        )

    def tolerance_ok(self, plan: MixPlan) -> bool:
        """配比偏差是否还在公差内。"""

        return self._limits.mix_ok(plan.deviation_pct)

    def snapshot(self, plan: MixPlan | None = None) -> dict:
        with self._lock:
            payload = {
                "plans": self._plans,
                "tolerance_pct": self._limits.mix_tolerance_pct,
            }
        if plan is not None:
            payload["latest"] = plan.as_dict()
            payload["in_tolerance"] = self.tolerance_ok(plan)
        return payload
