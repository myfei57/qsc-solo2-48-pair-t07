"""配比计算。

生料配比按投料秤的当前标定基线折算：秤的标定偏移一变，配比必须跟着变，
否则料方长期按老偏移配料，窑内玻璃成分漂移、析晶发白。碎玻璃掺配同理：
掺配决策一换版，每付料里碎玻璃与生料的拆分必须跟着换版，否则窑温与
料单对不上。所以配比计划要同时带上它用的基线代次与决策代次，装配与
控制台按两个代次核对"这一批是不是按最新基线、最新掺配决策算的"。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from ..config import BatchLimits
from ..eventbus import Event, EventBus


@dataclass(frozen=True)
class MixPlan:
    """一份配比计划。"""

    batch_kg: float
    offset_kg: float
    baseline_epoch: int
    cullet_ratio: float
    cullet_epoch: int
    cullet_kg: float
    raw_kg: float
    mix_total_kg: float
    deviation_pct: float

    def as_dict(self) -> dict:
        return {
            "batch_kg": round(self.batch_kg, 3),
            "offset_kg": round(self.offset_kg, 3),
            "baseline_epoch": self.baseline_epoch,
            "cullet_ratio": round(self.cullet_ratio, 4),
            "cullet_epoch": self.cullet_epoch,
            "cullet_kg": round(self.cullet_kg, 3),
            "raw_kg": round(self.raw_kg, 3),
            "mix_total_kg": round(self.mix_total_kg, 3),
            "deviation_pct": round(self.deviation_pct, 4),
        }


class MixPlanner:
    """按投料秤基线与掺配决策折算配比。"""

    def __init__(self, bus: EventBus, limits: BatchLimits) -> None:
        self._bus = bus
        self._limits = limits
        self._plans = 0
        self._lock = threading.RLock()

    def plan(self, baseline, cullet) -> MixPlan:
        """按给定基线与掺配决策折算一批配比。"""

        batch_kg = self._limits.charge_batch_kg
        offset_kg = float(baseline.offset_kg)
        epoch = int(baseline.epoch)
        cullet_ratio = float(cullet.ratio)
        cullet_epoch = int(cullet.epoch)
        mix_total_kg = batch_kg + offset_kg
        cullet_kg = mix_total_kg * cullet_ratio
        raw_kg = mix_total_kg - cullet_kg
        deviation_pct = (offset_kg / batch_kg) * 100.0
        result = MixPlan(
            batch_kg=batch_kg,
            offset_kg=offset_kg,
            baseline_epoch=epoch,
            cullet_ratio=cullet_ratio,
            cullet_epoch=cullet_epoch,
            cullet_kg=cullet_kg,
            raw_kg=raw_kg,
            mix_total_kg=mix_total_kg,
            deviation_pct=deviation_pct,
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
                cullet_epoch=cullet_epoch,
                cullet_ratio=round(cullet_ratio, 4),
                offset_kg=round(offset_kg, 3),
                deviation_pct=round(deviation_pct, 4),
            )
        )
        return result

    def matches(self, plan: MixPlan, baseline, cullet) -> bool:
        """核对配比计划是否用的就是当前基线与当前掺配决策。"""

        return (
            plan.baseline_epoch == int(baseline.epoch)
            and plan.offset_kg == float(baseline.offset_kg)
            and plan.cullet_epoch == int(cullet.epoch)
            and plan.cullet_ratio == float(cullet.ratio)
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
