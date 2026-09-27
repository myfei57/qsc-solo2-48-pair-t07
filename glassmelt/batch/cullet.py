"""碎玻璃掺配决策。

回收料多的时候窑温压不住、少的时候又浪费，根子是掺配比例只盯堆场、
不盯窑炉。这里的做法是把库存与熔化能力绑成一次决策：先按堆场库存与
到货算"期望掺配率"，再按当前拉引量做热平衡校核，求出窑炉兜得住的
"热态窗口"；期望率落不进窗口（或越过质量上限）就自动收回到边沿，
收回原因写进决策记录。配料计算只认最新一版决策，比例一变料单跟着变。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from ..clock import Clock
from ..config import BurnerLimits, CulletLimits
from ..errors import InterlockError
from ..eventbus import Event, EventBus


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class CulletDecision:
    """一版碎玻璃掺配决策。"""

    ratio: float
    desired_ratio: float
    window_min: float
    window_max: float
    quality_max: float
    gas_need_m3_h: float
    pull_tpd: float
    yard_t: float
    inbound_tpd: float
    reasons: tuple[str, ...]
    alarms: tuple[str, ...]
    epoch: int
    at: float

    @property
    def rolled_back(self) -> bool:
        """最终比例是否被迫偏离了库存侧期望。"""

        return abs(self.ratio - self.desired_ratio) > 1e-9

    def as_dict(self) -> dict:
        return {
            "ratio": round(self.ratio, 4),
            "desired_ratio": round(self.desired_ratio, 4),
            "rolled_back": self.rolled_back,
            "window_min": round(self.window_min, 4),
            "window_max": round(self.window_max, 4),
            "quality_max": round(self.quality_max, 4),
            "gas_need_m3_h": round(self.gas_need_m3_h, 2),
            "pull_tpd": round(self.pull_tpd, 3),
            "yard_t": round(self.yard_t, 1),
            "inbound_tpd": round(self.inbound_tpd, 2),
            "reasons": list(self.reasons),
            "alarms": list(self.alarms),
            "epoch": self.epoch,
            "at": round(self.at, 3),
        }


class CulletPlanner:
    """按回收料情况与熔化能力定掺配比例。"""

    def __init__(
        self,
        bus: EventBus,
        clock: Clock,
        limits: CulletLimits,
        burner: BurnerLimits,
    ) -> None:
        self._bus = bus
        self._clock = clock
        self._limits = limits
        self._burner = burner
        self._current: CulletDecision | None = None
        self._decisions = 0
        self._lock = threading.RLock()

    def gas_need_m3_h(self, pull_tpd: float, ratio: float) -> float:
        """给定拉引量与掺配率，算熔化所需燃气流量。

        熔化单耗按掺配率线性插值：全生料取 ``se_batch_kj_kg``，全碎玻璃
        取 ``se_cullet_kj_kg``——碎玻璃只补显热、没有分解反应热，掺配每
        高一截，熔化热耗就低一截。
        """

        ratio = _clamp01(ratio)
        specific = self._limits.se_batch_kj_kg * (1.0 - ratio) + (
            self._limits.se_cullet_kj_kg * ratio
        )
        kg_per_h = float(pull_tpd) * 1000.0 / 24.0
        return kg_per_h * specific / self._limits.gas_lhv_kj_m3

    def _window_raw(self, pull_tpd: float) -> tuple[float, float]:
        """热态窗口的未截断取值：由燃气负荷窗口反解掺配率。

        燃气需求随掺配率单调下降。需求不许低于最低稳定负荷（再低燃烧器
        压不下去，窑温压不住），也不许高于燃烧能力上限（化料跟不上），
        两个边界各反解出一个掺配率，就是窑炉兜得住的区间。
        """

        kg_per_h = float(pull_tpd) * 1000.0 / 24.0
        span = self._limits.se_batch_kj_kg - self._limits.se_cullet_kj_kg
        se_at_gas_max = (
            self._burner.gas_max_m3_h * self._limits.gas_lhv_kj_m3 / kg_per_h
        )
        se_at_gas_min = (
            self._limits.gas_min_stable_m3_h * self._limits.gas_lhv_kj_m3 / kg_per_h
        )
        r_min = (self._limits.se_batch_kj_kg - se_at_gas_max) / span
        r_max = (self._limits.se_batch_kj_kg - se_at_gas_min) / span
        return r_min, r_max

    def thermal_window(self, pull_tpd: float) -> tuple[float, float]:
        """当前拉引量下窑炉兜得住的掺配率窗口（截断到 [0, 1]）。"""

        if float(pull_tpd) <= 0.0:
            raise InterlockError("拉引量必须为正，无法做热态校核", component="batch")
        r_min, r_max = self._window_raw(pull_tpd)
        return _clamp01(r_min), _clamp01(r_max)

    def check(self, pull_tpd: float, ratio: float) -> dict:
        """窑温校核：该拉引量与掺配率下，窑炉兜不兜得住。"""

        gas = self.gas_need_m3_h(pull_tpd, ratio)
        window_min, window_max = self.thermal_window(pull_tpd)
        gas_max = self._burner.gas_max_m3_h
        gas_min = self._limits.gas_min_stable_m3_h
        ratio = _clamp01(ratio)
        if ratio > window_max:
            ok = False
            detail = (
                f"燃气需求 {gas:.0f} m³/h 低于最低稳定负荷 {gas_min:.0f} m³/h，"
                f"燃烧器压不下去，窑温压不住"
            )
        elif ratio < window_min:
            ok = False
            detail = (
                f"燃气需求 {gas:.0f} m³/h 超过燃烧能力上限 {gas_max:.0f} m³/h，"
                f"化料跟不上"
            )
        else:
            ok = True
            detail = (
                f"燃气需求 {gas:.0f} m³/h 落在 {gas_min:.0f}~{gas_max:.0f} m³/h "
                f"燃烧窗口内，窑温兜得住"
            )
        return {
            "ok": ok,
            "ratio": round(ratio, 4),
            "pull_tpd": round(float(pull_tpd), 3),
            "gas_need_m3_h": round(gas, 2),
            "gas_min_stable_m3_h": gas_min,
            "gas_max_m3_h": gas_max,
            "window_min": round(window_min, 4),
            "window_max": round(window_max, 4),
            "detail": detail,
        }

    def decide(self, pull_tpd: float, yard_t: float, inbound_tpd: float) -> CulletDecision:
        """出一版掺配决策：库存定期望，热态与质量定边界，越限自动收回。"""

        pull = float(pull_tpd)
        yard = float(yard_t)
        inbound = float(inbound_tpd)
        if pull <= 0.0:
            raise InterlockError("拉引量必须为正，无法做掺配决策", component="batch")
        if yard < 0.0 or inbound < 0.0:
            raise InterlockError("堆场库存与到货量不得为负", component="batch")

        limits = self._limits
        reasons: list[str] = []
        alarms: list[str] = []

        # 1) 库存侧期望：日到货加上库存向目标位的调节量，折算成掺配率。
        desired_use = inbound * limits.wash_yield + (
            yard - limits.yard_target_t
        ) / limits.adjust_days
        desired = _clamp01(desired_use / pull)

        # 2) 热态窗口：先按原始值识别"怎么调都兜不住"的极端工况。
        r_min_raw, r_max_raw = self._window_raw(pull)
        if r_min_raw > 1.0:
            alarms.append(
                "即使 100% 碎玻璃，熔化需气仍超燃烧能力上限，必须降拉引量"
            )
        if r_max_raw < 0.0:
            alarms.append(
                "即使全生料，熔化需气仍低于最低稳定负荷，窑温压不住，应提高拉引量"
            )
        window_min, window_max = _clamp01(r_min_raw), _clamp01(r_max_raw)

        # 3) 热态窗口与质量上限取交集，期望率落不进交集就收回到边沿。
        r_lo = window_min
        r_hi = min(window_max, limits.ratio_quality_max)
        if r_lo > r_hi:
            ratio = r_lo
            alarms.append(
                f"热态下限 {r_lo:.1%} 已高于可用上限 {r_hi:.1%}（质量上限 "
                f"{limits.ratio_quality_max:.1%}）：按热态下限执行，质量上限被突破，"
                f"建议降拉引量或换用成分更稳的碎玻璃"
            )
        else:
            ratio = min(max(desired, r_lo), r_hi)
            if ratio < desired - 1e-9:
                if window_max <= limits.ratio_quality_max:
                    reasons.append(
                        f"堆场期望掺配 {desired:.1%}（库存 {yard:.0f} t、"
                        f"日到货 {inbound:.0f} t），但热态上限 {r_hi:.1%}："
                        f"按期望率燃气需求仅 "
                        f"{self.gas_need_m3_h(pull, desired):.0f} m³/h，"
                        f"低于最低稳定负荷 {limits.gas_min_stable_m3_h:.0f} m³/h，"
                        f"窑温压不住，已自动收回至 {ratio:.1%}"
                    )
                else:
                    reasons.append(
                        f"堆场期望掺配 {desired:.1%}（库存 {yard:.0f} t、"
                        f"日到货 {inbound:.0f} t），超过质量上限 "
                        f"{limits.ratio_quality_max:.1%}：碎玻璃成分波动会带偏"
                        f"玻璃成分，已自动收回至 {ratio:.1%}"
                    )
            elif ratio > desired + 1e-9:
                reasons.append(
                    f"堆场只能支持 {desired:.1%}（库存 {yard:.0f} t、"
                    f"日到货 {inbound:.0f} t），低于热态下限 {r_lo:.1%}："
                    f"按期望率燃气需求 "
                    f"{self.gas_need_m3_h(pull, desired):.0f} m³/h 超过燃烧能力上限 "
                    f"{self._burner.gas_max_m3_h:.0f} m³/h，化料跟不上，"
                    f"已自动抬升至 {ratio:.1%}"
                )

        # 4) 库存平衡校核：按决策比例执行后，堆场是净增还是净耗。
        consumed = ratio * pull
        net = inbound * limits.wash_yield - consumed
        if net > 0.0:
            days_full = (limits.yard_capacity_t - yard) / net
            if days_full < limits.adjust_days:
                alarms.append(
                    f"按 {ratio:.1%} 执行后堆场仍日净增 {net:.1f} t，"
                    f"约 {days_full:.0f} 天逼近容量 {limits.yard_capacity_t:.0f} t，"
                    f"建议安排碎玻璃外售或提高拉引量"
                )
        elif net < 0.0:
            days_left = (yard - limits.yard_min_t) / (-net)
            if days_left < limits.adjust_days:
                alarms.append(
                    f"按 {ratio:.1%} 执行后堆场日净耗 {-net:.1f} t，"
                    f"约 {days_left:.0f} 天触及安全库存 {limits.yard_min_t:.0f} t，"
                    f"需外购碎玻璃或降拉引量"
                )

        with self._lock:
            self._decisions += 1
            decision = CulletDecision(
                ratio=ratio,
                desired_ratio=desired,
                window_min=window_min,
                window_max=window_max,
                quality_max=limits.ratio_quality_max,
                gas_need_m3_h=self.gas_need_m3_h(pull, ratio),
                pull_tpd=pull,
                yard_t=yard,
                inbound_tpd=inbound,
                reasons=tuple(reasons),
                alarms=tuple(alarms),
                epoch=self._decisions,
                at=self._clock.now(),
            )
            self._current = decision
        severity = "warning" if alarms else ("notice" if reasons else "info")
        self._bus.publish(
            Event.of(
                "batch",
                "batch",
                "cullet_ratio_decided",
                subject="cullet_ratio",
                severity=severity,
                ratio=round(ratio, 4),
                desired_ratio=round(desired, 4),
                rolled_back=decision.rolled_back,
                window_min=round(window_min, 4),
                window_max=round(window_max, 4),
                reasons=list(reasons),
                alarms=list(alarms),
                epoch=decision.epoch,
            )
        )
        return decision

    def current(self) -> CulletDecision | None:
        """返回最新一版决策；还没有决策时返回 None。"""

        with self._lock:
            return self._current

    def snapshot(self) -> dict:
        with self._lock:
            current = self._current
            decisions = self._decisions
        return {
            "decisions": decisions,
            "quality_max": self._limits.ratio_quality_max,
            "gas_min_stable_m3_h": self._limits.gas_min_stable_m3_h,
            "current": current.as_dict() if current is not None else None,
        }
