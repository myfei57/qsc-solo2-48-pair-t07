"""碎玻璃掺配决策。

回收料多少与熔化能力共同决定熟料比：先按回收料可供量、质量等级与工艺制度
算出物料侧目标比例，再做窑温校核——估算该比例下的入炉热负荷与熔化温度。
热负荷跌破最低稳定燃烧（燃料已压到最低、窑温还是压不住）或冲出燃料系统
最大供热（熔化能力不足），都说明这个比例窑炉兜不住，于是自动收回到可行
窗口边并留安全裕量，收回原因随决策一起落事件，事后能说清"为什么是这个
比例"。定下来的比例直接进配比计算，配料单按新比例出。
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

from ..config import CulletLimits, FurnaceLimits
from ..errors import InterlockError
from ..eventbus import Event, EventBus


@dataclass(frozen=True)
class CulletSupply:
    """回收料（碎玻璃）供应情况。"""

    inventory_t: float
    incoming_tpd: float
    grade: str = "B"
    moisture: float = 0.01
    contamination: float = 0.01

    @classmethod
    def from_mapping(cls, payload: dict) -> "CulletSupply":
        """从控制台／调用方给的映射构造供应情况，缺项或非法值一律拒绝。"""

        if not isinstance(payload, dict):
            raise InterlockError("回收料供应参数必须是映射", component="batch")
        try:
            supply = cls(
                inventory_t=float(payload["inventory_t"]),
                incoming_tpd=float(payload["incoming_tpd"]),
                grade=str(payload.get("grade", "B")),
                moisture=float(payload.get("moisture", 0.01)),
                contamination=float(payload.get("contamination", 0.01)),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise InterlockError(
                f"回收料供应参数不完整或非法: {error}", component="batch"
            ) from error
        if supply.inventory_t < 0 or supply.incoming_tpd < 0:
            raise InterlockError("回收料库存与到货量不得为负", component="batch")
        if not 0.0 <= supply.moisture < 1.0 or not 0.0 <= supply.contamination < 1.0:
            raise InterlockError("回收料含水率与杂质率必须在 [0, 1) 内", component="batch")
        return supply

    def effective_factor(self) -> float:
        """有效利用率：扣除水分与杂质后真正能进窑的部分。"""

        return (1.0 - self.moisture) * (1.0 - self.contamination)

    def sustainable_ratio_pct(self, limits: CulletLimits) -> float:
        """按保障天数可持续支撑的熟料比（%）。"""

        usable_tpd = (
            self.incoming_tpd + self.inventory_t / limits.horizon_days
        ) * self.effective_factor()
        if limits.pull_tpd <= 0:
            return 0.0
        return max(0.0, min(100.0, usable_tpd / limits.pull_tpd * 100.0))

    def coverage_days(self, ratio_pct: float, limits: CulletLimits) -> float:
        """按给定熟料比，库存结合日到货可维持的天数（到货可覆盖时为无穷）。"""

        need_tpd = ratio_pct / 100.0 * limits.pull_tpd / self.effective_factor()
        net_draw = need_tpd - self.incoming_tpd
        if net_draw <= 0:
            return math.inf
        return self.inventory_t / net_draw


@dataclass(frozen=True)
class Violation:
    """一项窑温校核越限。"""

    name: str
    message: str
    direction: str  # "reduce" 需降低比例 / "raise" 需提高比例

    def as_dict(self) -> dict:
        return {"name": self.name, "message": self.message, "direction": self.direction}


@dataclass(frozen=True)
class Adjustment:
    """一次比例自动调整记录。"""

    constraint: str
    from_pct: float
    to_pct: float
    reason: str

    def as_dict(self) -> dict:
        return {
            "constraint": self.constraint,
            "from_pct": round(self.from_pct, 3),
            "to_pct": round(self.to_pct, 3),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CulletDecision:
    """一次掺配决策的完整结论。"""

    ratio_pct: float
    desired_pct: float
    caps: dict
    binding: str
    window_lo_pct: float
    window_hi_pct: float
    adjustments: tuple[Adjustment, ...]
    alarms: tuple[str, ...]
    violations: tuple[Violation, ...]
    heat_gj_h: float
    est_temp_c: float
    coverage_days: float

    def as_dict(self) -> dict:
        return {
            "ratio_pct": round(self.ratio_pct, 3),
            "desired_pct": round(self.desired_pct, 3),
            "caps": {name: round(value, 3) for name, value in self.caps.items()},
            "binding": self.binding,
            "window": {"lo_pct": round(self.window_lo_pct, 3), "hi_pct": round(self.window_hi_pct, 3)},
            "adjustments": [item.as_dict() for item in self.adjustments],
            "alarms": list(self.alarms),
            "violations": [item.as_dict() for item in self.violations],
            "heat_gj_h": round(self.heat_gj_h, 3),
            "est_temp_c": round(self.est_temp_c, 2),
            "coverage_days": round(self.coverage_days, 1)
            if math.isfinite(self.coverage_days)
            else -1.0,
        }


class CulletPlanner:
    """按回收料与熔化能力定熟料比，并做窑温校核与自动收回。"""

    def __init__(self, bus: EventBus, cullet: CulletLimits, furnace: FurnaceLimits) -> None:
        self._bus = bus
        self._limits = cullet
        self._furnace = furnace
        self._ratio_pct = 0.0
        self._decisions = 0
        self._last: CulletDecision | None = None
        self._lock = threading.RLock()
        # 入炉热负荷需求对熟料比是线性的：Q(r) = q0 + q1·r（r 以百分点计）
        tph = cullet.pull_tpd / 24.0
        self._q0 = (
            tph * cullet.q_batch_gj_t / cullet.thermal_efficiency + cullet.fixed_loss_gj_h
        )
        self._q1 = (
            -tph
            * (cullet.q_batch_gj_t - cullet.q_cullet_gj_t)
            / cullet.thermal_efficiency
            / 100.0
        )

    @property
    def ratio_pct(self) -> float:
        """当前生效的熟料比，配比计算按它出料单。"""

        with self._lock:
            return self._ratio_pct

    def heat_demand_gj_h(self, ratio_pct: float) -> float:
        """给定熟料比下的入炉热负荷需求 GJ/h。"""

        return self._q0 + self._q1 * float(ratio_pct)

    def est_melt_temp_c(self, ratio_pct: float) -> float:
        """给定熟料比下的熔化温度估算（参考工况附近线性化）。"""

        limits = self._limits
        return (
            self._furnace.melt_temp_target_c
            + limits.temp_per_gj * (self.heat_demand_gj_h(ratio_pct) - limits.ref_heat_gj_h)
            + limits.temp_per_ratio_pct * (float(ratio_pct) - limits.ref_ratio_pct)
        )

    def feasible_window_pct(self) -> dict:
        """解析求解窑炉兜得住的熟料比窗口（各约束对比例都是线性的）。"""

        limits = self._limits
        uppers = {"最低稳定燃烧负荷": (limits.fuel_min_gj_h - self._q0) / self._q1}
        lowers = {"燃料系统最大供热": (limits.fuel_max_gj_h - self._q0) / self._q1}
        # 熔化温度 T(r) = t0 + t1·r
        t1 = limits.temp_per_gj * self._q1 + limits.temp_per_ratio_pct
        t0 = (
            self._furnace.melt_temp_target_c
            + limits.temp_per_gj * (self._q0 - limits.ref_heat_gj_h)
            - limits.temp_per_ratio_pct * limits.ref_ratio_pct
        )
        if t1 > 0:
            uppers["熔化温度上限"] = (self._furnace.melt_temp_max_c - t0) / t1
            lowers["熔化温度下限"] = (self._furnace.melt_temp_min_c - t0) / t1
        elif t1 < 0:
            lowers["熔化温度上限"] = (self._furnace.melt_temp_max_c - t0) / t1
            uppers["熔化温度下限"] = (self._furnace.melt_temp_min_c - t0) / t1
        hi_binding = min(uppers, key=lambda name: uppers[name])
        lo_binding = max(lowers, key=lambda name: lowers[name])
        return {
            "lo_pct": max(0.0, lowers[lo_binding]),
            "hi_pct": min(100.0, uppers[hi_binding]),
            "lo_binding": lo_binding,
            "hi_binding": hi_binding,
        }

    def check(self, ratio_pct: float) -> list[Violation]:
        """校核给定熟料比下窑温兜不兜得住，返回越限项（空表示兜得住）。"""

        limits = self._limits
        ratio = float(ratio_pct)
        out: list[Violation] = []
        heat = self.heat_demand_gj_h(ratio)
        if heat < limits.fuel_min_gj_h:
            out.append(
                Violation(
                    "最低稳定燃烧负荷",
                    f"熟料比 {ratio:.1f}% 时入炉热负荷需求 {heat:.1f} GJ/h，"
                    f"已低于最低稳定燃烧 {limits.fuel_min_gj_h:.1f} GJ/h，"
                    f"燃料压到最低窑温还是压不住",
                    "reduce",
                )
            )
        if heat > limits.fuel_max_gj_h:
            out.append(
                Violation(
                    "燃料系统最大供热",
                    f"熟料比 {ratio:.1f}% 时入炉热负荷需求 {heat:.1f} GJ/h，"
                    f"超过燃料系统最大供热 {limits.fuel_max_gj_h:.1f} GJ/h，熔化能力不足",
                    "raise",
                )
            )
        temp = self.est_melt_temp_c(ratio)
        if temp > self._furnace.melt_temp_max_c:
            out.append(
                Violation(
                    "熔化温度上限",
                    f"熟料比 {ratio:.1f}% 时熔化温度估算 {temp:.1f} ℃，"
                    f"超过上限 {self._furnace.melt_temp_max_c:.1f} ℃，窑体与玻璃液都兜不住",
                    "reduce",
                )
            )
        if temp < self._furnace.melt_temp_min_c:
            out.append(
                Violation(
                    "熔化温度下限",
                    f"熟料比 {ratio:.1f}% 时熔化温度估算 {temp:.1f} ℃，"
                    f"低于下限 {self._furnace.melt_temp_min_c:.1f} ℃，化料不透",
                    "raise",
                )
            )
        return out

    def decide(self, supply: CulletSupply) -> CulletDecision:
        """定一次熟料比：物料侧定目标 → 窑温校核 → 兜不住自动收回并说明原因。"""

        limits = self._limits
        caps = {
            "回收料供应": supply.sustainable_ratio_pct(limits),
            f"质量等级{supply.grade.strip().upper()}级": limits.grade_cap(supply.grade),
            "工艺制度上限": limits.ratio_max_pct,
        }
        binding = min(caps, key=lambda name: caps[name])
        desired = caps[binding]
        window = self.feasible_window_pct()
        adjustments: list[Adjustment] = []
        alarms: list[str] = []
        ratio = desired

        blocking = self.check(desired)
        reduce_side = [item for item in blocking if item.direction == "reduce"]
        raise_side = [item for item in blocking if item.direction == "raise"]
        if reduce_side:
            rolled = max(window["hi_pct"] - limits.safety_margin_pct, 0.0)
            reasons = "；".join(item.message for item in reduce_side)
            adjustments.append(
                Adjustment(
                    constraint=window["hi_binding"],
                    from_pct=desired,
                    to_pct=rolled,
                    reason=(
                        f"{reasons}。受「{window['hi_binding']}」约束，窑炉可行上限 "
                        f"{window['hi_pct']:.1f}%，自动收回到 {rolled:.1f}%"
                        f"（留 {limits.safety_margin_pct:.1f} 个百分点安全裕量）"
                    ),
                )
            )
            ratio = rolled
            # 收回后复核，仍兜不住就按裕量继续往下收
            for _ in range(5):
                leftover = [item for item in self.check(ratio) if item.direction == "reduce"]
                if not leftover:
                    break
                ratio = max(ratio - limits.safety_margin_pct, 0.0)
            else:
                alarms.append("连续回调仍无法让窑温回到可行窗口，请检查窑炉状态")
        elif raise_side:
            lifted = window["lo_pct"] + limits.safety_margin_pct
            if all(cap >= lifted for cap in caps.values()):
                reasons = "；".join(item.message for item in raise_side)
                adjustments.append(
                    Adjustment(
                        constraint=window["lo_binding"],
                        from_pct=desired,
                        to_pct=lifted,
                        reason=(
                            f"{reasons}。受「{window['lo_binding']}」约束，窑炉可行下限 "
                            f"{window['lo_pct']:.1f}%，自动上调到 {lifted:.1f}%"
                        ),
                    )
                )
                ratio = lifted
            else:
                alarms.append(
                    "；".join(item.message for item in raise_side)
                    + "。回收料供应不足以补上缺口，建议降低拉引量或紧急补充碎玻璃"
                )
        if ratio < limits.ratio_min_pct:
            alarms.append(
                f"熟料比 {ratio:.1f}% 低于工艺下限 {limits.ratio_min_pct:.1f}%："
                f"回收料供应不足，按可供量执行，注意配合料混合均匀性，建议补充外购碎玻璃"
            )

        remaining = tuple(self.check(ratio))
        for item in remaining:
            alarms.append(f"最终校核仍越限：{item.message}")
        decision = CulletDecision(
            ratio_pct=ratio,
            desired_pct=desired,
            caps=caps,
            binding=binding,
            window_lo_pct=window["lo_pct"],
            window_hi_pct=window["hi_pct"],
            adjustments=tuple(adjustments),
            alarms=tuple(alarms),
            violations=remaining,
            heat_gj_h=self.heat_demand_gj_h(ratio),
            est_temp_c=self.est_melt_temp_c(ratio),
            coverage_days=supply.coverage_days(ratio, limits),
        )
        with self._lock:
            self._ratio_pct = ratio
            self._decisions += 1
            self._last = decision
        self._publish(decision)
        return decision

    def _publish(self, decision: CulletDecision) -> None:
        self._bus.publish(
            Event.of(
                "batch",
                "batch",
                "cullet_ratio_decided",
                subject="cullet_planner",
                severity="notice",
                ratio_pct=round(decision.ratio_pct, 3),
                desired_pct=round(decision.desired_pct, 3),
                binding=decision.binding,
            )
        )
        for adjustment in decision.adjustments:
            self._bus.publish(
                Event.of(
                    "batch",
                    "batch",
                    "cullet_ratio_adjusted",
                    subject="cullet_planner",
                    severity="warning",
                    constraint=adjustment.constraint,
                    from_pct=round(adjustment.from_pct, 3),
                    to_pct=round(adjustment.to_pct, 3),
                    reason=adjustment.reason,
                )
            )

    def snapshot(self) -> dict:
        with self._lock:
            window = self.feasible_window_pct()
            return {
                "ratio_pct": round(self._ratio_pct, 3),
                "decisions": self._decisions,
                "window": {
                    "lo_pct": round(window["lo_pct"], 3),
                    "hi_pct": round(window["hi_pct"], 3),
                    "lo_binding": window["lo_binding"],
                    "hi_binding": window["hi_binding"],
                },
                "ratio_min_pct": self._limits.ratio_min_pct,
                "ratio_max_pct": self._limits.ratio_max_pct,
                "last": self._last.as_dict() if self._last is not None else None,
            }
