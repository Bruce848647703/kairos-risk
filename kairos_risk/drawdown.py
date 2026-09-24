"""回撤分析模块：水下曲线、最大回撤、回撤区间（持续期/恢复期）、Calmar 与持续期分布。

定义
----
以初始净值 1.0 为起点，净值曲线 ``W_t = ∏_{s≤t}(1+r_s)``，历史最高净值
``P_t = max(1, W_0..W_t)``，则

    回撤 dd_t = W_t/P_t − 1 ≤ 0        （「水下曲线」）
    最大回撤 MDD = max(−dd_t) ≥ 0      （返回正数幅度）

一个**回撤区间（episode）**从净值创新高开始，经历下跌到谷底，直到净值收复前高为止；
未收复的最后一个区间标记为 ``recovered=False``（即「当前回撤」）。
区间统计区分「下跌期」（峰 -> 谷）与「恢复期」（谷 -> 收复），二者之和为水下总时长。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ._util import ReturnsLike, as_returns_frame, check_positive_periods, describe_seq


def _series(returns: ReturnsLike) -> pd.Series:
    """把收益输入归一为单列 Series（保留期次索引）。"""
    frame = as_returns_frame(returns)
    if frame.shape[1] != 1:
        raise ValueError("回撤分析只接受单资产收益序列（Series / 一维 ndarray / 单列 DataFrame）")
    return frame.iloc[:, 0]


def wealth_curve(returns: ReturnsLike, start: float = 1.0) -> pd.Series:
    """净值曲线 W_t = start·∏(1+r)，index 与输入对齐。"""
    r = _series(returns)
    if np.any(r.to_numpy() <= -1.0):
        raise ValueError("存在 ≤ −100% 的收益，净值曲线无定义")
    base = float(start)
    if base <= 0:
        raise ValueError("start 必须为正")
    return pd.Series(base * np.cumprod(1.0 + r.to_numpy(dtype="float64")),
                     index=r.index, name="wealth")


def running_peak(returns: ReturnsLike, start: float = 1.0) -> pd.Series:
    """历史最高净值 P_t = max(start, W_0..W_t)。"""
    w = wealth_curve(returns, start).to_numpy(dtype="float64")
    peak = np.maximum.accumulate(np.maximum(w, float(start)))
    return pd.Series(peak, index=wealth_curve(returns, start).index, name="peak")


def drawdown_series(returns: ReturnsLike, start: float = 1.0) -> pd.Series:
    """回撤序列 dd_t = W_t/P_t − 1（≤ 0，负数表示相对前高的跌幅）。"""
    w = wealth_curve(returns, start)
    peak = np.maximum.accumulate(np.maximum(w.to_numpy(dtype="float64"), float(start)))
    return pd.Series(w.to_numpy(dtype="float64") / peak - 1.0, index=w.index, name="drawdown")


def underwater(returns: ReturnsLike, start: float = 1.0) -> pd.DataFrame:
    """水下曲线数据表：wealth / peak / drawdown / in_drawdown / periods_underwater。

    ``periods_underwater`` 为「本轮已在水下持续的期数」（创新高即归零），
    便于直接观察回撤老化程度。
    """
    w = wealth_curve(returns, start)
    arr = w.to_numpy(dtype="float64")
    peak = np.maximum.accumulate(np.maximum(arr, float(start)))
    dd = arr / peak - 1.0
    below = dd < 0.0
    runs = np.zeros(len(dd), dtype="int64")
    counter = 0
    for i, flag in enumerate(below):
        counter = counter + 1 if flag else 0
        runs[i] = counter
    return pd.DataFrame({
        "wealth": arr,
        "peak": peak,
        "drawdown": dd,
        "in_drawdown": below,
        "periods_underwater": runs,
    }, index=w.index)


def max_drawdown(returns: ReturnsLike, start: float = 1.0) -> float:
    """最大回撤（正数幅度）：MDD = max(1 − W_t/P_t)。无回撤时返回 0（不会出现 −0.0）。"""
    dd = drawdown_series(returns, start).to_numpy(dtype="float64")
    # dd 恒 ≤ 0；max(0.0, ·) 同时把「无回撤」时的 −0.0 归一为 0.0
    return max(0.0, float(-dd.min())) if dd.size else 0.0


def current_drawdown(returns: ReturnsLike, start: float = 1.0) -> float:
    """当前回撤（正数幅度）：最后一期相对历史前高的跌幅；处于新高时为 0。"""
    dd = drawdown_series(returns, start).to_numpy(dtype="float64")
    return max(0.0, float(-dd[-1])) if dd.size else 0.0


def longest_drawdown(returns: ReturnsLike, start: float = 1.0) -> int:
    """最长水下持续期（期数）：连续处于前高之下的最长段长度。"""
    uw = underwater(returns, start)
    return int(uw["periods_underwater"].max()) if len(uw) else 0


def time_underwater_ratio(returns: ReturnsLike, start: float = 1.0) -> float:
    """水下时间占比：处于回撤状态的期数 / 总期数。"""
    uw = underwater(returns, start)
    return float(uw["in_drawdown"].mean()) if len(uw) else 0.0


@dataclass(frozen=True, eq=False)
class DrawdownEpisode:
    """一个完整的回撤区间。

    字段
    ----
    start / trough / end: 峰值期、谷底期、收复期的索引标签（``end=None`` 表示尚未收复）；
    start_pos / trough_pos / end_pos: 对应的位置序号（``start_pos=-1`` 表示起点净值 1.0，
                          ``end_pos=-1`` 表示未收复）；
    peak_value / trough_value: 峰值与谷底净值；
    depth:      回撤深度（正数幅度）= 1 − trough/peak；
    down_periods:    下跌期数（峰 -> 谷）；
    recovery_periods: 恢复期数（谷 -> 收复前高），未收复时为 ``nan``；
    total_periods:   水下总期数（已收复为峰 -> 收复；未收复为峰 -> 样本末尾）；
    recovered:  是否已收复前高。
    """

    start: Any
    trough: Any
    end: Optional[Any]
    start_pos: int
    trough_pos: int
    end_pos: int
    peak_value: float
    trough_value: float
    depth: float
    down_periods: int
    recovery_periods: float
    total_periods: int
    recovered: bool

    def as_dict(self) -> Dict[str, Any]:
        """转成可放进 DataFrame 的字典。"""
        return {
            "start": self.start, "trough": self.trough, "end": self.end,
            "peak_value": self.peak_value, "trough_value": self.trough_value,
            "depth": self.depth, "down_periods": self.down_periods,
            "recovery_periods": self.recovery_periods, "total_periods": self.total_periods,
            "recovered": self.recovered,
        }


def drawdown_episodes(returns: ReturnsLike, min_depth: float = 0.0,
                      start: float = 1.0) -> List[DrawdownEpisode]:
    """扫描净值曲线，切分出全部回撤区间（含尚未收复的当前区间）。

    ``min_depth`` 用于过滤噪声级小回撤（按正数幅度比较，如 0.02 表示只看 ≥2% 的回撤）。
    """
    r = _series(returns)
    idx = r.index
    arr = r.to_numpy(dtype="float64")
    if np.any(arr <= -1.0):
        raise ValueError("存在 ≤ −100% 的收益，净值曲线无定义")
    n = arr.size
    wealth = float(start) * np.cumprod(1.0 + arr)
    min_depth = float(min_depth)
    if min_depth < 0:
        raise ValueError("min_depth 不能为负")

    episodes: List[DrawdownEpisode] = []
    peak_val = float(start)
    peak_pos = -1                      # −1 表示起点净值（第一期期初）
    trough_val = float(start)
    trough_pos = -1
    in_dd = False

    def _close(end_pos: int, recovered: bool) -> None:
        depth = 1.0 - trough_val / peak_val
        if depth < min_depth - 1e-15:
            return
        last = end_pos if end_pos >= 0 else n - 1
        rec = float(end_pos - trough_pos) if recovered else float("nan")
        episodes.append(DrawdownEpisode(
            start=idx[peak_pos] if peak_pos >= 0 else (idx[0] if n else None),
            trough=idx[trough_pos] if trough_pos >= 0 else (idx[0] if n else None),
            end=idx[end_pos] if end_pos >= 0 else None,
            start_pos=peak_pos,
            trough_pos=trough_pos,
            end_pos=end_pos,
            peak_value=peak_val,
            trough_value=trough_val,
            depth=float(depth),
            down_periods=int(trough_pos - peak_pos),
            recovery_periods=rec,
            total_periods=int(last - peak_pos),
            recovered=recovered,
        ))

    for t in range(n):
        w_t = float(wealth[t])
        if w_t >= peak_val:
            if in_dd:
                _close(t, True)
                in_dd = False
            peak_val = w_t
            peak_pos = t
            trough_val = w_t
            trough_pos = t
        else:
            if not in_dd:
                in_dd = True
                trough_val = w_t
                trough_pos = t
            elif w_t < trough_val:
                trough_val = w_t
                trough_pos = t
    if in_dd:
        _close(-1, False)
    return episodes


def top_drawdowns(returns: ReturnsLike, n: int = 5, min_depth: float = 0.0,
                  start: float = 1.0) -> List[DrawdownEpisode]:
    """按深度降序返回最大的 n 个回撤区间。"""
    if int(n) <= 0:
        raise ValueError("n 必须为正")
    eps = sorted(drawdown_episodes(returns, min_depth, start),
                 key=lambda e: e.depth, reverse=True)
    return eps[:int(n)]


def episodes_frame(episodes: List[DrawdownEpisode]) -> pd.DataFrame:
    """把回撤区间列表整理成 DataFrame（index=峰值期标签，按时间顺序）。"""
    if not episodes:
        return pd.DataFrame(columns=["start", "trough", "end", "peak_value", "trough_value",
                                     "depth", "down_periods", "recovery_periods",
                                     "total_periods", "recovered"])
    rows = [e.as_dict() for e in episodes]
    out = pd.DataFrame(rows)
    out.index = [e.start for e in episodes]
    out.index.name = "peak"
    return out


def duration_summary(episodes: List[DrawdownEpisode]) -> pd.DataFrame:
    """回撤持续期分布：对 depth / down_periods / recovery_periods / total_periods 做分位摘要。

    返回 index=统计量（count/min/p25/median/p75/max/mean），columns=各度量。
    未收复区间的 recovery_periods 为 NaN，不参与该项统计。
    """
    cols = {
        "depth": [e.depth for e in episodes],
        "down_periods": [e.down_periods for e in episodes],
        "recovery_periods": [e.recovery_periods for e in episodes
                             if math.isfinite(e.recovery_periods)],
        "total_periods": [e.total_periods for e in episodes],
    }
    summary = {k: describe_seq(v) for k, v in cols.items()}
    return pd.DataFrame(summary)


@dataclass(frozen=True, eq=False)
class DrawdownReport:
    """回撤体检报告（一次性给出常用指标 + 区间明细）。"""

    max_drawdown: float
    current_drawdown: float
    longest_periods: int
    time_underwater: float
    calmar: float
    n_episodes: int
    episodes: List[DrawdownEpisode]

    def frame(self) -> pd.DataFrame:
        """区间明细表。"""
        return episodes_frame(self.episodes)

    def summary(self) -> Dict[str, float]:
        """标量指标字典。"""
        return {
            "max_drawdown": self.max_drawdown,
            "current_drawdown": self.current_drawdown,
            "longest_drawdown_periods": float(self.longest_periods),
            "time_underwater_ratio": self.time_underwater,
            "calmar_ratio": self.calmar,
            "n_episodes": float(self.n_episodes),
        }


def calmar_ratio(returns: ReturnsLike, periods_per_year: float = 252.0,
                 start: float = 1.0) -> float:
    """Calmar 比率 = 几何年化收益 / 最大回撤（正数口径）。

    最大回撤为 0 时：收益为正则返回 ``inf``，否则返回 0。
    """
    ppy = check_positive_periods(periods_per_year)
    r = _series(returns).to_numpy(dtype="float64")
    if r.size == 0:
        raise ValueError("收益序列为空")
    if np.any(r <= -1.0):
        raise ValueError("存在 ≤ −100% 的收益，Calmar 无定义")
    ann = math.expm1(float(np.mean(np.log1p(r))) * ppy)
    mdd = max_drawdown(r, start)
    if mdd <= 1e-15:
        return float("inf") if ann > 0 else 0.0
    return ann / mdd


def drawdown_report(returns: ReturnsLike, periods_per_year: float = 252.0,
                    min_depth: float = 0.0, start: float = 1.0) -> DrawdownReport:
    """一次性计算最大回撤、当前回撤、最长水下期、水下时间占比、Calmar 与区间明细。"""
    eps = drawdown_episodes(returns, min_depth, start)
    return DrawdownReport(
        max_drawdown=max_drawdown(returns, start),
        current_drawdown=current_drawdown(returns, start),
        longest_periods=longest_drawdown(returns, start),
        time_underwater=time_underwater_ratio(returns, start),
        calmar=calmar_ratio(returns, periods_per_year, start),
        n_episodes=len(eps),
        episodes=eps,
    )


__all__ = [
    "wealth_curve", "running_peak", "drawdown_series", "underwater", "max_drawdown",
    "current_drawdown", "longest_drawdown", "time_underwater_ratio", "DrawdownEpisode",
    "drawdown_episodes", "top_drawdowns", "episodes_frame", "duration_summary",
    "calmar_ratio", "DrawdownReport", "drawdown_report",
]
