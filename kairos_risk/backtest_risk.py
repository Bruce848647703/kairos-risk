"""回测风险与防过拟合指标模块：概率夏普比率 PSR、缩水夏普比率 DSR、最小跟踪长度、
损失概率与胜率置信区间。

为什么需要「防过拟合」指标
--------------------------
传统夏普比率是一个**点估计**：它不告诉你「这个夏普有多可能是运气」。
在同一份数据上反复挑选参数（N 次试验）后，最优夏普的期望值本身就会随 N 增长，
这就是「回测过拟合」。本模块给出两套互补的显著性口径：

- **PSR（Probabilistic Sharpe Ratio）**：把夏普比率视为随机变量，用其
  一阶渐近分布（含偏度与峰度修正）计算「真实夏普高于基准 SR* 的概率」；
- **DSR（Deflated Sharpe Ratio）**：把基准 SR* 换成「N 次独立试验下的最大夏普
  期望值」，从而对多重检验做缩水（Bailey & López de Prado 的思路）。
  试验次数越多，门槛越高，DSR 越低——这正是它防过拟合的地方。

所有正态 CDF/PPF 都走 :mod:`kairos_risk._util`：scipy 可用时用 scipy，
否则回退到 ``math.erf`` 二分与自研不完全贝塔函数，两条路径结果一致。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._util import (
    EULER_MASCHERONI,
    ReturnsLike,
    betaincinv,
    check_confidence,
    clean_1d,
    norm_cdf,
    norm_ppf,
)


# --------------------------------------------------------------------------- #
# 夏普比率与其渐近方差
# --------------------------------------------------------------------------- #
def per_period_sharpe(returns: ReturnsLike, risk_free: float = 0.0) -> float:
    """**每期**夏普比率 SR̂ = (mean(r) − rf) / std(r)（ddof=1）。

    PSR / DSR 公式中的 SR 必须与观测频率一致（非年化），否则方差修正项会失真。
    """
    v = clean_1d(returns, "returns")
    if v.size < 2:
        raise ValueError("至少需要 2 个观测")
    sd = float(v.std(ddof=1))
    if sd <= 0:
        raise ValueError("收益标准差为零，夏普比率无定义")
    return (float(v.mean()) - float(risk_free)) / sd


def _moments_of(returns: ReturnsLike) -> Tuple[float, float]:
    """返回 (偏度 γ₃, **非超额**峰度 γ₄)，口径与 PSR/DSR 公式一致（正态 γ₄=3）。"""
    v = clean_1d(returns, "returns")
    n = float(v.size)
    sd = float(v.std(ddof=1))
    if n < 4 or sd <= 0:
        return 0.0, 3.0
    z = (v - v.mean()) / sd
    skew = float(n / ((n - 1.0) * (n - 2.0)) * np.sum(z ** 3))
    sum4 = float(np.sum(z ** 4))
    kurt = (n * (n + 1.0) * sum4 / ((n - 1.0) * (n - 2.0) * (n - 3.0))
            - 3.0 * (n - 1.0) ** 2 / ((n - 2.0) * (n - 3.0)))
    return skew, kurt + 3.0


def sharpe_std_error(n_obs: int, sharpe: float, skew: float = 0.0,
                     kurtosis: float = 3.0) -> float:
    """夏普比率的渐近标准误（Lo 2002 / Bailey-LdP 口径）：

        SE(SR̂) = √[ (1 − γ₃·SR̂ + (γ₄−1)/4·SR̂²) / (T − 1) ]

    偏度为正会降低 SE，肥尾（γ₄>3）会提高 SE，因此「同样的夏普」在肥尾策略上
    显著性更低。
    """
    t = float(n_obs)
    if t <= 1:
        raise ValueError(f"观测数必须 > 1，收到 {int(t)}")
    sr = float(sharpe)
    denom = 1.0 - float(skew) * sr + (float(kurtosis) - 1.0) / 4.0 * sr * sr
    if denom <= 0:
        raise ValueError(f"夏普方差修正项非正（{denom:.6g}），SE 无定义")
    return math.sqrt(denom / (t - 1.0))


# --------------------------------------------------------------------------- #
# PSR
# --------------------------------------------------------------------------- #
def probabilistic_sharpe_ratio(sharpe: float, n_obs: int, skew: float = 0.0,
                               kurtosis: float = 3.0,
                               benchmark_sharpe: float = 0.0) -> float:
    """概率夏普比率 PSR：真实夏普高于基准 SR* 的概率（0~1）。

        PSR = Φ[ (SR̂ − SR*)·√(T−1) / √(1 − γ₃·SR̂ + (γ₄−1)/4·SR̂²) ]

    参数
    ----
    sharpe:            每期（非年化）夏普比率 SR̂。
    n_obs:             观测期数 T。
    skew / kurtosis:   收益的偏度 γ₃ 与**非超额**峰度 γ₄（正态为 3）。
    benchmark_sharpe:  基准夏普 SR*（默认 0，即「跑赢零夏普」的概率）。

    T 越大、SR̂ 越高于 SR*、峰度越接近 3，PSR 越接近 1。
    """
    sr = float(sharpe)
    se = sharpe_std_error(n_obs, sr, skew, kurtosis)
    z = (sr - float(benchmark_sharpe)) / se
    return float(norm_cdf(z))


def psr_from_returns(returns: ReturnsLike, benchmark_sharpe: float = 0.0,
                     risk_free: float = 0.0) -> float:
    """直接从收益序列计算 PSR（自动估计每期夏普、偏度与峰度）。"""
    sr = per_period_sharpe(returns, risk_free)
    skew, kurt = _moments_of(returns)
    n = clean_1d(returns, "returns").size
    return probabilistic_sharpe_ratio(sr, n, skew=skew, kurtosis=kurt,
                                      benchmark_sharpe=benchmark_sharpe)


def minimum_track_length(sharpe: float, confidence: float = 0.95, skew: float = 0.0,
                         kurtosis: float = 3.0, benchmark_sharpe: float = 0.0) -> float:
    """最小跟踪长度 T_min：使 PSR ≥ confidence 所需的最少观测期数。

    由 PSR 公式反解：

        T_min = 1 + [ Φ⁻¹(confidence)·√(1 − γ₃·SR̂ + (γ₄−1)/4·SR̂²) / (SR̂ − SR*) ]²

    要求 SR̂ > SR*，否则永远达不到该置信度（抛 ValueError）。
    """
    c = check_confidence(confidence)
    sr = float(sharpe)
    gap = sr - float(benchmark_sharpe)
    if gap <= 0:
        raise ValueError(f"夏普 {sr} 不高于基准 {benchmark_sharpe}，最小跟踪长度无定义")
    denom = 1.0 - float(skew) * sr + (float(kurtosis) - 1.0) / 4.0 * sr * sr
    if denom <= 0:
        raise ValueError(f"夏普方差修正项非正（{denom:.6g}）")
    z = norm_ppf(c)
    return float(1.0 + (z * math.sqrt(denom) / gap) ** 2)


# --------------------------------------------------------------------------- #
# DSR：对多重检验做缩水
# --------------------------------------------------------------------------- #
def expected_max_sharpe(n_trials: int, sharpe_variance: float = 1.0) -> float:
    """N 次独立试验下最大夏普比率的期望值（DSR 的基准 SR₀）。

        E[max SR] ≈ √V · [ (1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ]

    γ 为欧拉-马歇罗尼常数，V 为各次试验夏普比率的方差（可用试验样本估计，
    或直接用其渐近值 1）。N ≤ 1 时不存在多重检验，返回 0。
    """
    n = int(n_trials)
    if n <= 0:
        raise ValueError("n_trials 必须为正")
    v = float(sharpe_variance)
    if v < 0:
        raise ValueError("sharpe_variance 不能为负")
    if n == 1:
        return 0.0
    gamma = EULER_MASCHERONI
    z1 = norm_ppf(1.0 - 1.0 / n)
    z2 = norm_ppf(1.0 - 1.0 / (n * math.e))
    return float(math.sqrt(v) * ((1.0 - gamma) * z1 + gamma * z2))


@dataclass(frozen=True, eq=False)
class DeflatedSharpe:
    """缩水夏普比率结果。

    字段
    ----
    dsr:              缩水夏普比率（真实夏普高于「试验最大值期望」的概率，0~1）；
    psr:              不做多重检验修正的概率夏普比率（SR* = benchmark_sharpe）；
    benchmark_sharpe: 用户给定的基准夏普 SR*；
    sr0:              DSR 使用的门槛 = max(E[max SR], SR*)；
    sharpe:           输入的每期夏普 SR̂；
    n_trials:         试验次数 N；
    n_obs:            观测期数 T；
    skew / kurtosis:  参与方差修正的偏度与非超额峰度；
    sharpe_variance:  试验间夏普方差 V；
    haircut:          相对 PSR 的缩水幅度 = psr − dsr（≥0，试验越多越大）。
    """

    dsr: float
    psr: float
    benchmark_sharpe: float
    sr0: float
    sharpe: float
    n_trials: int
    n_obs: int
    skew: float
    kurtosis: float
    sharpe_variance: float

    @property
    def haircut(self) -> float:
        """PSR 与 DSR 之差，量化「多重检验」吃掉了多少显著性。"""
        return self.psr - self.dsr


def deflated_sharpe_ratio(sharpe: float, n_obs: int, n_trials: int = 1,
                          skew: float = 0.0, kurtosis: float = 3.0,
                          sharpe_variance: Optional[float] = None,
                          trial_sharpes: Optional[Sequence[float]] = None,
                          benchmark_sharpe: float = 0.0) -> DeflatedSharpe:
    """缩水夏普比率 DSR（Bailey & López de Prado 思路的自研实现）。

    参数
    ----
    sharpe / n_obs:      每期夏普 SR̂ 与观测期数 T。
    n_trials:            试验次数 N（策略搜索/调参的尝试数）。
    skew / kurtosis:     收益偏度 γ₃ 与非超额峰度 γ₄（正态=3）。
    sharpe_variance:     试验间夏普方差 V；None 时优先用 ``trial_sharpes`` 的样本方差，
                         否则取渐近值 ``1/(n_obs−1)``（零假设下单次夏普估计的方差）。
    trial_sharpes:       各次试验的夏普序列（用于同时确定 N 与 V）。
    benchmark_sharpe:    外部基准 SR*；DSR 门槛取 max(E[max SR], SR*)。

    返回 :class:`DeflatedSharpe`。DSR 随 N 增大而单调不增——这是它作为
    「防过拟合」指标的核心性质。
    """
    sr = float(sharpe)
    t = int(n_obs)
    if t <= 1:
        raise ValueError(f"观测数必须 > 1，收到 {t}")
    trials: Optional[np.ndarray] = None
    if trial_sharpes is not None:
        trials = np.asarray(list(trial_sharpes), dtype="float64")
        trials = trials[np.isfinite(trials)]
        if trials.size == 0:
            raise ValueError("trial_sharpes 为空")
        n = int(max(n_trials, trials.size))
    else:
        n = int(n_trials)
    if n <= 0:
        raise ValueError("n_trials 必须为正")

    if sharpe_variance is not None:
        v = float(sharpe_variance)
    elif trials is not None and trials.size >= 2:
        v = float(trials.var(ddof=1))
    else:
        v = 1.0 / float(t - 1)
    if v < 0:
        raise ValueError("sharpe_variance 不能为负")

    sr0_trials = expected_max_sharpe(n, v)
    sr0 = max(sr0_trials, float(benchmark_sharpe))
    dsr = probabilistic_sharpe_ratio(sr, t, skew=skew, kurtosis=kurtosis, benchmark_sharpe=sr0)
    psr = probabilistic_sharpe_ratio(sr, t, skew=skew, kurtosis=kurtosis,
                                     benchmark_sharpe=float(benchmark_sharpe))
    return DeflatedSharpe(dsr=dsr, psr=psr, benchmark_sharpe=float(benchmark_sharpe),
                          sr0=sr0, sharpe=sr, n_trials=n, n_obs=t, skew=float(skew),
                          kurtosis=float(kurtosis), sharpe_variance=v)


def dsr_from_returns(returns: ReturnsLike, n_trials: int = 1,
                     trial_sharpes: Optional[Sequence[float]] = None,
                     risk_free: float = 0.0, benchmark_sharpe: float = 0.0) -> DeflatedSharpe:
    """直接从收益序列计算 DSR（自动估计每期夏普、偏度与峰度）。"""
    sr = per_period_sharpe(returns, risk_free)
    skew, kurt = _moments_of(returns)
    n = clean_1d(returns, "returns").size
    return deflated_sharpe_ratio(sr, n, n_trials=n_trials, skew=skew, kurtosis=kurt,
                                 trial_sharpes=trial_sharpes,
                                 benchmark_sharpe=benchmark_sharpe)


def trials_for_target_dsr(sharpe: float, n_obs: int, target_dsr: float = 0.95,
                          skew: float = 0.0, kurtosis: float = 3.0,
                          sharpe_variance: Optional[float] = None,
                          max_trials: int = 100000) -> int:
    """在保持 DSR ≥ target_dsr 的前提下，最多能承受多少次试验（策略搜索预算）。

    从 N=1 开始逐步增大，返回最后一个仍满足目标的 N；N=1 就不满足时返回 0。
    """
    tgt = check_confidence(target_dsr)
    best = 0
    n = 1
    while n <= int(max_trials):
        res = deflated_sharpe_ratio(sharpe, n_obs, n_trials=n, skew=skew, kurtosis=kurtosis,
                                    sharpe_variance=sharpe_variance)
        if res.dsr < tgt:
            break
        best = n
        if n < 10:
            n += 1
        else:
            n = int(n * 1.25) + 1            # 粗粒度指数扩张，避免大 N 下逐点扫描过慢
    return best


# --------------------------------------------------------------------------- #
# 损失概率与「最小损失概率」
# --------------------------------------------------------------------------- #
def loss_probability(returns: ReturnsLike, loss_level: float = 0.0,
                     method: str = "empirical") -> float:
    """单期损失不小于 ``loss_level`` 的概率 P(−r ≥ loss_level)。

    - ``"empirical"``：直接用样本频率（loss_level=0 时即「亏损期占比」）；
    - ``"normal"``：正态近似 P(r ≤ −loss_level) = Φ((−loss_level − μ)/σ)，
      可对样本外水平外推。
    """
    if method not in ("empirical", "normal"):
        raise ValueError("method 只支持 'empirical' 或 'normal'")
    v = clean_1d(returns, "returns")
    lvl = float(loss_level)
    if method == "empirical":
        return float(np.mean(-v >= lvl))
    mu, sd = float(v.mean()), float(v.std(ddof=1))
    if sd <= 0:
        raise ValueError("收益标准差为零，正态近似无定义")
    return float(norm_cdf((-lvl - mu) / sd))


def probability_of_min_loss(returns: ReturnsLike, horizon: int = 1,
                            method: str = "empirical") -> float:
    """最小损失概率：未来 ``horizon`` 期内**至少出现一次**「不优于历史最差单期」的概率。

    把「历史最小收益 r_min 对应的损失水平」当作需要复现的极端事件，
    其单期发生概率 p₁ 由 :func:`loss_probability` 给出（经验口径为 1/n，
    正态口径为对 r_min 的左尾外推），再按独立复合近似：

        P = 1 − (1 − p₁)^horizon

    该概率对 horizon 单调不减、值域 [0,1]，用于回答「再持有 h 期，
    会不会重演历史最惨的那一期」。
    """
    v = clean_1d(returns, "returns")
    h = int(horizon)
    if h <= 0:
        raise ValueError("horizon 必须为正")
    worst_loss = float(-v.min())
    p1 = loss_probability(v, worst_loss, method=method)
    p1 = min(max(p1, 0.0), 1.0)
    return float(1.0 - (1.0 - p1) ** h)


# --------------------------------------------------------------------------- #
# 胜率的置信区间
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class WinRateCI:
    """胜率点估计与置信区间。

    字段
    ----
    wins / n:    盈利期数与总期数；
    point:       胜率点估计 p̂ = wins/n；
    lower/upper: 置信区间上下界（含端点，落在 [0,1]）；
    confidence:  置信水平；
    method:      ``"wilson"`` / ``"normal"`` / ``"clopper_pearson"``。
    """

    wins: int
    n: int
    point: float
    lower: float
    upper: float
    confidence: float
    method: str

    def contains(self, p: float) -> bool:
        """判断某个胜率取值是否落在区间内。"""
        return self.lower <= float(p) <= self.upper

    @property
    def width(self) -> float:
        """区间宽度。"""
        return self.upper - self.lower


def win_rate(returns: ReturnsLike, threshold: float = 0.0) -> float:
    """胜率：收益严格大于 ``threshold`` 的期数占比。"""
    v = clean_1d(returns, "returns")
    return float(np.mean(v > float(threshold)))


def win_rate_ci(returns: ReturnsLike = None, wins: Optional[int] = None,
                n: Optional[int] = None, threshold: float = 0.0,
                confidence: float = 0.95, method: str = "wilson") -> WinRateCI:
    """胜率的置信区间，三种口径可选。

    - ``"wilson"``：Wilson score 区间，小样本与 p̂ 接近 0/1 时仍表现良好（默认）；
    - ``"normal"``：正态（Wald）近似 p̂ ± z·√(p̂(1−p̂)/n)，最简单但边界处会失真；
    - ``"clopper_pearson"``：基于不完全贝塔函数的**精确**区间（保守，覆盖率 ≥ 1−α），
      scipy 可用时用 ``special.betaincinv``，否则用自研二分反演。

    可直接给 ``returns``（自动按 ``threshold`` 统计盈利期数），或直接给 (wins, n)。
    """
    if method not in ("wilson", "normal", "clopper_pearson"):
        raise ValueError("method 只支持 'wilson' / 'normal' / 'clopper_pearson'")
    c = check_confidence(confidence)
    if returns is not None:
        v = clean_1d(returns, "returns")
        k = int(np.sum(v > float(threshold)))
        total = int(v.size)
    else:
        if wins is None or n is None:
            raise ValueError("必须提供 returns 或完整的 (wins, n)")
        k, total = int(wins), int(n)
    if total <= 0:
        raise ValueError("n 必须为正")
    if not (0 <= k <= total):
        raise ValueError(f"wins={k} 必须落在 [0, n={total}] 内")

    p = k / total
    alpha = 1.0 - c
    if method == "normal":
        z = norm_ppf(1.0 - alpha / 2.0)
        half = z * math.sqrt(max(p * (1.0 - p), 0.0) / total)
        lo, hi = p - half, p + half
    elif method == "wilson":
        z = norm_ppf(1.0 - alpha / 2.0)
        z2 = z * z
        denom = 1.0 + z2 / total
        center = (p + z2 / (2.0 * total)) / denom
        half = z * math.sqrt(p * (1.0 - p) / total + z2 / (4.0 * total * total)) / denom
        lo, hi = center - half, center + half
    else:
        lo = 0.0 if k == 0 else betaincinv(float(k), float(total - k + 1), alpha / 2.0)
        hi = 1.0 if k == total else betaincinv(float(k + 1), float(total - k), 1.0 - alpha / 2.0)
    return WinRateCI(wins=k, n=total, point=p, lower=float(min(max(lo, 0.0), 1.0)),
                     upper=float(min(max(hi, 0.0), 1.0)), confidence=c, method=method)


def backtest_risk_report(returns: ReturnsLike, n_trials: int = 1,
                         confidence: float = 0.95, risk_free: float = 0.0,
                         ci_method: str = "wilson",
                         horizon: int = 21) -> pd.DataFrame:
    """回测体检一页纸：把本模块的指标汇总成表（index=指标，columns=value/detail）。"""
    v = clean_1d(returns, "returns")
    sr = per_period_sharpe(v, risk_free)
    skew, kurt = _moments_of(v)
    dsr = deflated_sharpe_ratio(sr, v.size, n_trials=n_trials, skew=skew, kurtosis=kurt)
    ci = win_rate_ci(v, confidence=confidence, method=ci_method)
    rows = {
        "n_obs": (float(v.size), ""),
        "sharpe_per_period": (sr, ""),
        "skewness": (skew, ""),
        "kurtosis": (kurt, "非超额峰度，正态=3"),
        "psr": (dsr.psr, f"基准夏普={dsr.benchmark_sharpe:g}"),
        "dsr": (dsr.dsr, f"试验次数={dsr.n_trials}, 门槛 SR0={dsr.sr0:.6g}"),
        "dsr_haircut": (dsr.haircut, "PSR − DSR"),
        "min_track_length": (minimum_track_length(sr, confidence, skew, kurt)
                             if sr > 0 else float("inf"), f"置信度={confidence:g}"),
        "win_rate": (ci.point, ""),
        "win_rate_lower": (ci.lower, ci.method),
        "win_rate_upper": (ci.upper, ci.method),
        "loss_probability": (loss_probability(v, 0.0), "单期亏损概率（经验）"),
        "probability_of_min_loss": (probability_of_min_loss(v, horizon=horizon),
                                    f"horizon={horizon}"),
    }
    out = pd.DataFrame({"value": {k: val[0] for k, val in rows.items()},
                        "detail": {k: val[1] for k, val in rows.items()}})
    return out


__all__ = [
    "per_period_sharpe", "sharpe_std_error", "probabilistic_sharpe_ratio",
    "psr_from_returns", "minimum_track_length", "expected_max_sharpe",
    "DeflatedSharpe", "deflated_sharpe_ratio", "dsr_from_returns",
    "trials_for_target_dsr", "loss_probability", "probability_of_min_loss",
    "WinRateCI", "win_rate", "win_rate_ci", "backtest_risk_report",
]
