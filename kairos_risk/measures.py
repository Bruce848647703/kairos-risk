"""基础风险度量模块：波动、VaR（历史/参数/修正）、ES-CVaR、下行风险、跟踪误差、
信息比率、Beta/Jensen-Alpha/R²、高阶矩与相关矩阵。

约定
----
- 输入为「每期收益」：``pd.Series``（单资产）、``np.ndarray`` 或
  ``pd.DataFrame``（多资产面板，index=期次，columns=资产）；NaN 自动剔除。
- **同型进出**：Series/ndarray 进 -> float 出；DataFrame 进 -> 按资产索引的 Series 出。
- **符号约定**：VaR / ES 一律返回**正数损失值**，0.02 表示该分位下损失 2%。
- **年化口径**：``periods_per_year`` 默认 1（每期）；波动按 √T、收益按 T 缩放。
  ``mar``（最低可接受收益）为**每期**口径，``risk_free`` 为**年化**口径。
- **scipy 可选**：正态与学生 t 的分位数优先用 scipy，缺失时回退到
  ``math.erf`` 二分与自研不完全贝塔函数（见 :mod:`kairos_risk._util`）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from ._util import (
    ReturnsLike,
    align_pair,
    apply_per_asset,
    as_returns_frame,
    check_confidence,
    check_positive_periods,
    clean_1d,
    norm_pdf,
    norm_ppf,
    quantile_sorted,
    student_t_pdf,
    tail_mask,
    t_ppf,
)


# --------------------------------------------------------------------------- #
# 内部：矩估计原子函数（对已清洗的一维数组）
# --------------------------------------------------------------------------- #
def _std(values: np.ndarray, ddof: int = 1) -> float:
    """样本标准差；样本量不足或 ddof 非法时抛错。"""
    if values.size <= ddof:
        raise ValueError(f"样本量 {values.size} 不足以支撑 ddof={ddof}")
    return float(values.std(ddof=ddof))


def _skew_of(v: np.ndarray) -> float:
    """调整后的 Fisher-Pearson 偏度（口径同 ``pandas.Series.skew``），n<3 记 0。"""
    n = v.size
    if n < 3:
        return 0.0
    s = float(v.std(ddof=1))
    if s <= 0:
        return 0.0
    m3 = float(np.mean((v - v.mean()) ** 3))
    return n * n / ((n - 1.0) * (n - 2.0)) * m3 / s ** 3


def _kurt_of(v: np.ndarray) -> float:
    """超额峰度（口径同 ``pandas.Series.kurt``，正态为 0），n<4 记 0。

    G2 = n(n+1)·Σ((x−x̄)/s)⁴ / ((n−1)(n−2)(n−3)) − 3(n−1)²/((n−2)(n−3))，
    即对四阶标准化矩做小样本偏差修正后再减 3。
    """
    n = float(v.size)
    if v.size < 4:
        return 0.0
    s = float(v.std(ddof=1))
    if s <= 0:
        return 0.0
    sum4 = float(np.sum((v - v.mean()) ** 4)) / s ** 4
    term1 = n * (n + 1.0) * sum4 / ((n - 1.0) * (n - 2.0) * (n - 3.0))
    return term1 - 3.0 * (n - 1.0) ** 2 / ((n - 2.0) * (n - 3.0))


def _implied_t_df(excess_kurt: float) -> float:
    """由超额峰度反推学生 t 自由度：K = 6/(ν−4) -> ν = 4 + 6/K。

    K ≤ 0（尾部不比正态厚）时返回一个很大的 ν，使学生 t 退化为近似正态。
    """
    k = float(excess_kurt)
    if k <= 1e-8:
        return 1.0e6
    return float(min(max(4.0 + 6.0 / k, 2.05), 1.0e6))


# --------------------------------------------------------------------------- #
# 矩与波动
# --------------------------------------------------------------------------- #
def volatility(returns: ReturnsLike, periods_per_year: float = 1.0,
               ddof: int = 1) -> Union[float, pd.Series]:
    """收益波动率（标准差）；``periods_per_year>1`` 时按 √T 年化。"""
    ppy = check_positive_periods(periods_per_year)
    scale = math.sqrt(ppy)

    def _one(x: ReturnsLike) -> float:
        return _std(clean_1d(x, "returns"), ddof) * scale

    return apply_per_asset(returns, _one, name="volatility")


def skewness(returns: ReturnsLike) -> Union[float, pd.Series]:
    """样本偏度：>0 右尾更长，<0 左尾更长（金融收益通常为负偏）。"""
    return apply_per_asset(returns, lambda x: _skew_of(clean_1d(x, "returns")), name="skewness")


def kurtosis(returns: ReturnsLike) -> Union[float, pd.Series]:
    """样本**超额**峰度：正态为 0，>0 表示肥尾（极端收益比正态假设更频繁）。"""
    return apply_per_asset(returns, lambda x: _kurt_of(clean_1d(x, "returns")), name="kurtosis")


def correlation_matrix(returns: ReturnsLike) -> pd.DataFrame:
    """资产间相关系数矩阵（成对剔除 NaN，口径同 ``DataFrame.corr``）。

    常数收益资产的相关系数无定义（NaN），此处记 0 并把对角补回 1，
    便于直接用于风险分解等下游计算。
    """
    frame = as_returns_frame(returns, dropna=False)
    if frame.shape[1] < 2:
        raise ValueError("相关矩阵至少需要 2 个资产")
    corr = frame.corr().to_numpy(dtype="float64")
    if not np.all(np.isfinite(corr)):
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 1.0)
    return pd.DataFrame((corr + corr.T) / 2.0, index=frame.columns, columns=frame.columns)


def covariance_matrix(returns: ReturnsLike, ddof: int = 1) -> pd.DataFrame:
    """样本协方差矩阵：Σ = XᵀX/(n−ddof)，X 为逐列去均值后的收益面板（整行剔除 NaN）。"""
    frame = as_returns_frame(returns)
    n = len(frame)
    if n - ddof <= 0:
        raise ValueError(f"样本量 n={n} 不足以支撑 ddof={ddof}")
    x = frame.to_numpy(dtype="float64")
    x = x - x.mean(axis=0)
    cov = x.T @ x / float(n - ddof)
    return pd.DataFrame((cov + cov.T) / 2.0, index=frame.columns, columns=frame.columns)


def annualized_return(returns: ReturnsLike, periods_per_year: float = 252.0) -> Union[float, pd.Series]:
    """几何年化收益：(∏(1+r))^(ppy/n) − 1，比算术平均更贴近真实复利结果。"""
    ppy = check_positive_periods(periods_per_year)

    def _one(x: ReturnsLike) -> float:
        v = clean_1d(x, "returns")
        if np.any(v <= -1.0):
            raise ValueError("存在 ≤ −100% 的收益，几何年化收益无定义")
        return math.expm1(float(np.mean(np.log1p(v))) * ppy)

    return apply_per_asset(returns, _one, name="annualized_return")


# --------------------------------------------------------------------------- #
# VaR：历史 / 参数 / 修正（Cornish-Fisher）
# --------------------------------------------------------------------------- #
def historical_var(returns: ReturnsLike, confidence: float = 0.95) -> Union[float, pd.Series]:
    """历史模拟 VaR：损失 = −q_{1−c}(r)，返回正数损失值。

    分位数用线性插值（numpy 默认），因此对 ``confidence`` 单调不减；
    不做任何分布假设，能自然体现样本中的肥尾与偏度，但也无法外推到
    「样本中从未发生过的更极端情形」（那正是 :mod:`kairos_risk.tail` 的 EVT 用途）。
    """
    c = check_confidence(confidence)

    def _one(x: ReturnsLike) -> float:
        return -quantile_sorted(clean_1d(x, "returns"), 1.0 - c)

    return apply_per_asset(returns, _one, name="historical_var")


def cornish_fisher_z(confidence: float, skew: float, excess_kurt: float) -> float:
    """Cornish-Fisher 修正分位数 z_cf。

    在标准正态分位 z 上叠加偏度/峰度修正项：

        z_cf = z + (z²−1)·S/6 + (z³−3z)·K/24 − (2z³−5z)·S²/36

    S 为偏度、K 为**超额**峰度。左尾（高置信度）时负偏度会把 z_cf 推得更负，
    从而给出比正态假设更大的 VaR，符合「肥尾 + 左偏」资产的经验特征。
    """
    c = check_confidence(confidence)
    z = norm_ppf(1.0 - c)
    s, k = float(skew), float(excess_kurt)
    return float(z
                 + (z * z - 1.0) * s / 6.0
                 + (z ** 3 - 3.0 * z) * k / 24.0
                 - (2.0 * z ** 3 - 5.0 * z) * s * s / 36.0)


def parametric_var(returns: ReturnsLike = None, confidence: float = 0.95,
                   distribution: str = "normal", mean: Optional[float] = None,
                   std: Optional[float] = None, df: Optional[float] = None,
                   periods_per_year: float = 1.0) -> Union[float, pd.Series]:
    """参数法 VaR，返回正数损失值。

    参数
    ----
    returns:      收益序列/面板；也可直接给 (mean, std) 跳过矩估计。
    distribution: ``"normal"`` -> VaR = −(μ + z_{1−c}·σ)；
                  ``"student_t"`` -> 用学生 t 分位替代 z，尺度按 σ_t = σ·√((ν−2)/ν)
                  校准，使模型方差等于样本方差；尾部更厚，在高置信度（≥99%）下
                  VaR 显著大于正态法；95% 档因方差校准压缩了中段尺度，
                  小自由度时可能略低于正态法，属该口径的正常现象。
    df:           学生 t 自由度 ν；None 时由样本超额峰度反推 ν = 4 + 6/K。
    periods_per_year: >1 时 μ 线性、σ 按 √T 缩放（多期参数 VaR）。
    """
    c = check_confidence(confidence)
    ppy = check_positive_periods(periods_per_year)
    if distribution not in ("normal", "student_t"):
        raise ValueError("distribution 只支持 'normal' 或 'student_t'")
    z = norm_ppf(1.0 - c)

    def _one(x: Optional[ReturnsLike]) -> float:
        m, s, dof = mean, std, df
        if x is not None:
            v = clean_1d(x, "returns")
            if v.size < 2:
                raise ValueError("至少需要 2 个观测")
            m = float(v.mean())
            s = _std(v, 1)
            if dof is None:
                dof = _implied_t_df(_kurt_of(v))
        if m is None or s is None:
            raise ValueError("必须提供 returns 或完整的 (mean, std)")
        s = float(s)
        if s < 0:
            raise ValueError("std 不能为负")
        mu_t, sig_t = float(m) * ppy, s * math.sqrt(ppy)
        if distribution == "normal":
            return -(mu_t + z * sig_t)
        nu = float(dof)
        if nu <= 2:
            raise ValueError("学生 t 参数 VaR 要求自由度 > 2")
        scale = sig_t * math.sqrt((nu - 2.0) / nu)
        return -(mu_t + t_ppf(1.0 - c, nu) * scale)

    if returns is None:
        return _one(None)
    return apply_per_asset(returns, _one, name="parametric_var")


def modified_var(returns: ReturnsLike, confidence: float = 0.95) -> Union[float, pd.Series]:
    """修正 VaR（Cornish-Fisher）：VaR_mod = −(μ + z_cf·σ)。

    相比正态法能体现左偏与肥尾的尾部放大，又不像历史法那样受限于样本中
    是否恰好出现过极端观测，是两者之间的常用折中。
    """
    c = check_confidence(confidence)

    def _one(x: ReturnsLike) -> float:
        v = clean_1d(x, "returns")
        if v.size < 4:
            raise ValueError("修正 VaR 至少需要 4 个观测（要估偏度与峰度）")
        z_cf = cornish_fisher_z(c, _skew_of(v), _kurt_of(v))
        return -(float(v.mean()) + z_cf * _std(v, 1))

    return apply_per_asset(returns, _one, name="modified_var")


# --------------------------------------------------------------------------- #
# CVaR / ES
# --------------------------------------------------------------------------- #
def expected_shortfall(returns: ReturnsLike, confidence: float = 0.95,
                       method: str = "historical",
                       df: Optional[float] = None) -> Union[float, pd.Series]:
    """期望损失 ES / CVaR：尾部（损失不优于 VaR 分位）的平均损失，正数。

    method
    ------
    - ``"historical"``：ES = −mean(r | r ≤ q_{1−c})，纯经验、无分布假设，
      恒 ≥ 同置信度历史 VaR，且对置信度单调不减；
    - ``"normal"``：正态闭式 ES = −μ + σ·φ(z_α)/α，其中 α = 1−c；
    - ``"student_t"``：学生 t 闭式 ES，由 ∫_x^∞ t·f(t)dt = (ν+x²)·f(x)/(ν−1) 推得，
      尾部比正态更厚；``df`` 为 None 时由样本超额峰度反推自由度。
    """
    c = check_confidence(confidence)
    if method not in ("historical", "normal", "student_t"):
        raise ValueError("method 只支持 'historical' / 'normal' / 'student_t'")
    alpha = 1.0 - c
    z = norm_ppf(alpha)
    phi_z = float(norm_pdf(z))

    def _one(x: ReturnsLike) -> float:
        v = clean_1d(x, "returns")
        if method == "historical":
            return float(-v[tail_mask(v, c)].mean())
        if v.size < 2:
            raise ValueError("至少需要 2 个观测")
        mu, sigma = float(v.mean()), _std(v, 1)
        if method == "normal":
            return -mu + sigma * phi_z / alpha
        nu = float(df) if df is not None else _implied_t_df(_kurt_of(v))
        if nu <= 2:
            raise ValueError("学生 t ES 要求自由度 > 2")
        scale = sigma * math.sqrt((nu - 2.0) / nu)
        q = t_ppf(alpha, nu)
        # 标准化 t 的下尾条件均值 E[T | T ≤ q] = −f(q)(ν+q²)/((ν−1)α)
        tail_mean_t = -float(student_t_pdf(q, nu)) * (nu + q * q) / ((nu - 1.0) * alpha)
        return -(mu + scale * tail_mean_t)

    return apply_per_asset(returns, _one, name="expected_shortfall")


#: 语义别名：CVaR 与 ES 在本库中指同一度量
cvar = expected_shortfall


# --------------------------------------------------------------------------- #
# 下行风险与相对基准指标
# --------------------------------------------------------------------------- #
def downside_deviation(returns: ReturnsLike, mar: float = 0.0,
                       periods_per_year: float = 1.0) -> Union[float, pd.Series]:
    """下行标准差：只统计低于最低可接受收益 MAR 的偏差（高于 MAR 的部分记 0）。

    DD = √( Σ min(r−MAR, 0)² / n )，分母用**全部**观测数而非尾部计数，
    这是 Sortino 体系的标准口径，可避免样本中几乎无下行时度量爆炸。
    """
    ppy = check_positive_periods(periods_per_year)
    target = float(mar)
    scale = math.sqrt(ppy)

    def _one(x: ReturnsLike) -> float:
        v = clean_1d(x, "returns")
        diff = np.minimum(v - target, 0.0)
        return math.sqrt(float(diff @ diff) / v.size) * scale

    return apply_per_asset(returns, _one, name="downside_deviation")


def sortino_ratio(returns: ReturnsLike, mar: float = 0.0,
                  periods_per_year: float = 252.0) -> Union[float, pd.Series]:
    """Sortino 比率 = (mean(r) − MAR)·√ppy / 下行标准差(每期)。

    ``mar`` 为**每期**最低可接受收益（与 :func:`downside_deviation` 一致）。
    无下行样本时：均值为正返回 ``inf``，否则返回 0。
    """
    ppy = check_positive_periods(periods_per_year)
    target = float(mar)

    def _one(x: ReturnsLike) -> float:
        v = clean_1d(x, "returns")
        diff = np.minimum(v - target, 0.0)
        dd = math.sqrt(float(diff @ diff) / v.size)
        excess = float(v.mean()) - target
        if dd <= 0:
            return float("inf") if excess > 0 else 0.0
        return excess * math.sqrt(ppy) / dd

    return apply_per_asset(returns, _one, name="sortino_ratio")


def sharpe_ratio(returns: ReturnsLike, risk_free: float = 0.0,
                 periods_per_year: float = 252.0, ddof: int = 1) -> Union[float, pd.Series]:
    """夏普比率 = (几何年化收益 − 年化无风险利率) / 年化波动。"""
    ppy = check_positive_periods(periods_per_year)
    rf = float(risk_free)

    def _one(x: ReturnsLike) -> float:
        v = clean_1d(x, "returns")
        vol = _std(v, ddof) * math.sqrt(ppy)
        if vol <= 0:
            raise ValueError("波动为零，夏普比率无定义")
        ann = math.expm1(float(np.mean(np.log1p(v))) * ppy) if not np.any(v <= -1.0) \
            else float(v.mean()) * ppy
        return (ann - rf) / vol

    return apply_per_asset(returns, _one, name="sharpe_ratio")


def tracking_error(returns: ReturnsLike, benchmark: ReturnsLike,
                   periods_per_year: float = 1.0, ddof: int = 1) -> Union[float, pd.Series]:
    """跟踪误差：主动收益 (r − r_b) 的标准差，可按 √ppy 年化。"""
    ppy = check_positive_periods(periods_per_year)
    scale = math.sqrt(ppy)

    def _one(x: ReturnsLike) -> float:
        a, b, _ = align_pair(x, benchmark)
        return _std(a - b, ddof) * scale

    return apply_per_asset(returns, _one, name="tracking_error")


def information_ratio(returns: ReturnsLike, benchmark: ReturnsLike,
                      periods_per_year: float = 252.0, ddof: int = 1) -> Union[float, pd.Series]:
    """信息比率 = 年化主动收益 / 年化跟踪误差，衡量相对基准的稳定超额能力。"""
    ppy = check_positive_periods(periods_per_year)

    def _one(x: ReturnsLike) -> float:
        a, b, _ = align_pair(x, benchmark)
        active = a - b
        te = _std(active, ddof)
        if te <= 0:
            raise ValueError("跟踪误差为零，信息比率无定义")
        return float(active.mean()) * ppy / (te * math.sqrt(ppy))

    return apply_per_asset(returns, _one, name="information_ratio")


# --------------------------------------------------------------------------- #
# 对基准回归：Beta / Jensen Alpha / R²
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class RegressionStats:
    """资产对基准的单因子回归结果（CAPM 口径）。

    字段
    ----
    beta:             斜率 β = Cov(r, r_b)/Var(r_b)；
    alpha:            每期 Jensen α = mean(r−rf) − β·mean(r_b−rf)；
    alpha_annualized: α × periods_per_year；
    r2:               拟合优度（单因子下等于相关系数平方），衡量被基准解释的方差占比；
    corr:             与基准的相关系数；
    resid_std:        残差标准差（特质波动，每期，ddof=2）；
    n_obs:            参与回归的有效观测数。
    """

    beta: float
    alpha: float
    alpha_annualized: float
    r2: float
    corr: float
    resid_std: float
    n_obs: int


def regression_stats(returns: ReturnsLike, benchmark: ReturnsLike, risk_free: float = 0.0,
                     periods_per_year: float = 252.0) -> RegressionStats:
    """资产收益对基准收益做一元回归，返回 Beta / Jensen-Alpha / R² 等统计量。

    回归在**超额收益**空间进行：(r − rf) = α + β(r_b − rf) + ε，
    ``risk_free`` 为年化利率，内部按 periods_per_year 折算为每期。
    pandas 输入按索引交集对齐，纯数组输入按位置配对。
    """
    ppy = check_positive_periods(periods_per_year)
    a, b, _ = align_pair(returns, benchmark)
    n = a.size
    if n < 3:
        raise ValueError("回归至少需要 3 个观测")
    rf = float(risk_free) / ppy
    xa, xb = a - rf, b - rf
    ma, mb = float(xa.mean()), float(xb.mean())
    db = xb - mb
    ss_b = float(db @ db)
    if ss_b <= 0:
        raise ValueError("基准收益方差为零，Beta 无定义")
    beta = float((xa - ma) @ db) / ss_b
    alpha = ma - beta * mb
    resid = (xa - ma) - beta * db
    ss_a = float((xa - ma) @ (xa - ma))
    r2 = 0.0 if ss_a <= 0 else min(max(beta * beta * ss_b / ss_a, 0.0), 1.0)
    corr = 0.0 if ss_a <= 0 else beta * math.sqrt(ss_b / ss_a)
    resid_std = math.sqrt(max(float(resid @ resid) / float(n - 2), 0.0))
    return RegressionStats(beta=beta, alpha=alpha, alpha_annualized=alpha * ppy,
                           r2=r2, corr=float(corr), resid_std=resid_std, n_obs=n)


def beta(returns: ReturnsLike, benchmark: ReturnsLike) -> Union[float, pd.Series]:
    """Beta = Cov(r, r_b)/Var(r_b)，即对基准回归的斜率。"""
    return apply_per_asset(returns, lambda x: regression_stats(x, benchmark).beta, name="beta")


def jensen_alpha(returns: ReturnsLike, benchmark: ReturnsLike, risk_free: float = 0.0,
                 periods_per_year: float = 252.0, annualized: bool = True) -> Union[float, pd.Series]:
    """Jensen Alpha：扣除基准暴露后的超额收益（默认年化口径）。"""
    def _one(x: ReturnsLike) -> float:
        st = regression_stats(x, benchmark, risk_free=risk_free, periods_per_year=periods_per_year)
        return st.alpha_annualized if annualized else st.alpha

    return apply_per_asset(returns, _one, name="jensen_alpha")


def r_squared(returns: ReturnsLike, benchmark: ReturnsLike) -> Union[float, pd.Series]:
    """R²：基准可解释的收益方差占比（单因子下等于相关系数平方）。"""
    return apply_per_asset(returns, lambda x: regression_stats(x, benchmark).r2, name="r_squared")


def regression_table(returns: ReturnsLike, benchmark: ReturnsLike, risk_free: float = 0.0,
                     periods_per_year: float = 252.0) -> pd.DataFrame:
    """多资产对同一基准的回归统计汇总表（index=资产，columns=统计量）。"""
    frame = as_returns_frame(returns)
    rows: Dict[str, Dict[str, float]] = {}
    for col in frame.columns:
        st = regression_stats(frame[col], benchmark, risk_free=risk_free,
                              periods_per_year=periods_per_year)
        rows[str(col)] = {"beta": st.beta, "alpha": st.alpha,
                          "alpha_annualized": st.alpha_annualized, "r2": st.r2,
                          "corr": st.corr, "resid_std": st.resid_std,
                          "n_obs": float(st.n_obs)}
    return pd.DataFrame.from_dict(rows, orient="index")


# --------------------------------------------------------------------------- #
# 一键风险摘要
# --------------------------------------------------------------------------- #
def risk_report(returns: ReturnsLike, confidence: float = 0.95,
                periods_per_year: float = 252.0) -> pd.DataFrame:
    """把常用风险指标汇总成一张表（index=指标，columns=资产）。

    覆盖年化收益/波动/夏普、偏度/超额峰度、历史 VaR、参数 VaR（正态与学生 t）、
    修正 VaR、ES（历史/正态/学生 t）、下行标准差与 Sortino，便于快速体检收益面板。
    """
    c = check_confidence(confidence)
    frame = as_returns_frame(returns)
    metrics = {
        "annualized_return": annualized_return(frame, periods_per_year),
        "volatility": volatility(frame, periods_per_year),
        "sharpe": sharpe_ratio(frame, periods_per_year=periods_per_year),
        "skewness": skewness(frame),
        "excess_kurtosis": kurtosis(frame),
        "historical_var": historical_var(frame, c),
        "parametric_var_normal": parametric_var(frame, c, distribution="normal"),
        "parametric_var_student_t": parametric_var(frame, c, distribution="student_t"),
        "modified_var": modified_var(frame, c),
        "es_historical": expected_shortfall(frame, c, method="historical"),
        "es_normal": expected_shortfall(frame, c, method="normal"),
        "es_student_t": expected_shortfall(frame, c, method="student_t"),
        "downside_deviation": downside_deviation(frame, periods_per_year=periods_per_year),
        "sortino": sortino_ratio(frame, periods_per_year=periods_per_year),
    }
    out = pd.DataFrame({k: (v if isinstance(v, pd.Series) else pd.Series([float(v)]))
                        for k, v in metrics.items()})
    out.index = [str(c_) for c_ in frame.columns]
    return out.T


__all__ = [
    "volatility", "skewness", "kurtosis", "correlation_matrix", "covariance_matrix",
    "annualized_return", "historical_var", "parametric_var", "modified_var",
    "cornish_fisher_z", "expected_shortfall", "cvar", "downside_deviation",
    "sortino_ratio", "sharpe_ratio", "tracking_error", "information_ratio",
    "beta", "jensen_alpha", "r_squared", "regression_stats", "regression_table",
    "RegressionStats", "risk_report",
]
