"""组合风险分解模块：波动的边际/成分贡献（MRC/CRC）、成分 VaR / 成分 ES（欧拉分配）、
分散化比率与风险集中度。

核心思想
--------
组合波动 σ_p = √(wᵀΣw) 是权重的一次齐次函数，由欧拉定理：

    σ_p = Σ_i w_i · ∂σ_p/∂w_i = Σ_i w_i · (Σw)_i / σ_p

因此「成分风险贡献」CRC_i = w_i·(Σw)_i/σ_p **精确可加**（Σ CRC_i = σ_p），
CRC_i/σ_p 即该资产对组合总风险的占比，是风险平价与风险预算的校验口径。

同样的欧拉思路可搬到尾部：把 ES 写成尾部样本上的线性泛函
ES = −E[wᵀr | r_p 落在尾部] = Σ_i w_i·(−E[r_i | 尾部])，
于是成分 ES 也精确可加；成分 VaR 则用「尾部均值估计 + 按 VaR/ES 比例缩放」
的工程近似（严格的 VaR 边际需要条件密度 E[r_i | r_p = q]，样本上不可直接估计）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from ._util import (
    ReturnsLike,
    Weights,
    as_returns_frame,
    as_weights,
    check_confidence,
    quantile_sorted,
    wrap_1d,
)


def _as_cov(cov: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, Optional[pd.Index]]:
    """把协方差输入归一为 (对称化 ndarray, 列索引或 None)。"""
    if isinstance(cov, pd.DataFrame):
        arr = cov.to_numpy(dtype="float64")
        index: Optional[pd.Index] = cov.columns
    else:
        arr = np.asarray(cov, dtype="float64")
        index = None
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"cov 必须是方阵，当前形状 {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("cov 含 NaN/Inf")
    return (arr + arr.T) / 2.0, index


def _align(weights: Weights, cov: Union[pd.DataFrame, np.ndarray]):
    """对齐「(权重, 协方差)」成对输入：Series 权重按协方差列名 reindex。"""
    sigma, index = _as_cov(cov)
    w, _ = as_weights(weights, index, name="weights")
    if w.size != sigma.shape[0]:
        raise ValueError(f"weights 维度 {w.size} 与 cov 维度 {sigma.shape[0]} 不一致")
    return w, sigma, index


# --------------------------------------------------------------------------- #
# 波动分解（MRC / CRC）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class RiskDecomposition:
    """组合波动的风险分解结果（向量字段与输入 weights 同型：Series 或 ndarray）。

    字段
    ----
    marginal:      边际风险贡献 MRC_i = ∂σ_p/∂w_i = (Σw)_i/σ_p；
    component:     成分风险贡献 CRC_i = w_i·MRC_i，其和**恰等于** σ_p（欧拉可加性）；
    pct:           风险占比 CRC_i/σ_p，其和为 1；
    volatility:    组合波动 σ_p = √(wᵀΣw)；
    weighted_volatility: 加权个体波动 Σ_i |w_i|·σ_i（完全不分散时的波动上界口径）；
    diversification_ratio: 分散化比率 = weighted_volatility/σ_p ≥ 1，越大越分散；
    herfindahl:    风险占比的赫芬达尔指数 Σ pct_i²，衡量风险集中度；
    effective_n:   有效资产数 1/Σ pct_i²，风险贡献均等时等于资产个数。
    """

    marginal: Weights
    component: Weights
    pct: Weights
    volatility: float
    weighted_volatility: float
    diversification_ratio: float
    herfindahl: float
    effective_n: float


def component_risk(weights: Weights, cov: Union[pd.DataFrame, np.ndarray]) -> RiskDecomposition:
    """计算组合波动的边际/成分风险贡献、分散化比率与风险集中度。"""
    w, sigma, index = _align(weights, cov)
    var = float(w @ sigma @ w)
    vol = math.sqrt(max(var, 0.0))
    if vol <= 1e-14:
        raise ValueError("组合波动接近零，风险贡献无定义")
    sw = sigma @ w
    marginal = sw / vol
    component = w * marginal
    pct = component / vol
    asset_vol = np.sqrt(np.maximum(np.diag(sigma), 0.0))
    weighted_vol = float(np.abs(w) @ asset_vol)
    herf = float(pct @ pct)
    return RiskDecomposition(
        marginal=wrap_1d(marginal, index, name="marginal"),
        component=wrap_1d(component, index, name="component"),
        pct=wrap_1d(pct, index, name="pct"),
        volatility=vol,
        weighted_volatility=weighted_vol,
        diversification_ratio=(weighted_vol / vol) if weighted_vol > 0 else float("nan"),
        herfindahl=herf,
        effective_n=(1.0 / herf) if herf > 0 else float("inf"),
    )


def marginal_risk(weights: Weights, cov: Union[pd.DataFrame, np.ndarray]) -> Weights:
    """边际风险贡献 MRC = Σw/σ_p（与输入同型）。"""
    return component_risk(weights, cov).marginal


def diversification_ratio(weights: Weights, cov: Union[pd.DataFrame, np.ndarray]) -> float:
    """分散化比率 DR = Σ_i |w_i|σ_i / σ_p。

    DR = 1 表示完全无分散效果（如单一资产，或所有资产完全正相关）；
    DR 越大说明相关性带来的分散收益越显著。
    """
    return component_risk(weights, cov).diversification_ratio


def risk_effective_n(weights: Weights, cov: Union[pd.DataFrame, np.ndarray]) -> float:
    """风险意义上的有效资产数 1/Σ pct_i²（等风险贡献时等于资产个数）。"""
    return component_risk(weights, cov).effective_n


# --------------------------------------------------------------------------- #
# 尾部分解（成分 VaR / 成分 ES）
# --------------------------------------------------------------------------- #
def portfolio_returns(returns: ReturnsLike, weights: Weights) -> pd.Series:
    """组合每期收益 r_p = R·w（返回带期次索引的 Series）。"""
    frame = as_returns_frame(returns)
    w, _ = as_weights(weights, frame.columns, name="weights")
    return pd.Series(frame.to_numpy(dtype="float64") @ w, index=frame.index, name="portfolio")


@dataclass(frozen=True, eq=False)
class TailDecomposition:
    """组合尾部风险（VaR / ES）的欧拉分解结果。

    字段
    ----
    measure:    ``"es"`` 或 ``"var"``；
    value:      组合层面的 VaR / ES（正数损失）；
    marginal:   尾部边际贡献 ∂/∂w_i = −E[r_i | 尾部]（``"var"`` 时为缩放后的值）；
    component:  w_i·marginal_i，其和等于 ``value``（可加性）；
    pct:        component_i/value；
    threshold:  组合收益的尾部分位 q_{1−c}（负数，尾部即 r_p ≤ threshold）；
    n_tail:     尾部样本数；
    n_obs:      总样本数；
    confidence: 置信度 c。
    """

    measure: str
    value: float
    marginal: Weights
    component: Weights
    pct: Weights
    threshold: float
    n_tail: int
    n_obs: int
    confidence: float


def tail_mask(port_rets: np.ndarray, confidence: float) -> np.ndarray:
    """返回组合收益的「损失尾部」掩码：r_p ≤ q_{1−c}（至少包含最差一期）。"""
    c = check_confidence(confidence)
    threshold = quantile_sorted(port_rets, 1.0 - c)
    mask = port_rets <= threshold
    if not mask.any():
        mask = port_rets <= port_rets.min()
    return mask


def _tail_decompose(weights: Weights, returns: ReturnsLike, confidence: float,
                    measure: str) -> TailDecomposition:
    """成分 VaR / 成分 ES 的共同实现。

    尾部样本上取各资产收益均值，得到 ES 的精确欧拉边际；``measure="var"`` 时
    再乘以 VaR/ES 比例，使成分之和恰好等于组合 VaR（工程上通用的可加近似）。
    """
    frame = as_returns_frame(returns)
    w, _ = as_weights(weights, frame.columns, name="weights")
    # 「同型进出」：仅 DataFrame 输入才带资产索引输出 Series，ndarray 输入输出 ndarray
    columns = frame.columns if isinstance(returns, pd.DataFrame) else None
    if w.size != frame.shape[1]:
        raise ValueError(f"weights 维度 {w.size} 与资产数 {frame.shape[1]} 不一致")
    x = frame.to_numpy(dtype="float64")
    port = x @ w
    mask = tail_mask(port, confidence)
    tail = x[mask]
    n_tail = int(tail.shape[0])

    es_marginal = -tail.mean(axis=0)              # ∂ES/∂w_i = −E[r_i | 尾部]
    es_value = float(w @ es_marginal)             # = −mean(port[尾部])，即组合 ES
    threshold = float(quantile_sorted(port, 1.0 - check_confidence(confidence)))

    if measure == "es":
        marginal, value = es_marginal, es_value
    else:
        value = -float(threshold)                 # 组合历史 VaR（正数损失）
        scale = value / es_value if abs(es_value) > 1e-18 else 0.0
        marginal = es_marginal * scale
    component = w * marginal
    pct = component / value if abs(value) > 1e-18 else np.zeros_like(component)
    return TailDecomposition(
        measure=measure,
        value=value,
        marginal=wrap_1d(marginal, columns, name="marginal"),
        component=wrap_1d(component, columns, name="component"),
        pct=wrap_1d(pct, columns, name="pct"),
        threshold=threshold,
        n_tail=n_tail,
        n_obs=int(x.shape[0]),
        confidence=check_confidence(confidence),
    )


def component_es(weights: Weights, returns: ReturnsLike,
                 confidence: float = 0.95) -> TailDecomposition:
    """成分 ES（精确欧拉分配）：Σ_i component_i = 组合 ES。"""
    return _tail_decompose(weights, returns, confidence, "es")


def component_var(weights: Weights, returns: ReturnsLike,
                  confidence: float = 0.95) -> TailDecomposition:
    """成分 VaR（尾部均值估计后按 VaR/ES 缩放）：Σ_i component_i = 组合 VaR。"""
    return _tail_decompose(weights, returns, confidence, "var")


def worst_tail_sample(weights: Weights, returns: ReturnsLike,
                      confidence: float = 0.95) -> pd.DataFrame:
    """返回落在组合损失尾部的原始资产收益子面板（用于人工核查成分风险来源）。"""
    frame = as_returns_frame(returns)
    w, _ = as_weights(weights, frame.columns, name="weights")
    port = frame.to_numpy(dtype="float64") @ w
    return frame[tail_mask(port, confidence)]
