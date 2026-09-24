"""因子风险模型模块：因子暴露估计、因子协方差、特质风险与组合的因子风险分解。

模型
----
多因子线性收益模型（每资产一条时序回归）：

    r_it = α_i + Σ_k β_ik·f_kt + ε_it ,   E[ε] = 0, Cov(ε) = D（对角）

据此资产协方差被分解为「因子部分 + 特质部分」：

    Σ ≈ B·F·Bᵀ + D

其中 B 为暴露矩阵（n×k）、F 为因子收益协方差（k×k）、D 为特质方差对角阵。
组合权重 w 下的组合暴露 b_p = Bᵀw，组合方差 = b_pᵀF b_p + wᵀDw，
两部分都是权重的二次型，可再用欧拉定理逐因子 / 逐资产做**精确可加**的贡献分解。

实现约定
--------
- 暴露与 α 用 ``numpy.linalg.lstsq``（带截距列）一次性对全部资产做最小二乘，
  不依赖 scipy / sklearn；
- 因子协方差用因子收益的样本协方差（ddof=1）；
- 特质方差用残差平方和 /(T − 参数个数)，为无偏估计；
- 支持单因子（市场模型）与多因子；``market_model`` 是单因子便捷入口。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from ._util import Weights, as_returns_frame, as_weights, wrap_1d

PanelLike = Union[pd.DataFrame, np.ndarray]


def _aligned(asset_returns: PanelLike, factor_returns: PanelLike):
    """资产面板与因子面板按索引内连接对齐，返回 (Y, F, assets, factors, index)。"""
    y_frame = as_returns_frame(asset_returns, name="asset_returns")
    f_frame = as_returns_frame(factor_returns, name="factor_returns")
    overlap = y_frame.index.intersection(f_frame.index)
    if len(overlap) < 3:
        raise ValueError(f"资产与因子收益的共同观测仅 {len(overlap)} 个，至少需要 3 个")
    y_frame = y_frame.loc[overlap]
    f_frame = f_frame.loc[overlap]
    common = [c for c in y_frame.columns if c in f_frame.columns]
    if common:
        raise ValueError(f"资产与因子存在同名列 {common}，请先重命名以免歧义")
    return (y_frame.to_numpy(dtype="float64"), f_frame.to_numpy(dtype="float64"),
            y_frame.columns, f_frame.columns, overlap)


@dataclass(frozen=True, eq=False)
class FactorModel:
    """因子风险模型估计结果。

    字段
    ----
    exposures:  暴露矩阵 B（index=资产，columns=因子）；
    alphas:     每资产每期截距 α；
    factor_cov: 因子收益协方差 F（k×k）；
    idio_var:   每资产特质方差 D_ii（残差方差，无偏）；
    r2:         每资产拟合优度（因子解释的方差占比）；
    resid_std:  每资产残差标准差（特质波动）；
    assets / factors: 资产与因子名称索引；
    n_obs:      参与估计的期数；
    has_alpha:  是否估计了截距项。
    """

    exposures: pd.DataFrame
    alphas: pd.Series
    factor_cov: pd.DataFrame
    idio_var: pd.Series
    r2: pd.Series
    resid_std: pd.Series
    assets: pd.Index
    factors: pd.Index
    n_obs: int
    has_alpha: bool = True

    @property
    def n_factors(self) -> int:
        """因子个数 k。"""
        return int(self.exposures.shape[1])

    def implied_cov(self) -> pd.DataFrame:
        """模型隐含协方差 Σ = B·F·Bᵀ + D（结构化的低秩 + 对角近似）。"""
        b = self.exposures.to_numpy(dtype="float64")
        f = self.factor_cov.to_numpy(dtype="float64")
        d = self.idio_var.to_numpy(dtype="float64")
        sigma = b @ f @ b.T + np.diag(d)
        cols = self.assets
        return pd.DataFrame((sigma + sigma.T) / 2.0, index=cols, columns=cols)

    def total_var(self) -> float:
        """模型隐含的每资产方差之和的对角迹（诊断用）。"""
        return float(np.trace(self.implied_cov().to_numpy()))


def fit_factor_model(asset_returns: PanelLike, factor_returns: PanelLike,
                     fit_alpha: bool = True) -> FactorModel:
    """用最小二乘估计因子暴露、α、因子协方差与特质风险（``numpy.linalg.lstsq``）。

    参数
    ----
    asset_returns:  资产收益面板（index=期次，columns=资产）。
    factor_returns: 因子收益面板（index=期次，columns=因子）。
    fit_alpha:      True 时在设计矩阵首列加常数项估计每期 α（标准 CAPM/多因子做法）；
                    False 时强制过原点回归（纯暴露模型）。

    返回 :class:`FactorModel`。要求共同观测数 > 参数个数，否则抛 ValueError。
    """
    y, f, assets, factors, _ = _aligned(asset_returns, factor_returns)
    n_obs, n_assets = y.shape
    k = f.shape[1]
    if k == 0:
        raise ValueError("因子面板至少需要一个因子")
    design = np.hstack([np.ones((n_obs, 1)), f]) if fit_alpha else f
    n_params = design.shape[1]
    if n_obs <= n_params + 1:
        raise ValueError(f"观测数 {n_obs} 相对参数数 {n_params} 过少，无法估计")
    coef, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank < n_params:
        raise ValueError(f"设计矩阵秩亏（rank={rank} < {n_params}），因子间可能完全共线")
    fitted = design @ coef
    resid = y - fitted
    alphas = coef[0] if fit_alpha else np.zeros(n_assets)
    exposures = (coef[1:] if fit_alpha else coef).T                 # (n_assets, k)

    fc = f - f.mean(axis=0)
    factor_cov = fc.T @ fc / float(n_obs - 1)
    idio_var = (resid * resid).sum(axis=0) / float(n_obs - n_params)
    total_ss = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    resid_ss = (resid * resid).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(total_ss > 0, 1.0 - resid_ss / np.maximum(total_ss, 1e-300), 0.0)
    r2 = np.clip(r2, 0.0, 1.0)

    return FactorModel(
        exposures=pd.DataFrame(exposures, index=assets, columns=factors),
        alphas=pd.Series(alphas, index=assets, name="alpha"),
        factor_cov=pd.DataFrame((factor_cov + factor_cov.T) / 2.0,
                                index=factors, columns=factors),
        idio_var=pd.Series(np.maximum(idio_var, 0.0), index=assets, name="idio_var"),
        r2=pd.Series(r2, index=assets, name="r2"),
        resid_std=pd.Series(np.sqrt(np.maximum(idio_var, 0.0)), index=assets, name="resid_std"),
        assets=assets,
        factors=factors,
        n_obs=n_obs,
        has_alpha=bool(fit_alpha),
    )


def market_model(asset_returns: PanelLike, market_returns: Union[pd.Series, np.ndarray],
                 market_name: str = "MKT", fit_alpha: bool = True) -> FactorModel:
    """单因子（市场模型）便捷入口：把一维市场收益包成单列因子面板后拟合。"""
    if isinstance(market_returns, pd.Series):
        f = market_returns.to_frame(market_returns.name or market_name)
    else:
        arr = np.asarray(market_returns, dtype="float64").ravel()
        idx = None
        if isinstance(asset_returns, pd.DataFrame):
            idx = asset_returns.index
        f = pd.DataFrame({market_name: arr}, index=idx)
    return fit_factor_model(asset_returns, f, fit_alpha=fit_alpha)


@dataclass(frozen=True, eq=False)
class FactorRiskBreakdown:
    """组合层面的因子风险 / 特质风险分解（所有贡献均为**方差口径的占比**）。

    字段
    ----
    portfolio_exposure: b_p = Bᵀw，组合对各因子的暴露；
    factor_variance:    b_pᵀF b_p，组合方差中的因子部分；
    idio_variance:      wᵀDw = Σ_i w_i²·D_ii，组合方差中的特质部分；
    total_variance:     两者之和，等于 wᵀ(BFBᵀ+D)w；
    volatility:         √total_variance（每期）；
    factor_contrib:     逐因子欧拉贡献 b_p,k·(F b_p)_k，和 = factor_variance；
    factor_pct:         factor_contrib/total_variance；
    idio_contrib:       逐资产特质贡献 w_i²·D_ii，和 = idio_variance；
    idio_pct:           idio_contrib/total_variance；
    asset_contrib:      逐资产**总**风险贡献（对隐含协方差 Σ 的欧拉分解），和 = σ_p；
    asset_pct:          asset_contrib/σ_p；
    factor_share:       factor_variance/total_variance（因子风险占比）。
    """

    portfolio_exposure: pd.Series
    factor_variance: float
    idio_variance: float
    total_variance: float
    volatility: float
    factor_contrib: pd.Series
    factor_pct: pd.Series
    idio_contrib: pd.Series
    idio_pct: pd.Series
    asset_contrib: pd.Series
    asset_pct: pd.Series
    factor_share: float
    weights: Optional[pd.Series] = None


def factor_risk_decomposition(weights: Weights, model: FactorModel) -> FactorRiskBreakdown:
    """把组合风险拆成「逐因子贡献 + 逐资产特质贡献」，两部分各自精确可加。

    因子部分是 b_p 的二次型，欧拉分解给出 contrib_k = b_p,k·(F b_p)_k；
    特质部分因 D 为对角阵，天然逐资产可加 contrib_i = w_i²·D_ii。
    """
    w, _ = as_weights(weights, model.assets, name="weights")
    b = model.exposures.to_numpy(dtype="float64")
    f = model.factor_cov.to_numpy(dtype="float64")
    d = model.idio_var.to_numpy(dtype="float64")
    if w.size != b.shape[0]:
        raise ValueError(f"weights 维度 {w.size} 与资产数 {b.shape[0]} 不一致")

    b_p = b.T @ w                                   # 组合因子暴露
    f_bp = f @ b_p
    factor_var = float(b_p @ f_bp)
    idio_var = float((w * w) @ d)
    total_var = factor_var + idio_var
    vol = math.sqrt(max(total_var, 0.0))

    denom = total_var if total_var > 1e-300 else float("nan")
    factor_contrib = b_p * f_bp
    idio_contrib = w * w * d

    sigma = b @ f @ b.T + np.diag(d)
    sw = sigma @ w
    asset_contrib = w * sw / vol if vol > 1e-14 else np.zeros_like(w)

    return FactorRiskBreakdown(
        portfolio_exposure=pd.Series(b_p, index=model.factors, name="portfolio_exposure"),
        factor_variance=factor_var,
        idio_variance=idio_var,
        total_variance=total_var,
        volatility=vol,
        factor_contrib=pd.Series(factor_contrib, index=model.factors, name="factor_contrib"),
        factor_pct=pd.Series(factor_contrib / denom, index=model.factors, name="factor_pct"),
        idio_contrib=pd.Series(idio_contrib, index=model.assets, name="idio_contrib"),
        idio_pct=pd.Series(idio_contrib / denom, index=model.assets, name="idio_pct"),
        asset_contrib=pd.Series(asset_contrib, index=model.assets, name="asset_contrib"),
        asset_pct=pd.Series(asset_contrib / vol if vol > 1e-14 else np.zeros_like(w),
                            index=model.assets, name="asset_pct"),
        factor_share=(factor_var / denom) if total_var > 1e-300 else float("nan"),
        weights=pd.Series(w, index=model.assets, name="weight"),
    )


def factor_variance(weights: Weights, model: FactorModel) -> float:
    """组合方差中的因子部分 b_pᵀF b_p。"""
    return factor_risk_decomposition(weights, model).factor_variance


def idiosyncratic_variance(weights: Weights, model: FactorModel) -> float:
    """组合方差中的特质部分 wᵀDw。"""
    return factor_risk_decomposition(weights, model).idio_variance


def portfolio_exposure(weights: Weights, model: FactorModel) -> pd.Series:
    """组合因子暴露 b_p = Bᵀw。"""
    return factor_risk_decomposition(weights, model).portfolio_exposure


def factor_var_contributions(model: FactorModel) -> pd.DataFrame:
    """因子协方差的结构诊断表：每个因子的方差、与其它因子的平均相关。"""
    f = model.factor_cov
    d = np.sqrt(np.diag(f.to_numpy(dtype="float64")))
    rows: Dict[str, Dict[str, float]] = {}
    corr = f.to_numpy(dtype="float64") / np.outer(d, d) if np.all(d > 0) else np.eye(len(d))
    for i, name in enumerate(f.columns):
        rows[str(name)] = {
            "std": float(d[i]),
            "variance": float(d[i] ** 2),
            "mean_abs_corr": float(np.mean(np.abs(np.delete(corr[i], i)))) if len(d) > 1 else 0.0,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def exposure_summary(model: FactorModel) -> pd.DataFrame:
    """每资产的暴露/α/R²/特质波动汇总表（index=资产）。"""
    out = model.exposures.copy()
    out.insert(0, "alpha", model.alphas.to_numpy())
    out["r2"] = model.r2.to_numpy()
    out["idio_std"] = model.resid_std.to_numpy()
    return out


__all__ = [
    "FactorModel", "FactorRiskBreakdown", "fit_factor_model", "market_model",
    "factor_risk_decomposition", "factor_variance", "idiosyncratic_variance",
    "portfolio_exposure", "factor_var_contributions", "exposure_summary",
]
