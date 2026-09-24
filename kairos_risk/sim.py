"""合成数据生成器：为示例与测试提供离线、固定 seed、可复现的收益/情景样本。

覆盖三类需求
------------
- **肥尾单资产收益**：学生 t、混合正态（regime switching）、帕累托型极端损失，
  用于校验 VaR/ES/EVT 在肥尾下的表现；
- **多资产面板**：给定波动与相关结构的多元正态 / 多元 t / 混合正态，
  用于风险分解、压力测试与回撤分析；
- **因子结构面板**：已知暴露 B、因子波动与特质波动的生成过程，
  同时返回**解析真值协方差**，便于白盒校验因子模型的还原精度。

所有函数默认使用 ``numpy.random.default_rng(seed)``，相同参数必得相同结果；
不联网、不读文件、不依赖 scipy。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

ArrayLike = Union[np.ndarray, Sequence[float]]

#: 默认资产名，多资产面板在未指定 names 时使用
DEFAULT_NAMES = ("ASSET_A", "ASSET_B", "ASSET_C", "ASSET_D", "ASSET_E", "ASSET_F")


def _business_index(n: int, start: str = "2018-01-02") -> pd.DatetimeIndex:
    """生成 n 个工作日的日期索引（离线、确定）。"""
    return pd.bdate_range(start, periods=int(n))


def _check_positive(x: ArrayLike, n: int, name: str) -> np.ndarray:
    """校验并归一「长度 n 的正数向量」参数。"""
    arr = np.asarray(x, dtype="float64").ravel()
    if arr.size != n:
        raise ValueError(f"{name} 长度 {arr.size} 与资产数 {n} 不一致")
    if np.any(arr <= 0) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} 必须全为正且有限")
    return arr


def _unit_variance_t(rng: np.random.Generator, size: Tuple[int, ...], df: float) -> np.ndarray:
    """方差为 1 的学生 t 抽样：T·√((ν−2)/ν)，便于与正态情形直接比较尺度。"""
    if df <= 2:
        raise ValueError("学生 t 的自由度必须 > 2（否则方差不存在）")
    t = rng.standard_t(float(df), size=size)
    return t * math.sqrt((float(df) - 2.0) / float(df))


def _equicorr(n: int, rho: float) -> np.ndarray:
    """等相关矩阵 ρ·11ᵀ + (1−ρ)I（ρ ∈ (−1/(n−1), 1) 时严格正定）。"""
    if not (-1.0 / max(n - 1, 1) < rho < 1.0):
        raise ValueError(f"rho={rho} 超出等相关矩阵的正定范围")
    return rho * np.ones((n, n)) + (1.0 - rho) * np.eye(n)


def _as_corr(corr: Optional[ArrayLike], n: int, rho: float) -> np.ndarray:
    """把相关矩阵输入归一为对称、单位对角的 PSD 矩阵（None 时用等相关）。"""
    if corr is None:
        return _equicorr(n, rho)
    arr = np.asarray(corr, dtype="float64")
    if arr.shape != (n, n):
        raise ValueError(f"corr 形状 {arr.shape} 与资产数 {n} 不一致")
    arr = (arr + arr.T) / 2.0
    if not np.allclose(np.diag(arr), 1.0, atol=1e-9):
        raise ValueError("corr 的对角必须全为 1")
    eig = np.linalg.eigvalsh(arr)
    if eig.min() <= 0:
        raise ValueError(f"corr 必须正定，最小特征值 {eig.min():.3e}")
    return arr


def make_student_t_returns(n: int = 2000, seed: int = 5, df: float = 4.0,
                           scale: float = 0.01, loc: float = 0.0,
                           start: str = "2018-01-02") -> pd.Series:
    """单资产学生 t 肥尾收益（方差 = scale²，均值 = loc）。"""
    rng = np.random.default_rng(int(seed))
    x = _unit_variance_t(rng, (int(n),), df) * float(scale) + float(loc)
    return pd.Series(x, index=_business_index(n, start), name="student_t")


def make_mixture_returns(n: int = 4000, seed: int = 13, crisis_prob: float = 0.05,
                         calm_sigma: float = 0.008, crisis_sigma: float = 0.05,
                         calm_mu: float = 0.0006, crisis_mu: float = -0.02,
                         start: str = "2018-01-02") -> pd.Series:
    """单资产混合正态收益：以 ``crisis_prob`` 概率进入「危机态」（波动放大、均值为负）。

    这类数据天然呈现负偏与肥尾，是检验 Cornish-Fisher 修正 VaR 与 EVT 的理想样本。
    """
    p = float(crisis_prob)
    if not (0.0 <= p < 1.0):
        raise ValueError("crisis_prob 必须落在 [0,1) 内")
    rng = np.random.default_rng(int(seed))
    crisis = rng.random(int(n)) < p
    sigma = np.where(crisis, float(crisis_sigma), float(calm_sigma))
    mu = np.where(crisis, float(crisis_mu), float(calm_mu))
    x = mu + sigma * rng.standard_normal(int(n))
    return pd.Series(x, index=_business_index(n, start), name="mixture")


def make_pareto_losses(n: int = 5000, seed: int = 11, xi: float = 0.3,
                       scale: float = 0.01, threshold: float = 0.02,
                       start: str = "2018-01-02") -> pd.Series:
    """帕累托/GPD 型极端损失序列（返回**收益**口径，即负的损失值）。

    生成过程：损失 L = threshold + Y，其中 Y ~ GPD(ξ, scale) 用逆变换
    Y = scale·((1−U)^(−ξ) − 1)/ξ 精确抽样（ξ→0 时取指数极限）。
    因此 :func:`kairos_risk.tail.fit_gpd` 在该样本上应还原出 (ξ, scale)。
    """
    if float(xi) == 0.0:
        raise ValueError("xi=0 请用指数分布工具；本函数只覆盖 ξ≠0 的 GPD 情形")
    rng = np.random.default_rng(int(seed))
    u = rng.random(int(n))
    y = float(scale) * ((1.0 - u) ** (-float(xi)) - 1.0) / float(xi)
    losses = float(threshold) + y
    return pd.Series(-losses, index=_business_index(n, start), name="pareto_loss")


def make_multi_asset_panel(n_assets: int = 4, n_periods: int = 750, seed: int = 3,
                           kind: str = "normal", df: float = 5.0,
                           vols: Optional[ArrayLike] = None,
                           corr: Optional[ArrayLike] = None, rho: float = 0.35,
                           drift: Optional[ArrayLike] = None,
                           names: Optional[Sequence[str]] = None,
                           crisis_prob: float = 0.06, crisis_scale: float = 3.5,
                           crisis_shift: float = -0.02,
                           start: str = "2018-01-02") -> pd.DataFrame:
    """多资产收益面板，支持 ``normal`` / ``student_t`` / ``mixture`` 三种分布。

    参数
    ----
    kind:   ``"normal"`` 多元正态；``"student_t"`` 多元 t（共用一个卡方缩放因子，
            保留椭圆相关结构且尾部同步加厚）；``"mixture"`` 正态混合
            （按 ``crisis_prob`` 抽取共同的危机期，波动放大 ``crisis_scale`` 倍、
            收益整体平移 ``crisis_shift``，形成同步暴跌样本）。
    vols:   各资产每期波动（默认 0.010~0.018 递减）；
    corr:   相关矩阵；None 时用等相关 ``rho``；
    drift:  各资产每期漂移（默认 0.0004~0.0001 递减）。

    返回 DataFrame(index=工作日, columns=资产名)。
    """
    if kind not in ("normal", "student_t", "mixture"):
        raise ValueError("kind 只支持 'normal' / 'student_t' / 'mixture'")
    k, n = int(n_assets), int(n_periods)
    if k <= 0 or n < 2:
        raise ValueError("n_assets 必须为正且 n_periods ≥ 2")
    rng = np.random.default_rng(int(seed))
    if vols is None:
        vols_arr = np.linspace(0.010, 0.018, k)
    else:
        vols_arr = _check_positive(vols, k, "vols")
    if drift is None:
        drift_arr = np.linspace(0.0004, 0.0001, k)
    else:
        drift_arr = np.asarray(drift, dtype="float64").ravel()
        if drift_arr.size != k:
            raise ValueError(f"drift 长度 {drift_arr.size} 与资产数 {k} 不一致")
    r = _as_corr(corr, k, float(rho))
    chol = np.linalg.cholesky(r)

    if kind == "student_t":
        z = _unit_variance_t(rng, (n, k), df)
    else:
        z = rng.standard_normal((n, k))
    x = z @ chol.T                                   # 引入目标相关结构
    if kind == "mixture":
        crisis = rng.random(n) < float(crisis_prob)
        scale = np.where(crisis, float(crisis_scale), 1.0)
        shift = np.where(crisis, float(crisis_shift), 0.0)
        x = x * scale[:, None] + shift[:, None]
    out = x * vols_arr + drift_arr
    cols = list(names) if names is not None else list(DEFAULT_NAMES[:k]) \
        if k <= len(DEFAULT_NAMES) else [f"ASSET_{i}" for i in range(k)]
    if len(cols) != k:
        raise ValueError(f"names 长度 {len(cols)} 与资产数 {k} 不一致")
    return pd.DataFrame(out, index=_business_index(n, start), columns=cols)


def make_crash_panel(n_assets: int = 4, n_periods: int = 500, seed: int = 17,
                     crash_positions: Sequence[int] = (120, 260, 400),
                     crash_magnitude: float = -0.18, betas: Optional[ArrayLike] = None,
                     vols: Optional[ArrayLike] = None,
                     names: Optional[Sequence[str]] = None,
                     start: str = "2018-01-02") -> pd.DataFrame:
    """带若干次「同步暴跌日」的多资产面板，专供压力测试 / 历史重放演示。

    暴跌日的资产收益 ≈ beta_i × crash_magnitude（叠加少量噪声），
    因此历史重放会把这几天挑为最差窗口。
    """
    k, n = int(n_assets), int(n_periods)
    panel = make_multi_asset_panel(n_assets=k, n_periods=n, seed=int(seed), kind="normal",
                                  vols=vols, names=names, rho=0.4, start=start)
    if betas is None:
        betas_arr = np.linspace(1.3, 0.4, k)
    else:
        betas_arr = np.asarray(betas, dtype="float64").ravel()
        if betas_arr.size != k:
            raise ValueError(f"betas 长度 {betas_arr.size} 与资产数 {k} 不一致")
    rng = np.random.default_rng(int(seed) + 1000)
    for pos in crash_positions:
        p = int(pos)
        if not (0 <= p < n):
            raise ValueError(f"crash_positions 含越界位置 {p}（样本长度 {n}）")
        panel.iloc[p] = betas_arr * float(crash_magnitude) + rng.standard_normal(k) * 0.004
    return panel


@dataclass(frozen=True, eq=False)
class FactorPanel:
    """因子结构合成面板，附带生成过程的解析真值（便于白盒校验因子模型）。

    字段
    ----
    assets:      资产收益面板（index=期次, columns=资产）；
    factors:     因子收益面板（index=期次, columns=因子）；
    exposures:   真实暴露矩阵 B（index=资产, columns=因子）；
    alphas:      真实每期 α；
    idio_std:    真实特质波动；
    factor_std:  各因子波动（因子间独立）；
    true_cov:    解析真值协方差 Σ = B·F·Bᵀ + D；
    n_obs:       期数。
    """

    assets: pd.DataFrame
    factors: pd.DataFrame
    exposures: pd.DataFrame
    alphas: pd.Series
    idio_std: pd.Series
    factor_std: pd.Series
    true_cov: pd.DataFrame
    n_obs: int


def make_factor_panel(n_assets: int = 6, n_factors: int = 2, n_periods: int = 600,
                      seed: int = 7, df: float = 8.0, tail: bool = True,
                      exposure_scale: float = 0.7, idio_std: float = 0.006,
                      factor_std: float = 0.010, alpha: float = 0.0002,
                      exposures: Optional[np.ndarray] = None,
                      names: Optional[Sequence[str]] = None,
                      factor_names: Optional[Sequence[str]] = None,
                      start: str = "2018-01-02") -> FactorPanel:
    """生成已知暴露的多因子收益面板：r = α + B·f + ε。

    参数
    ----
    tail:  True 时因子与特质噪声都用方差归一的学生 t（自由度 df），制造肥尾；
    exposures: 可显式指定真实暴露矩阵 B（形状 n_assets×n_factors），便于构造回归测试。

    因子之间相互独立，故真值协方差为 Σ = B·diag(factor_std²)·Bᵀ + diag(idio_std²)。
    """
    p, k, n = int(n_assets), int(n_factors), int(n_periods)
    if p <= 0 or k <= 0 or n < k + 3:
        raise ValueError(f"非法维度：n_assets={p}, n_factors={k}, n_periods={n}")
    rng = np.random.default_rng(int(seed))
    if exposures is None:
        b = rng.standard_normal((p, k)) * float(exposure_scale)
    else:
        b = np.asarray(exposures, dtype="float64")
        if b.shape != (p, k):
            raise ValueError(f"exposures 形状 {b.shape} 与 (n_assets, n_factors)=({p},{k}) 不一致")
    f_std = np.full(k, float(factor_std)) if np.isscalar(factor_std) \
        else _check_positive(factor_std, k, "factor_std")
    i_std = np.full(p, float(idio_std)) if np.isscalar(idio_std) \
        else _check_positive(idio_std, p, "idio_std")

    if tail:
        f = _unit_variance_t(rng, (n, k), df) * f_std
        eps = _unit_variance_t(rng, (n, p), df) * i_std
    else:
        f = rng.standard_normal((n, k)) * f_std
        eps = rng.standard_normal((n, p)) * i_std

    y = f @ b.T + eps + float(alpha)
    idx = _business_index(n, start)
    assets = list(names) if names is not None else (
        list(DEFAULT_NAMES[:p]) if p <= len(DEFAULT_NAMES) else [f"ASSET_{i}" for i in range(p)])
    factors = list(factor_names) if factor_names is not None else [f"FACTOR_{j}" for j in range(k)]
    if len(assets) != p or len(factors) != k:
        raise ValueError("names / factor_names 长度与维度不一致")

    f_cov = np.diag(f_std ** 2)
    true_cov = b @ f_cov @ b.T + np.diag(i_std ** 2)
    return FactorPanel(
        assets=pd.DataFrame(y, index=idx, columns=assets),
        factors=pd.DataFrame(f, index=idx, columns=factors),
        exposures=pd.DataFrame(b, index=assets, columns=factors),
        alphas=pd.Series(np.full(p, float(alpha)), index=assets, name="alpha"),
        idio_std=pd.Series(i_std, index=assets, name="idio_std"),
        factor_std=pd.Series(f_std, index=factors, name="factor_std"),
        true_cov=pd.DataFrame((true_cov + true_cov.T) / 2.0, index=assets, columns=assets),
        n_obs=n,
    )


def make_drawdown_path(up_periods: int = 5, down_periods: int = 4,
                       recover_periods: int = 9, up: float = 0.02, down: float = -0.03,
                       recover: Optional[float] = None, start: str = "2020-01-02",
                       freq: str = "B") -> pd.Series:
    """确定性的「上涨 -> 下跌 -> 修复」三段式收益序列（无随机成分）。

    用于精确校验最大回撤、回撤区间与恢复期：
    净值峰 = (1+up)^up_periods，谷 = 峰·(1+down)^down_periods，
    深度 = 1 − 谷/峰；恢复期数取决于 ``recover`` 与 ``recover_periods``，
    若不足以收复前高，则该区间以「未收复」结束（当前回撤）。
    """
    up_n, dn_n, rc_n = int(up_periods), int(down_periods), int(recover_periods)
    if up_n < 0 or dn_n < 0 or rc_n < 0:
        raise ValueError("各阶段期数不能为负")
    if dn_n == 0:
        raise ValueError("down_periods 必须为正，否则不构成回撤")
    rc = float(up) if recover is None else float(recover)
    parts = [np.full(up_n, float(up)), np.full(dn_n, float(down)), np.full(rc_n, rc)]
    arr = np.concatenate(parts)
    for v in arr:
        if v <= -1.0:
            raise ValueError("收益率不得 ≤ −100%")
    n = arr.size
    if n == 0:
        raise ValueError("序列为空")
    index = pd.bdate_range(start, periods=n) if freq == "B" else pd.RangeIndex(n)
    return pd.Series(arr, index=index, name="drawdown_path")


__all__ = [
    "DEFAULT_NAMES", "make_student_t_returns", "make_mixture_returns",
    "make_pareto_losses", "make_multi_asset_panel", "make_crash_panel",
    "FactorPanel", "make_factor_panel", "make_drawdown_path",
]
