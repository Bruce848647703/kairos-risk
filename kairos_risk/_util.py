"""内部通用工具：分布函数、输入归一化与 scipy 可选增强。

不属于公开 API，供 measures / decomposition / factor / tail / backtest_risk 复用。

设计要点
--------
- **单一 scipy 入口**：全库只通过 :func:`_try_import_scipy` 取用 scipy；缺失（或测试中被
  monkeypatch）时自动回退到纯 numpy / math 的自研实现，因此无 scipy 环境也能跑通全部功能。
- **纯 Python 特殊函数**：正则不完全贝塔函数（Lentz 连分式）配合 ``math.erf/erfc``，
  自研支撑标准正态与学生 t 的 CDF/PPF、Clopper-Pearson 精确区间。
- **同型进出**：DataFrame 输入返回按资产索引的 Series，Series/ndarray 输入返回标量，
  避免调用方反复做类型判断。
- **符号约定**：所有「损失」类度量返回正数（0.02 表示损失 2%）。
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

ReturnsLike = Union[pd.DataFrame, pd.Series, np.ndarray]
Weights = Union[np.ndarray, pd.Series]

#: 欧拉-马歇罗尼常数，用于「多次试验下的最大夏普期望」公式
EULER_MASCHERONI = 0.577215664901532861


# --------------------------------------------------------------------------- #
# scipy 可选增强：全库唯一入口，便于测试 monkeypatch 出「无 scipy」环境
# --------------------------------------------------------------------------- #
def _try_import_scipy() -> Optional[SimpleNamespace]:
    """尝试导入 scipy 的 ``stats`` 与 ``special``；不可用时返回 ``None``。

    返回 ``SimpleNamespace(stats=..., special=...)``。任何调用方都必须先判空，
    为空时走本模块的纯 numpy/math 回退实现。测试可通过 monkeypatch 本函数
    （令其返回 ``None``）来验证回退路径。
    """
    try:
        from scipy import special, stats  # noqa: F401
    except Exception:  # pragma: no cover - 取决于运行环境
        return None
    return SimpleNamespace(stats=stats, special=special)


# --------------------------------------------------------------------------- #
# 标准正态分布
# --------------------------------------------------------------------------- #
def norm_cdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """标准正态 CDF Φ(x)。

    scipy 可用时走 ``special.ndtr``（向量化且精度高）；否则用
    ``Φ(x) = 0.5·(1 + erf(x/√2))`` 的纯 math 回退。标量进标量出。
    """
    scalar = np.ndim(x) == 0
    arr = np.atleast_1d(np.asarray(x, dtype="float64"))
    sp = _try_import_scipy()
    if sp is not None:
        out = np.asarray(sp.special.ndtr(arr), dtype="float64")
    else:
        out = np.array([0.5 * math.erfc(-v / math.sqrt(2.0)) for v in arr], dtype="float64")
    return float(out[0]) if scalar else out


def norm_pdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """标准正态密度 φ(x) = exp(−x²/2)/√(2π)（纯 numpy，无需 scipy）。"""
    arr = np.asarray(x, dtype="float64")
    out = np.exp(-0.5 * arr * arr) / math.sqrt(2.0 * math.pi)
    return float(out) if out.ndim == 0 else out


def norm_ppf(p: float) -> float:
    """标准正态分位函数 Φ⁻¹(p)。

    scipy 可用时用 ``special.ndtri``；否则在 ``Φ(z) = 0.5·erfc(−z/√2)`` 上做
    100 次二分求根（区间 [−40, 40]，分辨率远超双精度），确定且零依赖。
    """
    q = float(p)
    if not (0.0 < q < 1.0):
        raise ValueError(f"p 必须落在 (0,1) 内，收到 {q}")
    sp = _try_import_scipy()
    if sp is not None:
        return float(sp.special.ndtri(q))
    lo, hi = -40.0, 40.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if 0.5 * math.erfc(-mid / math.sqrt(2.0)) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# 正则不完全贝塔函数（自研 Lentz 连分式）—— 学生 t 与精确二项区间的基石
# --------------------------------------------------------------------------- #
def _betacf(a: float, b: float, x: float, max_iter: int = 400,
            eps: float = 1e-15, fpmin: float = 1e-300) -> float:
    """不完全贝塔函数的连分式部分（修正 Lentz 算法）。

    连分式对 x < (a+1)/(a+b+2) 收敛快，调用方 :func:`betainc` 会用对称式
    ``I_x(a,b) = 1 − I_{1−x}(b,a)`` 保证始终落在快收敛区间。
    """
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """正则不完全贝塔函数 I_x(a,b) = B(x;a,b)/B(a,b)，值域 [0,1]。

    前置因子用 ``lgamma`` 在对数域计算，避免 a、b 较大时的阶乘溢出。
    """
    a, b, x = float(a), float(b), float(x)
    if a <= 0.0 or b <= 0.0:
        raise ValueError(f"a、b 必须为正，收到 a={a}, b={b}")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(log_beta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def betaincinv(a: float, b: float, p: float, max_iter: int = 200) -> float:
    """正则不完全贝塔函数的反函数 I⁻¹_p(a,b)。

    scipy 可用时走 ``special.betaincinv``；否则利用 I_x(a,b) 关于 x 在 [0,1]
    单调不减的性质做二分求根（用于 Clopper-Pearson 精确二项区间）。
    """
    p = float(p)
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p 必须落在 [0,1] 内，收到 {p}")
    if a <= 0.0 or b <= 0.0:
        raise ValueError(f"a、b 必须为正，收到 a={a}, b={b}")
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    sp = _try_import_scipy()
    if sp is not None:
        return float(sp.special.betaincinv(float(a), float(b), p))
    lo, hi = 0.0, 1.0
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        if betainc(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# 学生 t 分布
# --------------------------------------------------------------------------- #
def student_t_pdf(t: Union[float, np.ndarray], df: float) -> Union[float, np.ndarray]:
    """学生 t 密度（纯 numpy，用 lgamma 计算归一化常数）。"""
    df = float(df)
    if df <= 0:
        raise ValueError(f"自由度必须为正，收到 {df}")
    arr = np.asarray(t, dtype="float64")
    log_const = math.lgamma(0.5 * (df + 1.0)) - math.lgamma(0.5 * df) - 0.5 * math.log(df * math.pi)
    out = np.exp(log_const - 0.5 * (df + 1.0) * np.log1p(arr * arr / df))
    return float(out) if out.ndim == 0 else out


def t_cdf(t: float, df: float) -> float:
    """学生 t 的 CDF：P(T ≤ t)。

    用上尾恒等式 ``P(T > t) = 0.5·I_{ν/(ν+t²)}(ν/2, 1/2)``（t>0），t<0 时对称。
    scipy 可用时走 ``stats.t.cdf``，否则用自研 :func:`betainc`。
    """
    t, df = float(t), float(df)
    if df <= 0:
        raise ValueError(f"自由度必须为正，收到 {df}")
    sp = _try_import_scipy()
    if sp is not None:
        return float(sp.stats.t.cdf(t, df))
    if t == 0.0:
        return 0.5
    x = df / (df + t * t)
    upper = 0.5 * betainc(0.5 * df, 0.5, x)
    return 1.0 - upper if t > 0.0 else upper


def t_ppf(p: float, df: float) -> float:
    """学生 t 的分位函数 F⁻¹(p)。

    scipy 可用时走 ``stats.t.ppf``；否则先用倍增法把根括住，再对 :func:`t_cdf`
    做 200 次二分（纯 Python 回退，无需任何特殊函数库）。
    """
    q, df = float(p), float(df)
    if not (0.0 < q < 1.0):
        raise ValueError(f"p 必须落在 (0,1) 内，收到 {q}")
    if df <= 0:
        raise ValueError(f"自由度必须为正，收到 {df}")
    if q == 0.5:
        return 0.0                      # 中位数精确为 0，两条路径口径一致
    sp = _try_import_scipy()
    if sp is not None:
        return float(sp.stats.t.ppf(q, df))
    lo, hi = -1.0, 1.0
    while t_cdf(lo, df) > q and lo > -1e14:
        lo *= 2.0
    while t_cdf(hi, df) < q and hi < 1e14:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# 输入归一化与「同型进出」
# --------------------------------------------------------------------------- #
def clean_1d(x: Any, name: str = "returns") -> np.ndarray:
    """把收益序列归一为一维 float ndarray 并剔除非有限值。"""
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(f"{name} 为多列 DataFrame，请指定单列或改用面板接口")
        arr = x.iloc[:, 0].to_numpy(dtype="float64")
    elif isinstance(x, pd.Series):
        arr = x.to_numpy(dtype="float64")
    else:
        arr = np.asarray(x, dtype="float64").ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError(f"{name} 在剔除非有限值后为空")
    return arr


def as_returns_frame(returns: ReturnsLike, name: str = "returns",
                     dropna: bool = True) -> pd.DataFrame:
    """把收益输入归一成 float 面板 DataFrame(index=期次, columns=资产)。

    Series / 一维 ndarray 会被升为单列面板（列名沿用 Series.name 或 ``"asset"``），
    使上层函数可以用同一套逐列逻辑处理单资产与多资产输入。
    """
    if isinstance(returns, pd.Series):
        df = returns.to_frame(name=returns.name if returns.name is not None else "asset")
    elif isinstance(returns, pd.DataFrame):
        df = returns
    else:
        arr = np.asarray(returns, dtype="float64")
        if arr.ndim == 1:
            df = pd.DataFrame({"asset": arr})
        elif arr.ndim == 2:
            df = pd.DataFrame(arr)
        else:
            raise ValueError(f"{name} 必须是一维序列或二维面板，收到 {arr.ndim} 维")
    df = df.astype("float64")
    if dropna:
        df = df.dropna(how="any").sort_index()
    if df.empty or df.shape[1] == 0:
        raise ValueError(f"{name} 面板在剔除 NaN 后为空")
    return df


def apply_per_asset(returns: ReturnsLike, fn: Callable[..., float],
                    name: Optional[str] = None, **kwargs: Any) -> Union[float, pd.Series]:
    """「同型进出」调度器：Series/ndarray -> 标量；DataFrame -> 按资产索引的 Series。

    ``fn`` 接收**单资产**的一维序列（``pd.Series``）与关键字参数，返回 float。
    """
    if isinstance(returns, pd.DataFrame):
        values = {col: float(fn(returns[col], **kwargs)) for col in returns.columns}
        out = pd.Series(values, name=name)
        return out.reindex(returns.columns)
    return float(fn(returns, **kwargs))


def align_pair(a: Any, b: Any, name_a: str = "returns",
               name_b: str = "benchmark") -> Tuple[np.ndarray, np.ndarray, Optional[pd.Index]]:
    """成对序列对齐：按索引交集（或按位置）剔除任一方的缺失，返回等长数组。

    两个 pandas 对象时按索引 inner join 对齐；只要有一方是纯 ndarray 则按位置
    配对（长度必须一致）。返回 ``(a, b, 输出用索引或 None)``。
    """
    if isinstance(a, (pd.Series, pd.DataFrame)) and isinstance(b, (pd.Series, pd.DataFrame)):
        sa = a.iloc[:, 0] if isinstance(a, pd.DataFrame) else a
        sb = b.iloc[:, 0] if isinstance(b, pd.DataFrame) else b
        joined = pd.concat([sa.rename("a"), sb.rename("b")], axis=1, join="inner").dropna()
        if len(joined) == 0:
            raise ValueError(f"{name_a} 与 {name_b} 没有共同的有效索引")
        return (joined["a"].to_numpy(dtype="float64"),
                joined["b"].to_numpy(dtype="float64"),
                joined.index)
    va = clean_1d(a, name_a)
    vb = clean_1d(b, name_b)
    if va.size != vb.size:
        raise ValueError(f"{name_a} 与 {name_b} 长度不一致：{va.size} vs {vb.size}")
    return va, vb, None


def as_weights(weights: Any, columns: Optional[pd.Index] = None,
               name: str = "weights") -> Tuple[np.ndarray, Optional[pd.Index]]:
    """把组合权重归一为 (一维 ndarray, 输出索引)。

    ``weights`` 为 Series 且给出 ``columns`` 时按资产名 reindex 对齐，
    杜绝资产顺序不同导致的静默错位；list/ndarray 输入按位置解释。
    """
    index: Optional[pd.Index] = None
    if isinstance(weights, pd.Series):
        if columns is not None:
            missing = [c for c in columns if c not in weights.index]
            if missing:
                raise ValueError(f"{name} 缺少资产权重：{missing}")
            series = weights.reindex(columns)
        else:
            series = weights
            columns = weights.index
        arr = series.to_numpy(dtype="float64")
        index = columns
    else:
        arr = np.asarray(weights, dtype="float64").ravel()
        if columns is not None and arr.size != len(columns):
            raise ValueError(f"{name} 维度 {arr.size} 与资产数 {len(columns)} 不一致")
        if arr.ndim != 1:
            raise ValueError(f"{name} 必须是一维权重向量")
        index = columns
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} 含 NaN/Inf")
    if arr.size == 0:
        raise ValueError(f"{name} 为空")
    return arr, index


def wrap_1d(values: np.ndarray, index: Optional[pd.Index] = None,
            name: Optional[str] = None) -> Weights:
    """有资产索引则输出 Series，否则输出 ndarray，保持「同型进出」。"""
    arr = np.asarray(values, dtype="float64")
    if index is not None:
        return pd.Series(arr, index=index, name=name)
    return arr


def as_mapping(x: Union[Mapping[str, float], pd.Series, None],
               name: str = "shocks") -> Dict[str, float]:
    """把「资产 -> 数值」的映射输入归一为 ``Dict[str, float]``（None -> 空字典）。"""
    if x is None:
        return {}
    if isinstance(x, pd.Series):
        return {str(k): float(v) for k, v in x.items() if np.isfinite(v)}
    if isinstance(x, Mapping):
        return {str(k): float(v) for k, v in x.items() if np.isfinite(v)}
    raise TypeError(f"{name} 必须是 Mapping 或 Series，收到 {type(x).__name__}")


def check_confidence(confidence: float) -> float:
    """校验置信度落在 (0,1) 内，返回 float。"""
    c = float(confidence)
    if not (0.0 < c < 1.0):
        raise ValueError(f"confidence 必须落在 (0,1) 内，收到 {c}")
    return c


def check_positive_periods(periods_per_year: float) -> float:
    """校验年化期数为正，返回 float。"""
    ppy = float(periods_per_year)
    if ppy <= 0:
        raise ValueError(f"periods_per_year 必须为正，收到 {ppy}")
    return ppy


def quantile_sorted(arr: np.ndarray, q: float) -> float:
    """线性插值分位数（与 ``np.quantile`` 默认一致），单独抽出便于复用与测试。"""
    return float(np.quantile(arr, float(q)))


def tail_mask(values: np.ndarray, confidence: float) -> np.ndarray:
    """返回「损失尾部」布尔掩码：values ≤ q_{1−c}（至少含一个最差观测）。"""
    c = check_confidence(confidence)
    threshold = quantile_sorted(values, 1.0 - c)
    mask = values <= threshold
    if not mask.any():  # 极端情况（大量并列）保底取最差观测
        mask = values <= values.min()
    return mask


def describe_seq(values: Sequence[float]) -> Dict[str, float]:
    """给一组统计量算 min/分位/mean/max 摘要，供回撤持续期分布等复用。"""
    arr = np.asarray(list(values), dtype="float64")
    if arr.size == 0:
        return {"count": 0.0, "min": float("nan"), "p25": float("nan"), "median": float("nan"),
                "p75": float("nan"), "max": float("nan"), "mean": float("nan")}
    return {
        "count": float(arr.size),
        "min": float(arr.min()),
        "p25": float(np.quantile(arr, 0.25)),
        "median": float(np.median(arr)),
        "p75": float(np.quantile(arr, 0.75)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }
