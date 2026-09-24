"""极值理论（EVT）尾部风险模块：Hill 尾部指数、POT/GPD 尾部 VaR-ES、尾部比率。

为什么需要 EVT
--------------
历史 VaR 只能回答「样本里发生过多大的损失」，而风控真正关心的是
「样本外可能有多大」。极值理论用尾部渐近分布外推：

- **Hill 估计**：对上尾的顺序统计量做对数间距平均，估计尾部指数 ξ
  （ξ>0 为肥尾/帕累托型，ξ=0 为指数型，ξ<0 为有界尾）；
- **POT（Peaks Over Threshold）**：把超过阈值 u 的超出量拟合广义帕累托分布 GPD(ξ, β)，
  再结合超出概率 ζ = N_u/N 外推任意高置信度的 VaR 与 ES（McNeil–Frey 公式）。

scipy 可选
----------
GPD 拟合默认走 scipy 的极大似然；scipy 缺失时自动回退到**自研概率加权矩（PWM/Hosking）
估计**（另有经典矩估计可选），两条路径都是纯 numpy 实现，结果在同一量级上互相印证。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ._util import (
    ReturnsLike,
    _try_import_scipy,
    check_confidence,
    clean_1d,
    quantile_sorted,
)

_EPS_XI = 1e-8


def _losses(returns: ReturnsLike, side: str = "loss") -> np.ndarray:
    """把收益转成「待研究的上尾变量」：side="loss" 取 −r（亏损为正），"gain" 取 r。"""
    if side not in ("loss", "gain"):
        raise ValueError("side 只支持 'loss' 或 'gain'")
    v = clean_1d(returns, "returns")
    return -v if side == "loss" else v


# --------------------------------------------------------------------------- #
# Hill 估计
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class HillResult:
    """Hill 尾部指数估计结果。

    字段
    ----
    xi:        尾部指数 ξ̂（>0 肥尾，≈0 指数尾，<0 短尾有界）；
    se:        渐近标准误 ξ̂/√k；
    k:         参与估计的上尾顺序统计量个数；
    threshold: 阈值 u = 第 (k+1) 大的观测；
    n_exceed:  超过阈值的观测数；
    n_obs:     总观测数；
    tail_index: 尾部分布指数 α = 1/ξ（帕累托口径，ξ≤0 时为 inf）。
    """

    xi: float
    se: float
    k: int
    threshold: float
    n_exceed: int
    n_obs: int

    @property
    def tail_index(self) -> float:
        """帕累托尾指数 α = 1/ξ；ξ ≤ 0 时视为无穷（尾部不比指数更厚）。"""
        return float("inf") if self.xi <= 0 else 1.0 / self.xi


def hill_estimator(returns: ReturnsLike, k: Optional[int] = None,
                   tail_quantile: float = 0.9, side: str = "loss") -> HillResult:
    """Hill 尾部指数估计：ξ̂ = (1/k)·Σ_{i=1..k} ln(X_(i)/X_(k+1))。

    参数
    ----
    returns:       收益序列（默认研究**亏损**上尾，即 −r 的右尾）。
    k:             使用的上尾顺序统计量个数；None 时按 ``tail_quantile`` 推出
                   k = round(n·(1−tail_quantile))，并夹在 [5, n−2] 之间。
    tail_quantile: 仅在 k=None 时生效的尾部分位（0.9 表示研究最大的 10%）。
    side:          ``"loss"``（左尾风险，默认）或 ``"gain"``（右尾）。

    k 的选择是经典权衡：k 太小方差大，k 太大偏差大（超出渐近区间），
    实践中配合 :func:`hill_path` 观察 ξ̂ 随 k 的稳定平台。
    """
    x = _losses(returns, side)
    n = x.size
    if n < 8:
        raise ValueError(f"Hill 估计至少需要 8 个观测，收到 {n}")
    if k is None:
        q = float(tail_quantile)
        if not (0.0 < q < 1.0):
            raise ValueError(f"tail_quantile 必须落在 (0,1) 内，收到 {q}")
        k = int(round(n * (1.0 - q)))
    k = int(k)
    if k < 2 or k > n - 2:
        raise ValueError(f"k 必须落在 [2, n−2] = [2, {n - 2}] 内，收到 {k}")

    desc = np.sort(x)[::-1]                     # 降序：X_(1) ≥ X_(2) ≥ ...
    threshold = float(desc[k])                  # X_(k+1)
    if threshold <= 0:
        raise ValueError(
            f"尾部阈值 X_(k+1)={threshold:.6g} 非正，对数间距无定义；"
            "请减小 k（提高 tail_quantile）或先对数据做位置平移"
        )
    top = desc[:k]
    if np.any(top <= 0):
        raise ValueError("上尾观测出现非正值，无法做 Hill 估计")
    xi = float(np.mean(np.log(top / threshold)))
    return HillResult(xi=xi, se=abs(xi) / math.sqrt(k), k=k, threshold=threshold,
                      n_exceed=int(np.sum(x > threshold)), n_obs=n)


def hill_path(returns: ReturnsLike, k_min: int = 10, k_max: Optional[int] = None,
              n_points: int = 25, side: str = "loss") -> pd.DataFrame:
    """在一段 k 网格上重复 Hill 估计，返回 k / threshold / xi / se 表（用于选 k）。

    ξ̂ 在「渐近区间」内应大致平稳；显著的上翘通常意味着 k 已进入偏差区。
    """
    x = _losses(returns, side)
    n = x.size
    k_lo = max(2, int(k_min))
    k_hi = int(k_max) if k_max is not None else max(k_lo + 1, n // 4)
    k_hi = min(k_hi, n - 2)
    if k_hi <= k_lo:
        raise ValueError("k 网格为空，请增大样本或调整 k_min/k_max")
    ks = np.unique(np.linspace(k_lo, k_hi, min(int(n_points), k_hi - k_lo + 1)).astype(int))
    rows = []
    for k in ks:
        try:
            # x 已经是「亏损」变量，故内部统一按 side="gain" 处理，避免二次取负
            res = hill_estimator(x, k=int(k), side="gain")
        except ValueError:
            continue
        rows.append({"k": res.k, "threshold": res.threshold, "xi": res.xi, "se": res.se})
    if not rows:
        raise ValueError("k 网格上全部 Hill 估计失败（阈值可能非正）")
    return pd.DataFrame(rows).set_index("k")


# --------------------------------------------------------------------------- #
# POT / GPD
# --------------------------------------------------------------------------- #
def exceedances(returns: ReturnsLike, threshold: Optional[float] = None,
                tail_quantile: float = 0.95, side: str = "loss") -> Tuple[np.ndarray, float]:
    """取出超过阈值的超出量 y = X − u（X 为亏损或收益，取决于 side）。

    ``threshold=None`` 时按 ``tail_quantile`` 取经验分位阈值。返回 (超出量, 阈值)。
    """
    x = _losses(returns, side)
    u = float(threshold) if threshold is not None else quantile_sorted(x, float(tail_quantile))
    y = x[x > u] - u
    if y.size < 5:
        raise ValueError(f"超过阈值的观测仅 {y.size} 个（至少需要 5 个），请降低阈值")
    return y, u


def gpd_pwm(y: np.ndarray) -> Tuple[float, float]:
    """GPD 的概率加权矩（PWM / Hosking）估计，纯 numpy 自研实现。

    对升序样本 y_(1) ≤ ... ≤ y_(n) 构造两个概率加权矩

        a0 = mean(y) = E[Y],
        a1 = (1/n)·Σ_{i=1..n} ((i−1)/(n−1))·y_(i) ≈ E[Y·F(Y)],

    GPD 的理论值为 α_0 = β/(1−ξ)、α_1 = (β/ξ)[1/((1−ξ)(2−ξ)) − 1/2]
    （由 α_r = ∫_0^1 u^r·Q(u)du 与 Q(u) = β((1−u)^(−ξ)−1)/ξ 直接积分得到）。
    令 r = a1/a0，消去 β 后 r = (3−ξ)/(2(2−ξ))，反解得

        ξ̂ = (3a0 − 4a1)/(a0 − 2a1),   β̂ = a0·(1 − ξ̂)

    无需迭代、无初值敏感问题，是 scipy 缺失时的稳健回退（精度与 MLE 同量级）。
    """
    ys = np.sort(np.asarray(y, dtype="float64"))
    n = ys.size
    if n < 2:
        raise ValueError("PWM 估计至少需要 2 个超出量")
    if np.any(ys <= 0):
        raise ValueError("超出量必须全为正")
    a0 = float(ys.mean())
    ranks = np.arange(n, dtype="float64") / (n - 1.0)
    a1 = float(np.mean(ranks * ys))
    denom = a0 - 2.0 * a1
    if abs(denom) <= 1e-15 * max(abs(a0), 1e-300):
        raise ValueError("PWM 估计退化（a0 ≈ 2a1，ξ 无定义）")
    xi = (3.0 * a0 - 4.0 * a1) / denom
    beta = a0 * (1.0 - xi)
    if beta <= 0:
        raise ValueError(f"PWM 得到非正尺度 β={beta:.6g}，样本可能不适合 GPD")
    return float(xi), float(beta)


def gpd_moments(y: np.ndarray) -> Tuple[float, float]:
    """GPD 的经典矩估计（前两阶矩），纯 numpy 实现。

    理论矩 m1 = β/(1−ξ)、m2 = 2β²/((1−2ξ)(1−ξ))（要求 ξ < 1/2 才有有限二阶矩），
    令 R = m2/(2m1²) = (1−ξ)/(1−2ξ)，反解得

        ξ̂ = (1 − R)/(1 − 2R),   β̂ = m1·(1 − ξ̂)

    因 ξ̂ 被天然限制在 (−∞, 1/2)，对极厚尾样本会低估 ξ，故仅作 PWM 的对照。
    """
    v = np.asarray(y, dtype="float64")
    if v.size < 2:
        raise ValueError("矩估计至少需要 2 个超出量")
    if np.any(v <= 0):
        raise ValueError("超出量必须全为正")
    m1 = float(v.mean())
    m2 = float(np.mean(v * v))
    if m1 <= 0:
        raise ValueError("矩估计要求正的均值超出量")
    r = m2 / (2.0 * m1 * m1)
    denom = 1.0 - 2.0 * r
    if abs(denom) <= 1e-15:
        raise ValueError("矩估计退化（R ≈ 1/2，ξ 无定义）")
    xi = (1.0 - r) / denom
    beta = m1 * (1.0 - xi)
    if beta <= 0:
        raise ValueError(f"矩估计得到非正尺度 β={beta:.6g}，样本可能不适合 GPD")
    return float(xi), float(beta)


def gpd_mle(y: np.ndarray) -> Tuple[float, float]:
    """GPD 极大似然拟合（**需要 scipy**，位置参数固定为 0）。"""
    sp = _try_import_scipy()
    if sp is None:
        raise ImportError(
            "GPD 极大似然需要 scipy，请安装：pip install scipy（或 pip install kairos-risk[scipy]）；"
            "也可改用 method='pwm' / 'moments' 的纯 numpy 估计。"
        )
    v = np.asarray(y, dtype="float64")
    try:
        c, loc, scale = sp.stats.genpareto.fit(v, floc=0.0)
    except Exception as exc:  # pragma: no cover - scipy 版本差异
        raise RuntimeError(f"scipy GPD 拟合失败：{exc}") from exc
    if not np.isfinite([c, scale]).all() or scale <= 0:
        raise RuntimeError("scipy GPD 拟合返回非有限或非正尺度")
    return float(c), float(scale)


@dataclass(frozen=True, eq=False)
class GpdFit:
    """POT/GPD 拟合结果，可直接外推尾部 VaR 与 ES。

    字段
    ----
    xi / scale:        GPD 形状参数 ξ 与尺度参数 β；
    threshold:         阈值 u；
    n_exceed / n_obs:  超出量个数 Nu 与总观测数 N；
    exceedance_prob:   ζ = Nu/N（超出阈值的概率）；
    method:            拟合方法（``"scipy_mle"`` / ``"pwm"`` / ``"moments"``）；
    side:              研究的是亏损尾（``"loss"``）还是收益尾（``"gain"``）。
    """

    xi: float
    scale: float
    threshold: float
    n_exceed: int
    n_obs: int
    exceedance_prob: float
    method: str
    side: str = "loss"

    def survival(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """无条件生存函数 P(X > x) = ζ·(1 + ξ(x−u)/β)^(−1/ξ)（x ≥ u；ξ→0 取指数极限）。"""
        arr = np.asarray(x, dtype="float64")
        z = np.maximum(1.0 + self.xi * (arr - self.threshold) / self.scale, 1e-300)
        # 阈值之外（arr < threshold）的幂运算可能上溢，随后被 where 掩成 NaN，
        # 这里显式忽略该 harmless 的 overflow 告警
        with np.errstate(over="ignore", invalid="ignore"):
            if abs(self.xi) < _EPS_XI:
                tail = np.exp(-(arr - self.threshold) / self.scale)
            else:
                tail = z ** (-1.0 / self.xi)
            out = self.exceedance_prob * tail
        out = np.where(arr >= self.threshold, out, np.nan)
        return float(out) if out.ndim == 0 else out

    def mean_excess(self, level: Optional[float] = None) -> float:
        """平均超出函数 E[X − x | X > x] = (β + ξ(x−u))/(1−ξ)；ξ ≥ 1 时均值无穷。"""
        if self.xi >= 1.0:
            return float("inf")
        x = self.threshold if level is None else float(level)
        return (self.scale + self.xi * (x - self.threshold)) / (1.0 - self.xi)


def fit_gpd(returns: ReturnsLike = None, exceedances_: Optional[Sequence[float]] = None,
            threshold: Optional[float] = None, tail_quantile: float = 0.95,
            method: str = "auto", side: str = "loss") -> GpdFit:
    """用 POT 法拟合 GPD。

    参数
    ----
    returns:      收益序列（默认研究亏损尾）。
    exceedances_: 也可直接给「已减阈值的超出量」序列（此时 ``threshold`` 必填）。
    threshold:    阈值 u；None 时按 ``tail_quantile`` 取经验分位。
    method:       ``"auto"``（有 scipy 用 MLE，否则 PWM）、``"mle"``、``"pwm"``、``"moments"``。
                  MLE 数值失败时自动回退 PWM，保证不中断。
    """
    if method not in ("auto", "mle", "pwm", "moments"):
        raise ValueError("method 只支持 'auto' / 'mle' / 'pwm' / 'moments'")
    if exceedances_ is not None:
        y = np.asarray(list(exceedances_), dtype="float64")
        y = y[np.isfinite(y)]
        if threshold is None:
            raise ValueError("直接给超出量时必须提供 threshold")
        u = float(threshold)
        n_obs = int(y.size)
        n_exceed = int(y.size)
        zeta = 1.0
        if np.any(y <= 0):
            raise ValueError("超出量必须全为正")
    else:
        if returns is None:
            raise ValueError("必须提供 returns 或 exceedances_")
        x = _losses(returns, side)
        y, u = exceedances(x, threshold=threshold, tail_quantile=tail_quantile, side="gain")
        n_obs = int(x.size)
        n_exceed = int(y.size)
        zeta = n_exceed / n_obs if n_obs else float("nan")

    if method == "auto":
        sp = _try_import_scipy()
        use = "mle" if sp is not None else "pwm"
    else:
        use = method
    if use == "mle":
        try:
            xi, scale = gpd_mle(y)
            used = "scipy_mle"
        except (ImportError, RuntimeError):
            if method == "mle":
                raise
            xi, scale = gpd_pwm(y)
            used = "pwm"
    elif use == "pwm":
        xi, scale = gpd_pwm(y)
        used = "pwm"
    else:
        xi, scale = gpd_moments(y)
        used = "moments"

    return GpdFit(xi=xi, scale=scale, threshold=u, n_exceed=n_exceed, n_obs=n_obs,
                  exceedance_prob=zeta, method=used, side=side)


def pot_var(fit: GpdFit, confidence: float = 0.99) -> float:
    """POT/GPD 尾部 VaR（正数损失口径，与 :func:`kairos_risk.measures.historical_var` 同单位）。

    McNeil–Frey 外推：q_c = u + (β/ξ)·[((1−c)/ζ)^(−ξ) − 1]，ξ→0 时取对数极限
    q_c = u + β·ln(ζ/(1−c))。
    """
    c = check_confidence(confidence)
    zeta = fit.exceedance_prob
    if not (0.0 < zeta <= 1.0):
        raise ValueError(f"超出概率 ζ={zeta} 非法")
    p = 1.0 - c
    if p >= zeta:
        raise ValueError(
            f"置信度 {c} 对应的尾概率 {p} 不小于超出概率 ζ={zeta:.4g}，"
            "已落在 GPD 拟合区间之外，请降低阈值或降低置信度"
        )
    ratio = p / zeta
    if abs(fit.xi) < _EPS_XI:
        return float(fit.threshold + fit.scale * math.log(1.0 / ratio))
    return float(fit.threshold + fit.scale / fit.xi * (ratio ** (-fit.xi) - 1.0))


def pot_es(fit: GpdFit, confidence: float = 0.99) -> float:
    """POT/GPD 尾部 ES（正数损失）：ES_c = (q_c + β − ξ·u)/(1 − ξ)，要求 ξ < 1。

    ξ ≥ 1 时 GPD 均值发散，返回 ``inf``（提示尾部厚到期望损失不存在）。
    """
    c = check_confidence(confidence)
    if fit.xi >= 1.0:
        return float("inf")
    q = pot_var(fit, c)
    return float((q + fit.scale - fit.xi * fit.threshold) / (1.0 - fit.xi))


def tail_ratio(returns: ReturnsLike, confidence: float = 0.95) -> float:
    """尾部比率 = 历史 ES / 历史 VaR（同置信度，均为正数损失）。

    正态分布下该比率在 c=0.95 时约为 1.25（c=0.99 时约 1.15）；数值越大说明
    尾部越厚，即「越过 VaR 之后损失还会继续恶化」的程度越高。
    """
    c = check_confidence(confidence)
    v = _losses(returns, "loss")
    var = quantile_sorted(v, c)
    if var <= 0:
        raise ValueError(f"历史 VaR={var:.6g} 非正，尾部比率无意义")
    tail = v[v >= var]
    if tail.size == 0:
        tail = v[v >= v.max()]
    return float(tail.mean()) / var


def tail_summary(returns: ReturnsLike, confidence: float = 0.99,
                 tail_quantile: float = 0.95, method: str = "auto",
                 hill_k: Optional[int] = None) -> pd.DataFrame:
    """尾部风险一页纸摘要：经验 VaR/ES、Hill ξ、GPD(ξ, β)、POT VaR/ES、尾部比率。

    返回单列 ``value`` 的 DataFrame（index=指标名），``attrs["method"]`` 记录
    实际使用的 GPD 拟合方法（scipy_mle / pwm / moments）。
    """
    c = check_confidence(confidence)
    loss = _losses(returns, "loss")
    hill = hill_estimator(returns, k=hill_k, side="loss")
    fit = fit_gpd(returns, tail_quantile=tail_quantile, method=method, side="loss")
    emp_var = quantile_sorted(loss, c)
    emp_tail = loss[loss >= emp_var]
    emp_es = float(emp_tail.mean()) if emp_tail.size else float("nan")
    values = {
        "empirical_var": emp_var,
        "empirical_es": emp_es,
        "empirical_tail_ratio": tail_ratio(returns, c),
        "hill_xi": hill.xi,
        "hill_se": hill.se,
        "hill_k": float(hill.k),
        "gpd_xi": fit.xi,
        "gpd_scale": fit.scale,
        "gpd_threshold": fit.threshold,
        "gpd_exceedance_prob": fit.exceedance_prob,
        "pot_var": pot_var(fit, c),
        "pot_es": pot_es(fit, c),
    }
    out = pd.DataFrame({"value": pd.Series(values, dtype="float64")})
    out.attrs["method"] = fit.method
    return out


__all__ = [
    "HillResult", "hill_estimator", "hill_path", "exceedances", "gpd_pwm",
    "gpd_moments", "gpd_mle", "GpdFit", "fit_gpd", "pot_var", "pot_es",
    "tail_ratio", "tail_summary",
]
