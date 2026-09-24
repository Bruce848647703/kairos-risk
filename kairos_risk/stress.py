"""情景与压力测试模块：冲击情景损益、历史重放、反向压力测试。

三类互补的压力视角
------------------
1. **假设情景（hypothetical）**：由 :class:`Scenario` 描述一组冲击（市场整体下跌、
   波动放大、个别资产暴跌、多因子联动），用一阶线性近似估算组合损益，
   并可选地按放大后的波动重算参数 VaR；
2. **历史重放（historical replay）**：从历史中挑出组合表现最差的 N 个窗口，
   把当期各资产收益重放到当前权重上，回答「若重演那段时间会亏多少」；
3. **反向压力测试（reverse stress test）**：给定损失目标，反推需要多大的冲击倍数，
   回答「什么样的情景才会把我打穿」——这是监管与风控更关心的提问方向。

约定：损益 ``pnl`` 为**带符号**金额（负数=亏损），``loss`` 为正数损失口径。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ._util import ReturnsLike, as_mapping, as_returns_frame, as_weights, check_confidence, norm_ppf

Weights = Union[np.ndarray, pd.Series]


# --------------------------------------------------------------------------- #
# 情景定义
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Scenario:
    """一个压力情景：由「市场冲击 / 因子冲击 / 个股特质冲击 / 波动放大」组合而成。

    字段
    ----
    name:          情景名称；
    asset_shocks:  个股特质冲击（资产 -> 收益率变动），**加性**叠加在其它冲击之上；
    market_shock:  市场整体冲击幅度（如 −0.20 表示市场下跌 20%），需配合 ``betas``；
    betas:         各资产对市场的 beta，用于把 market_shock 传导到资产；
    factor_shocks: 多因子冲击（因子 -> 收益率变动），需配合 ``exposures``；
    exposures:     资产因子暴露矩阵（index=资产, columns=因子）；
    vol_multiplier: 波动放大倍数（2.0 = 波动翻倍），仅影响 VaR/波动类指标。

    合成规则：shock_i = β_i·market_shock + Σ_k B_ik·f_k + asset_shocks_i。
    """

    name: str = "scenario"
    asset_shocks: Optional[Mapping[str, float]] = None
    market_shock: float = 0.0
    betas: Optional[Mapping[str, float]] = None
    factor_shocks: Optional[Mapping[str, float]] = None
    exposures: Optional[pd.DataFrame] = None
    vol_multiplier: float = 1.0

    # ---- 构造便捷入口 ----
    @classmethod
    def market(cls, betas: Mapping[str, float], shock: float = -0.20,
               name: str = "market_crash", vol_multiplier: float = 1.0) -> "Scenario":
        """市场整体冲击情景（按 beta 传导）。"""
        return cls(name=name, market_shock=float(shock), betas=betas,
                   vol_multiplier=float(vol_multiplier))

    @classmethod
    def idiosyncratic(cls, shocks: Mapping[str, float], name: str = "name_crash",
                      vol_multiplier: float = 1.0) -> "Scenario":
        """个别资产暴跌情景（只冲击指定资产）。"""
        return cls(name=name, asset_shocks=shocks, vol_multiplier=float(vol_multiplier))

    @classmethod
    def vol_spike(cls, multiplier: float = 2.0, name: str = "vol_spike") -> "Scenario":
        """纯波动放大情景（收益冲击为 0，只影响 VaR / 波动）。"""
        if float(multiplier) <= 0:
            raise ValueError("vol_multiplier 必须为正")
        return cls(name=name, vol_multiplier=float(multiplier))

    @classmethod
    def from_history(cls, returns: ReturnsLike, date: object,
                     name: Optional[str] = None) -> "Scenario":
        """把历史某一期的实际资产收益打包成情景（历史情景法）。"""
        frame = as_returns_frame(returns)
        if date not in frame.index:
            raise KeyError(f"日期 {date} 不在收益面板中")
        row = frame.loc[date]
        return cls(name=name or f"history_{date}",
                   asset_shocks={str(c): float(v) for c, v in row.items()})

    # ---- 冲击合成 ----
    def scale(self, multiplier: float) -> "Scenario":
        """按倍数缩放全部冲击（反向压力测试的搜索维度）。

        收益类冲击按 λ 线性缩放；波动倍数按 ``1 + |λ|·(vol_multiplier − 1)`` 缩放，
        保证 λ=1 还原原情景、λ=0 退化为「无冲击、波动不变」，且 λ<0
        （反向情景）时波动放大依然为正、随 |λ| 单调。
        """
        lam = float(multiplier)
        return Scenario(
            name=f"{self.name}x{lam:g}",
            asset_shocks={k: v * lam for k, v in as_mapping(self.asset_shocks).items()},
            market_shock=self.market_shock * lam,
            betas=self.betas,
            factor_shocks={k: v * lam for k, v in as_mapping(self.factor_shocks).items()},
            exposures=self.exposures,
            vol_multiplier=1.0 + abs(lam) * (self.vol_multiplier - 1.0),
        )

    def shocks(self, assets: pd.Index) -> pd.Series:
        """把情景展开成「每资产收益率冲击」的 Series（按给定资产顺序）。"""
        cols = list(assets)
        total = np.zeros(len(cols), dtype="float64")
        pos = {str(c): i for i, c in enumerate(cols)}

        if float(self.market_shock) != 0.0:
            beta_map = as_mapping(self.betas, "betas")
            if not beta_map:
                raise ValueError("market_shock 需要同时提供 betas")
            missing = [c for c in cols if str(c) not in beta_map]
            if missing:
                raise ValueError(f"betas 缺少资产：{missing}")
            for c in cols:
                total[pos[str(c)]] += beta_map[str(c)] * float(self.market_shock)

        f_shocks = as_mapping(self.factor_shocks, "factor_shocks")
        if f_shocks:
            if self.exposures is None:
                raise ValueError("factor_shocks 需要同时提供 exposures")
            expo = self.exposures.reindex(index=cols)
            if expo.isna().any().any():
                raise ValueError("exposures 未覆盖全部资产或因子")
            factors = list(expo.columns)
            missing = [f for f in factors if f not in f_shocks]
            if missing:
                raise ValueError(f"factor_shocks 缺少因子：{missing}")
            fvec = np.array([f_shocks[f] for f in factors], dtype="float64")
            total = total + expo.to_numpy(dtype="float64") @ fvec

        for asset, shock in as_mapping(self.asset_shocks, "asset_shocks").items():
            if asset in pos:
                total[pos[asset]] += shock
            else:
                raise ValueError(f"asset_shocks 含未知资产：{asset}")

        if float(self.vol_multiplier) <= 0:
            raise ValueError("vol_multiplier 必须为正")
        return pd.Series(total, index=assets, name="shock")


# --------------------------------------------------------------------------- #
# 假设情景损益
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class StressResult:
    """情景压力测试结果。

    字段
    ----
    name / scenario:  情景名称与情景对象；
    shocks:           每资产冲击；
    weights:          组合权重；
    pnl_pct:          组合损益率 = Σ_i w_i·shock_i（一阶线性近似）；
    pnl:              组合损益金额 = value × pnl_pct（负数=亏损）；
    contribution:     每资产对损益的贡献 w_i·shock_i（和 = pnl_pct）；
    value_before / value_after: 冲击前后的组合价值；
    vol_baseline / vol_stressed: 冲击前后组合波动（需要 cov，stressed 按 vol_multiplier 放大）；
    var_baseline / var_stressed: 冲击前后的参数 VaR（**损失率**口径的正数，需要 cov；
                        乘以 value 得金额）。
    """

    name: str
    scenario: Scenario
    shocks: pd.Series
    weights: pd.Series
    pnl_pct: float
    pnl: float
    contribution: pd.Series
    value_before: float
    value_after: float
    vol_baseline: Optional[float] = None
    vol_stressed: Optional[float] = None
    var_baseline: Optional[float] = None
    var_stressed: Optional[float] = None


def apply_scenario(weights: Weights, scenario: Scenario, value: float = 1.0,
                   cov: Optional[pd.DataFrame] = None, confidence: float = 0.95) -> StressResult:
    """对组合施加情景冲击，估算损益与（可选的）冲击后参数 VaR。

    参数
    ----
    weights:  组合权重（Series 或数组）。
    scenario: :class:`Scenario`。
    value:    组合市值（用于把损益率换算成金额）。
    cov:      资产协方差；给出时额外计算冲击前后的波动与参数 VaR。
    confidence: 参数 VaR 的置信度。

    损益采用**一阶线性近似** ``Σ w_i·shock_i``（等价于冲击幅度不大时的组合再估值），
    对大幅冲击会略高估多头损失/低估凸性收益，属于压力测试的常规保守口径。
    冲击后 VaR 把均值平移为 pnl_pct、波动按 vol_multiplier 放大：
    ``VaR_stressed = −(pnl_pct + z_{1−c}·σ_stressed)``。
    """
    frame_cols: Optional[pd.Index] = cov.columns if isinstance(cov, pd.DataFrame) else None
    w_arr, index = as_weights(weights, frame_cols, name="weights")
    if index is None:
        if scenario.betas is not None or scenario.asset_shocks is not None:
            index = pd.Index(sorted(as_mapping(scenario.betas).keys()
                                    or as_mapping(scenario.asset_shocks).keys()))
        else:
            index = pd.RangeIndex(w_arr.size)
        if len(index) != w_arr.size:
            raise ValueError("weights 为纯数组且无法从情景推断资产名，请传入 Series 权重")
    shocks = scenario.shocks(index)
    w = pd.Series(w_arr, index=index, name="weight")
    contrib = w * shocks
    pnl_pct = float(contrib.sum())
    v0 = float(value)
    if v0 <= 0:
        raise ValueError("value 必须为正")

    vol_base = vol_st = var_base = var_st = None
    if cov is not None:
        sigma = cov.reindex(index=index, columns=index).to_numpy(dtype="float64")
        if np.any(~np.isfinite(sigma)):
            raise ValueError("cov 未覆盖全部资产或含 NaN")
        sigma = (sigma + sigma.T) / 2.0
        var = max(float(w_arr @ sigma @ w_arr), 0.0)
        vol_base = math.sqrt(var)
        vol_st = vol_base * float(scenario.vol_multiplier)
        z = norm_ppf(1.0 - check_confidence(confidence))
        var_base = -(z * vol_base)
        var_st = -(pnl_pct + z * vol_st)

    return StressResult(
        name=scenario.name, scenario=scenario, shocks=shocks, weights=w,
        pnl_pct=pnl_pct, pnl=v0 * pnl_pct, contribution=contrib,
        value_before=v0, value_after=v0 * (1.0 + pnl_pct),
        vol_baseline=vol_base, vol_stressed=vol_st,
        var_baseline=var_base, var_stressed=var_st,
    )


def scenario_table(weights: Weights, scenarios: Sequence[Scenario], value: float = 1.0,
                   cov: Optional[pd.DataFrame] = None,
                   confidence: float = 0.95) -> pd.DataFrame:
    """批量情景对比表：index=情景名，columns=损益率/损益/冲击后 VaR 等。"""
    if len(scenarios) == 0:
        raise ValueError("scenarios 不能为空")
    rows: Dict[str, Dict[str, float]] = {}
    for sc in scenarios:
        res = apply_scenario(weights, sc, value=value, cov=cov, confidence=confidence)
        rows[sc.name] = {
            "pnl_pct": res.pnl_pct,
            "pnl": res.pnl,
            "value_after": res.value_after,
            "worst_asset_shock": float(res.shocks.min()) if len(res.shocks) else 0.0,
            "vol_stressed": float("nan") if res.vol_stressed is None else res.vol_stressed,
            "var_stressed": float("nan") if res.var_stressed is None else res.var_stressed,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


# --------------------------------------------------------------------------- #
# 历史重放
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class ReplayResult:
    """历史重放结果。

    字段
    ----
    frame:      明细表（index=窗口结束期，columns=各资产贡献 + portfolio_return + pnl）；
    n_windows:  可用的历史窗口总数；
    window:     窗口长度（期数）；
    worst_pnl:  最差窗口的损益金额（负数）；
    total_pnl:  所选窗口的损益合计；
    mean_pnl:   所选窗口的平均损益；
    hit_dates:  所选窗口的结束期标签（按损失从大到小排序）。
    """

    frame: pd.DataFrame
    n_windows: int
    window: int
    worst_pnl: float
    total_pnl: float
    mean_pnl: float
    hit_dates: List[object]


def historical_replay(returns: ReturnsLike, weights: Weights, n_worst: int = 10,
                      window: int = 1, value: float = 1.0,
                      non_overlapping: bool = False) -> ReplayResult:
    """找出历史上组合表现最差的 N 个窗口，把当期资产收益重放到当前权重上。

    参数
    ----
    n_worst:  选取的最差窗口数。
    window:   窗口长度（期数）；>1 时按复利累计窗口收益。
    value:    组合市值。
    non_overlapping: True 时贪心跳过与已选窗口重叠的窗口（避免同一段行情被重复计入）。

    明细表中 ``contribution_i = w_i·R_i(window)``；window=1 时各贡献之和精确等于
    组合收益，window>1 时组合列用**组合自身复利**计算，与逐资产贡献之和存在
    二阶差异（隐含期内再平衡假设）。
    """
    if int(n_worst) <= 0:
        raise ValueError("n_worst 必须为正")
    win = int(window)
    if win <= 0:
        raise ValueError("window 必须为正")
    frame = as_returns_frame(returns)
    w_arr, _ = as_weights(weights, frame.columns, name="weights")
    w = pd.Series(w_arr, index=frame.columns, name="weight")
    if float(value) <= 0:
        raise ValueError("value 必须为正")

    if win == 1:
        multi = frame
    else:
        if len(frame) < win:
            raise ValueError(f"样本期数 {len(frame)} 小于窗口长度 {win}")
        multi = (1.0 + frame).rolling(win).apply(np.prod, raw=True) - 1.0
        multi = multi.dropna(how="any")
    if len(multi) == 0:
        raise ValueError("历史窗口在剔除 NaN 后为空")

    port = multi.to_numpy(dtype="float64") @ w_arr
    port = pd.Series(port, index=multi.index, name="portfolio_return")
    order = np.argsort(port.to_numpy(), kind="stable")     # 损失从大到小
    chosen: List[int] = []
    for pos in order:
        if non_overlapping and any(abs(int(pos) - int(c)) < win for c in chosen):
            continue
        chosen.append(int(pos))
        if len(chosen) >= int(n_worst):
            break
    if not chosen:
        raise ValueError("没有可用的历史窗口")

    sub = multi.iloc[chosen]
    detail = sub.mul(w_arr, axis=1)
    detail.columns = [f"contrib_{c}" for c in sub.columns]
    detail["portfolio_return"] = port.iloc[chosen].to_numpy()
    detail["pnl"] = detail["portfolio_return"].to_numpy() * float(value)
    detail.index.name = "window_end"
    pnls = detail["pnl"].to_numpy(dtype="float64")
    return ReplayResult(
        frame=detail,
        n_windows=int(len(multi)),
        window=win,
        worst_pnl=float(pnls.min()),
        total_pnl=float(pnls.sum()),
        mean_pnl=float(pnls.mean()),
        hit_dates=list(detail.index),
    )


def worst_windows(returns: ReturnsLike, weights: Weights, n_worst: int = 10,
                  window: int = 1) -> pd.DataFrame:
    """便捷入口：直接返回最差窗口的明细表（:func:`historical_replay` 的 frame）。"""
    return historical_replay(returns, weights, n_worst=n_worst, window=window).frame


# --------------------------------------------------------------------------- #
# 反向压力测试
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class ReverseStressResult:
    """反向压力测试结果：达成目标损失所需的冲击倍数与对应情景。

    字段
    ----
    target_loss:  目标损失（正数）；
    achieved_loss: 实际达到的损失（正数，与目标之差 ≤ tol）；
    multiplier:   所需的冲击倍数 λ（λ<0 表示需要**反转**情景方向）；
    scenario:     λ 缩放后的情景对象；
    shocks:       对应情景下的每资产冲击；
    metric:       损失度量口径（``"pnl"`` 线性损益 或 ``"var"`` 冲击后参数 VaR）；
    converged / iterations: 是否收敛与二分迭代次数。
    """

    target_loss: float
    achieved_loss: float
    multiplier: float
    scenario: Scenario
    shocks: pd.Series
    metric: str
    converged: bool
    iterations: int


def reverse_stress_test(weights: Weights, scenario: Scenario, target_loss: float,
                        value: float = 1.0, cov: Optional[pd.DataFrame] = None,
                        confidence: float = 0.95, metric: str = "pnl",
                        max_multiplier: float = 1e4, tol: float = 1e-10,
                        max_iter: int = 200) -> ReverseStressResult:
    """反向压力测试：给定损失目标，二分搜索所需的冲击倍数 λ。

    把情景整体缩放 λ 倍（:meth:`Scenario.scale`），求解 ``loss(λ) = target_loss``：

    - ``metric="pnl"``：loss(λ) = −value·Σ w_i·(λ·shock_i)，对 λ 线性；
    - ``metric="var"``：loss(λ) = value × 冲击后参数 VaR（需要 ``cov``），
      含波动放大项，对 λ 单调但非线性，故统一用二分法求解。

    先在 λ>0 方向倍增扩界；若该方向是盈利（无法达成损失目标），自动改在 λ<0
    方向搜索（等价于把情景反向放大）。
    """
    tl = float(target_loss)
    if tl <= 0:
        raise ValueError("target_loss 必须为正数损失")
    if metric not in ("pnl", "var"):
        raise ValueError("metric 只支持 'pnl' 或 'var'")
    if metric == "var" and cov is None:
        raise ValueError("metric='var' 需要提供 cov")

    def loss_of(lam: float) -> float:
        res = apply_scenario(weights, scenario.scale(lam), value=value, cov=cov,
                             confidence=confidence)
        if metric == "pnl":
            return -res.pnl
        # var_stressed 为损失率口径，乘以 value 换算成与 target_loss 同单位的金额
        return float(res.var_stressed if res.var_stressed is not None else 0.0) * float(value)

    def _bisect(lo: float, hi: float) -> ReverseStressResult:
        # lo / hi 只需满足 loss(lo) < tl ≤ loss(hi)，二分对两个方向都成立
        iters = 0
        if loss_of(lo) >= tl:              # λ=lo 已达成目标（如 var 口径的基线 VaR）
            lam = lo
        else:
            for _ in range(int(max_iter)):
                iters += 1
                mid = 0.5 * (lo + hi)
                if loss_of(mid) < tl:
                    lo = mid
                else:
                    hi = mid
                if abs(hi - lo) <= tol * max(1.0, abs(hi)):
                    break
            lam = 0.5 * (lo + hi)
        achieved = loss_of(lam)
        sc = scenario.scale(lam)
        _, index = as_weights(weights, None, name="weights")
        try:
            shocks = sc.shocks(index if index is not None else pd.Index(
                sorted(as_mapping(sc.betas).keys() or as_mapping(sc.asset_shocks).keys())))
        except ValueError:
            shocks = pd.Series(dtype="float64")
        return ReverseStressResult(
            target_loss=tl, achieved_loss=achieved, multiplier=lam, scenario=sc,
            shocks=shocks, metric=metric, converged=abs(achieved - tl) <= max(tol, 1e-8 * tl),
            iterations=iters,
        )

    # 正向扩界
    hi = 1.0
    while loss_of(hi) < tl and hi < float(max_multiplier):
        hi *= 2.0
    if loss_of(hi) >= tl:
        return _bisect(0.0, hi)
    # 正向是盈利方向 -> 反向搜索
    lo = -1.0
    while loss_of(lo) < tl and lo > -float(max_multiplier):
        lo *= 2.0
    if loss_of(lo) < tl:
        raise ValueError(
            f"在 |λ| ≤ {max_multiplier:g} 内无法达成目标损失 {tl:g}；"
            "请检查情景方向或放宽 max_multiplier"
        )
    return _bisect(0.0, lo)


__all__ = [
    "Scenario", "StressResult", "apply_scenario", "scenario_table",
    "ReplayResult", "historical_replay", "worst_windows",
    "ReverseStressResult", "reverse_stress_test",
]
