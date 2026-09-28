"""真实 A 股行情上的**组合风险报告**：度量 → 成分分解 → 因子风险 → 回撤 → 压测 → EVT → PSR/DSR。

运行
----
    python examples/real_risk_report.py --data-dir /path/to/kairos-data/data/ashare

默认从同系列 ``kairos-data`` 仓库读取已提交的 38 只 A 股前复权日线（**离线、只读**），
构造一个静态组合（默认**逆波动率权重**），复用 ``kairos_risk`` 的公开 API 产出：

- ``research/real_risk/REPORT.md``          中文报告（分节展示真实数字 + 解读）
- ``research/real_risk/risk_metrics.json``  全部指标（严格合法 JSON，NaN/Inf -> null）
- ``research/real_risk/drawdown.csv``       月末水下曲线（小体积，可直接画图）

数据口径提醒：该数据集的前复权为**等差（减法）口径** ``P_adj = P_raw − D_t``，直接
``pct_change`` 会把高分红标的的收益放大 k=P_raw/P_adj 倍（极端者年化波动达 8000%）。
本脚本先用 :func:`kairos_risk.realdata.repair_close_panel` 做「涨跌停硬约束下的下界修复」，
再剔除复权价被压到近零、无法修复的标的，全过程与剔除清单都写进报告，绝不静默降级。
数据仅用于研究与方法演示，不构成投资建议。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import kairos_risk as kr
from kairos_risk import realdata as rd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO_ROOT, "research", "real_risk")
WEIGHT_SCHEMES = ("inverse_vol", "equal")
MARKET_FACTOR_NAME = "MKT_EW"


# --------------------------------------------------------------------------- #
# 通用小工具：数值格式化、markdown 表、JSON 安全化
# --------------------------------------------------------------------------- #
def have_scipy() -> bool:
    """探测 scipy 是否可用（GPD 极大似然与特殊函数的增强路径）。"""
    try:
        import scipy  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def finite(value: Any) -> Optional[float]:
    """把数值转成 JSON 友好的 float；NaN/±Inf 一律变 ``None``（JSON 无这些字面量）。"""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def json_safe(obj: Any) -> Any:
    """递归把 dict/list/numpy/pandas 标量转成**严格合法**的 JSON 结构。"""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, dt.date, dt.datetime)):
        return str(pd.Timestamp(obj).date())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return finite(obj)
    if isinstance(obj, pd.Series):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, pd.DataFrame):
        return {str(c): json_safe(obj[c]) for c in obj.columns}
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


def pct(value: Any, nd: int = 2) -> str:
    """格式化成百分数字符串（None/NaN -> ``n/a``）。"""
    x = finite(value)
    return "n/a" if x is None else f"{x * 100:.{nd}f}%"


def num(value: Any, nd: int = 4) -> str:
    """格式化成定点数字符串（None/NaN -> ``n/a``）。"""
    x = finite(value)
    return "n/a" if x is None else f"{x:.{nd}f}"


def money(value: Any) -> str:
    """格式化成「万元」金额字符串（负数保留符号，便于阅读损益）。"""
    x = finite(value)
    return "n/a" if x is None else f"{x / 1e4:,.1f} 万元"


def md_table(df: pd.DataFrame, nd: int = 4, index_name: str = "") -> str:
    """把 DataFrame 渲染成 GitHub 风味 markdown 表。

    数值列统一保留 ``nd`` 位小数；已是字符串的列（如预先格式化好的金额）原样输出。
    """
    frame = df.copy()
    cells: Dict[str, List[str]] = {}
    for col in frame.columns:
        s = frame[col]
        if pd.api.types.is_numeric_dtype(s):
            cells[str(col)] = ["n/a" if finite(v) is None else f"{float(v):.{nd}f}" for v in s]
        else:
            cells[str(col)] = ["" if v is None else str(v) for v in s]
    cols = list(cells)
    idx = [("" if i is None else str(i)) for i in frame.index]
    lines = [f"| {index_name or (frame.index.name or '')} | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    lines += [f"| {idx[k]} | " + " | ".join(cells[c][k] for c in cols) + " |"
              for k in range(len(idx))]
    return "\n".join(lines)


def dstr(label: Any) -> str:
    """把日期标签渲染成 ``YYYY-MM-DD``（``None``/NaN/NaT -> ``未收复``）。"""
    if label is None or label is pd.NaT:
        return "未收复"
    if isinstance(label, float) and math.isnan(label):
        return "未收复"
    ts = pd.Timestamp(label)
    return "未收复" if ts is pd.NaT or pd.isna(ts) else str(ts.date())


def series_map(s: pd.Series, names: Optional[List[str]] = None) -> Dict[str, Any]:
    """把 Series 转成 ``{资产名: 有限数值}`` 字典（可 JSON 化）。"""
    idx = names if names is not None else [str(i) for i in s.index]
    return {str(k): finite(v) for k, v in zip(idx, np.asarray(s, dtype="float64"))}


def frame_records(df: pd.DataFrame, nd: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    """把 DataFrame 转成 ``{行标签: {列名: 值}}`` 嵌套字典（可 JSON 化）。"""
    out: Dict[str, Dict[str, Any]] = {}
    for label, row in df.iterrows():
        rec: Dict[str, Any] = {}
        for col in df.columns:
            v = row[col]
            rec[str(col)] = finite(v) if pd.api.types.is_numeric_dtype(df[col]) else str(v)
        out[str(label)] = rec
    return out


# --------------------------------------------------------------------------- #
# 组合构造
# --------------------------------------------------------------------------- #
def build_weights(returns: pd.DataFrame, scheme: str = "inverse_vol") -> pd.Series:
    """按指定方案构造**静态**组合权重（和为 1 的多头组合）。

    - ``"equal"``：等权 1/n；
    - ``"inverse_vol"``：逆波动权重 w_i ∝ 1/σ_i（σ_i 为该标的的每期样本波动），
      即最朴素的「风险均衡」近似，低波标的获得更高权重。

    权重用**全样本**波动一次性确定且不随时间调整，属于样本内的静态组合口径
    （目的是风险度量与分解，不是可交易策略，故不含调仓与交易成本）。
    """
    if scheme not in WEIGHT_SCHEMES:
        raise ValueError(f"scheme 只支持 {WEIGHT_SCHEMES}，收到 {scheme!r}")
    cols = list(returns.columns)
    if scheme == "equal":
        return pd.Series(1.0 / len(cols), index=cols, name="weight")
    vol = kr.volatility(returns)                      # 每期波动（Series）
    if float(vol.min()) <= 0:
        raise ValueError("存在零波动标的，逆波动权重无定义；请改用 --scheme equal")
    inv = 1.0 / vol
    return (inv / inv.sum()).rename("weight")


# --------------------------------------------------------------------------- #
# 指标收集：在真实数据上跑完整条风险链路
# --------------------------------------------------------------------------- #
def collect(returns: pd.DataFrame, prices_raw: pd.DataFrame, weights: pd.Series,
            cfg: argparse.Namespace, data_diag: Dict[str, Any]) -> Dict[str, Any]:
    """在真实收益面板上依次计算度量/分解/因子/回撤/压测/尾部/回测指标。

    返回嵌套字典：数值部分可直接 JSON 化，markdown 表放在 ``_tables``、
    月末水下曲线放在 ``_monthly_underwater``（均以 ``_`` 前缀排除在 JSON 之外）。
    """
    ppy = float(cfg.periods_per_year)
    value = float(cfg.value)
    c95, c99, c999 = float(cfg.confidence), float(cfg.confidence99), float(cfg.confidence999)
    ann = math.sqrt(ppy)
    port = kr.portfolio_returns(returns, weights)
    cov = kr.covariance_matrix(returns)
    assets = [str(c) for c in returns.columns]
    w_map = series_map(weights, assets)
    tables: Dict[str, str] = {}
    M: Dict[str, Any] = {}

    # ---- 0. 元信息与数据口径 ------------------------------------------- #
    M["meta"] = {
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_dir": str(cfg.data_dir),
        "n_files": int(prices_raw.attrs.get("n_files", prices_raw.shape[1])),
        "sample_start": str(pd.Timestamp(returns.index[0]).date()),
        "sample_end": str(pd.Timestamp(returns.index[-1]).date()),
        "n_obs": int(returns.shape[0]),
        "n_assets": int(returns.shape[1]),
        "assets": assets,
        "scheme": str(cfg.scheme),
        "portfolio_value": value,
        "periods_per_year": ppy,
        "confidence": c95,
        "confidence99": c99,
        "confidence999": c999,
        "n_trials": int(cfg.n_trials),
        "kairos_risk_version": kr.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_available": have_scipy(),
        "command": f"python examples/real_risk_report.py --data-dir {cfg.data_dir} "
                   f"--scheme {cfg.scheme}",
    }
    M["data"] = data_diag

    # ---- 1. 组合与基础风险度量 ----------------------------------------- #
    rep_port = kr.risk_report(port, confidence=c95, periods_per_year=ppy)["portfolio"]
    measures: Dict[str, Any] = {
        "n_obs": int(port.size),
        "annualized_return": finite(kr.annualized_return(port, ppy)),
        "annualized_volatility": finite(kr.volatility(port, ppy)),
        "daily_volatility": finite(kr.volatility(port)),
        "sharpe": finite(kr.sharpe_ratio(port, periods_per_year=ppy)),
        "sortino": finite(kr.sortino_ratio(port, periods_per_year=ppy)),
        "skewness": finite(kr.skewness(port)),
        "excess_kurtosis": finite(kr.kurtosis(port)),
        "downside_deviation_annualized": finite(kr.downside_deviation(port, periods_per_year=ppy)),
        "best_day": finite(float(port.max())),
        "worst_day": finite(float(port.min())),
        "win_rate": finite(kr.win_rate(port)),
        "var": {}, "es": {}, "var_annualized_sqrt_rule": {},
    }
    for tag, c in (("95", c95), ("99", c99)):
        measures["var"][tag] = {
            "historical": finite(kr.historical_var(port, c)),
            "parametric_normal": finite(kr.parametric_var(port, c, distribution="normal")),
            "parametric_student_t": finite(kr.parametric_var(port, c, distribution="student_t")),
            "modified_cornish_fisher": finite(kr.modified_var(port, c)),
        }
        measures["es"][tag] = {
            "historical": finite(kr.expected_shortfall(port, c, method="historical")),
            "normal": finite(kr.expected_shortfall(port, c, method="normal")),
            "student_t": finite(kr.expected_shortfall(port, c, method="student_t")),
        }
        measures["var_annualized_sqrt_rule"][tag] = {
            k: (None if v is None else v * ann) for k, v in measures["var"][tag].items()
        }
    v95, v99 = measures["var"]["95"], measures["var"]["99"]
    e95, e99 = measures["es"]["95"], measures["es"]["99"]
    M["measures"] = measures

    rows = []
    for tag in ("95", "99"):
        rows.append({
            "历史 VaR": pct(measures["var"][tag]["historical"]),
            "参数 VaR(正态)": pct(measures["var"][tag]["parametric_normal"]),
            "参数 VaR(学生t)": pct(measures["var"][tag]["parametric_student_t"]),
            "修正 VaR(CF)": pct(measures["var"][tag]["modified_cornish_fisher"]),
            "历史 ES": pct(measures["es"][tag]["historical"]),
            "ES(正态)": pct(measures["es"][tag]["normal"]),
            "ES(学生t)": pct(measures["es"][tag]["student_t"]),
        })
    tables["var_es"] = md_table(pd.DataFrame(rows, index=pd.Index(["95%", "99%"], name="置信度")),
                                nd=4)

    w_sorted = pd.Series(w_map).sort_values(ascending=False)
    hhi = float((weights ** 2).sum())
    M["portfolio"] = {
        "n_assets": int(weights.size),
        "weight_sum": finite(float(weights.sum())),
        "weight_max": finite(float(w_sorted.iloc[0])),
        "weight_min": finite(float(w_sorted.iloc[-1])),
        "weight_median": finite(float(np.median(np.asarray(weights, dtype="float64")))),
        "weight_hhi": finite(hhi),
        "effective_n_by_weight": finite(1.0 / hhi),
        "top5_weights": {k: finite(v) for k, v in w_sorted.head(5).items()},
        "bottom5_weights": {k: finite(v) for k, v in w_sorted.tail(5).items()},
        "risk_report_portfolio": {str(k): finite(v) for k, v in rep_port.items()},
    }
    top8 = w_sorted.head(8)
    tables["weights"] = md_table(pd.DataFrame({
        "权重": top8,
        "年化波动": kr.volatility(returns, ppy).reindex(top8.index),
        "区间累计收益": ((1.0 + returns).prod() - 1.0).reindex(top8.index),
    }, index=pd.Index(top8.index, name="标的")), nd=4)

    # 逐资产真实概览
    per_asset = pd.DataFrame({
        "年化收益": kr.annualized_return(returns, ppy),
        "年化波动": kr.volatility(returns, ppy),
        f"VaR{int(c95 * 100)}%(每期)": kr.historical_var(returns, c95),
        f"ES{int(c95 * 100)}%(每期)": kr.expected_shortfall(returns, c95),
        "最大回撤": pd.Series({str(c): kr.max_drawdown(returns[c]) for c in returns.columns}),
        "偏度": kr.skewness(returns),
        "超额峰度": kr.kurtosis(returns),
    })
    per_asset.index = assets
    M["per_asset"] = frame_records(per_asset)
    tables["per_asset"] = md_table(per_asset.sort_values("年化波动"), nd=4, index_name="标的")

    # ---- 2. 成分风险分解（欧拉可加性） --------------------------------- #
    rdec = kr.component_risk(weights, cov)
    mrc = kr.marginal_risk(weights, cov)
    ew_ref = pd.Series(1.0 / returns.shape[1], index=returns.columns, name="weight")
    ew_rdec = kr.component_risk(ew_ref, cov)
    ce95 = kr.component_es(weights, returns, c95)
    cv95 = kr.component_var(weights, returns, c95)
    ce99 = kr.component_es(weights, returns, c99)
    comp_sum = float(np.sum(np.asarray(rdec.component, dtype="float64")))
    pct_s = pd.Series(np.asarray(rdec.pct, dtype="float64"), index=assets).sort_values(
        ascending=False)
    detail = pd.DataFrame({
        "权重": pd.Series(w_map),
        "边际风险MRC": pd.Series(series_map(mrc, assets)),
        "成分风险CRC": pd.Series(series_map(rdec.component, assets)),
        "风险占比": pct_s,
        "成分ES占比": pd.Series(series_map(ce95.pct, assets)),
    }).loc[pct_s.index]
    dr_bound = math.sqrt(float(returns.shape[1]))       # 两两不相关时逆波动组合的 DR 上界 √n
    decomp = {
        "portfolio_volatility_daily": finite(rdec.volatility),
        "portfolio_volatility_annualized": finite(rdec.volatility * ann),
        "component_sum": finite(comp_sum),
        "additivity_abs_error": finite(abs(comp_sum - rdec.volatility)),
        "weighted_volatility": finite(rdec.weighted_volatility),
        "diversification_ratio": finite(kr.diversification_ratio(weights, cov)),
        "diversification_ratio_independent_bound": finite(dr_bound),
        "diversification_realized_ratio": finite(rdec.diversification_ratio / dr_bound),
        "risk_herfindahl": finite(rdec.herfindahl),
        "risk_herfindahl_equal_share": finite(1.0 / returns.shape[1]),
        "risk_effective_n": finite(kr.risk_effective_n(weights, cov)),
        "equal_weight_reference": {
            "portfolio_volatility_annualized": finite(ew_rdec.volatility * ann),
            "diversification_ratio": finite(ew_rdec.diversification_ratio),
            "risk_effective_n": finite(ew_rdec.effective_n),
            "risk_herfindahl": finite(ew_rdec.herfindahl),
        },
        "component_pct_all": {k: finite(v) for k, v in pct_s.items()},
        "marginal_all": series_map(mrc, assets),
        "component_es_value": finite(ce95.value),
        "component_es_sum": finite(float(np.sum(np.asarray(ce95.component, dtype="float64")))),
        "component_es_additivity_error": finite(
            abs(float(np.sum(np.asarray(ce95.component, dtype="float64"))) - ce95.value)),
        "component_var_value": finite(cv95.value),
        "component_var_sum": finite(float(np.sum(np.asarray(cv95.component, dtype="float64")))),
        "component_var_additivity_error": finite(
            abs(float(np.sum(np.asarray(cv95.component, dtype="float64"))) - cv95.value)),
        "component_es99_value": finite(ce99.value),
        "tail_threshold_95": finite(ce95.threshold),
        "n_tail_95": int(ce95.n_tail),
        "top_contributors": {k: finite(v) for k, v in pct_s.head(10).items()},
    }
    M["decomposition"] = decomp
    tables["decomposition"] = md_table(detail.head(12), nd=4, index_name="标的")

    # ---- 3. 因子风险（等权组合作市场代理） ----------------------------- #
    mkt = kr.portfolio_returns(returns, ew_ref).rename(MARKET_FACTOR_NAME)
    fm = kr.market_model(returns, mkt, market_name=MARKET_FACTOR_NAME)
    brk = kr.factor_risk_decomposition(weights, fm)
    expo = kr.exposure_summary(fm)
    betas = pd.Series(np.asarray(expo[MARKET_FACTOR_NAME], dtype="float64"), index=assets)
    sample_vol = float(kr.volatility(port))
    factor = {
        "market_factor": f"{MARKET_FACTOR_NAME}（{len(assets)} 只标的的等权组合代理）",
        "portfolio_exposure": finite(float(brk.portfolio_exposure.iloc[0])),
        "factor_variance": finite(brk.factor_variance),
        "idio_variance": finite(brk.idio_variance),
        "total_variance": finite(brk.total_variance),
        "factor_share": finite(brk.factor_share),
        "idio_share": finite(1.0 - brk.factor_share),
        "model_volatility_daily": finite(brk.volatility),
        "sample_volatility_daily": finite(sample_vol),
        "model_vs_sample_vol_ratio": finite(brk.volatility / sample_vol),
        "factor_additivity_error": finite(
            abs(brk.factor_variance + brk.idio_variance - brk.total_variance)),
        "beta_mean": finite(float(betas.mean())),
        "beta_median": finite(float(betas.median())),
        "beta_min": finite(float(betas.min())),
        "beta_max": finite(float(betas.max())),
        "r2_mean": finite(float(fm.r2.mean())),
        "r2_median": finite(float(fm.r2.median())),
        "beta_all": series_map(betas, assets),
        "r2_all": series_map(fm.r2, assets),
        "idio_contrib_pct": series_map(brk.idio_pct, assets),
        "top_idio_contributors": {str(k): finite(v) for k, v in
                                  brk.idio_pct.sort_values(ascending=False).head(5).items()},
        "market_annualized_return": finite(kr.annualized_return(mkt, ppy)),
        "market_annualized_volatility": finite(kr.volatility(mkt, ppy)),
    }
    M["factor"] = factor
    ftbl = pd.DataFrame({
        "beta": betas,
        "R2": pd.Series(series_map(fm.r2, assets)),
        "alpha(每期)": pd.Series(series_map(fm.alphas, assets)),
        "特质波动(每期)": pd.Series(series_map(fm.resid_std, assets)),
        "权重": pd.Series(w_map),
    }).sort_values("beta", ascending=False)
    tables["factor"] = md_table(ftbl.head(12), nd=4, index_name="标的")

    # ---- 4. 回撤分析 ---------------------------------------------------- #
    ddr = kr.drawdown_report(port, periods_per_year=ppy)
    top_eps = kr.top_drawdowns(port, n=int(cfg.top_drawdowns))
    dur = kr.duration_summary(ddr.episodes)
    worst_ep = max(ddr.episodes, key=lambda e: e.depth) if ddr.episodes else None
    drawdown: Dict[str, Any] = {str(k): finite(v) for k, v in ddr.summary().items()}
    drawdown.update({
        "n_episodes": int(ddr.n_episodes),
        "longest_drawdown_periods": int(ddr.longest_periods),
        "wealth_final": finite(float(kr.wealth_curve(port).iloc[-1])),
        "wealth_peak": finite(float(kr.running_peak(port).max())),
        "worst_episode": None if worst_ep is None else {
            "peak_date": dstr(worst_ep.start), "trough_date": dstr(worst_ep.trough),
            "recovery_date": dstr(worst_ep.end), "depth": finite(worst_ep.depth),
            "down_periods": int(worst_ep.down_periods),
            "recovery_periods": finite(worst_ep.recovery_periods),
            "total_periods": int(worst_ep.total_periods),
            "recovered": bool(worst_ep.recovered),
        },
        "top_episodes": [{
            "peak_date": dstr(e.start), "trough_date": dstr(e.trough),
            "recovery_date": dstr(e.end), "depth": finite(e.depth),
            "down_periods": int(e.down_periods),
            "recovery_periods": finite(e.recovery_periods),
            "total_periods": int(e.total_periods), "recovered": bool(e.recovered),
        } for e in top_eps],
        "duration_summary": frame_records(dur),
    })
    M["drawdown"] = drawdown

    ep_tbl = kr.episodes_frame(top_eps).copy()
    ep_tbl.index = [dstr(i) for i in ep_tbl.index]
    ep_tbl = pd.DataFrame({
        "谷底日": [dstr(v) for v in ep_tbl["trough"]],
        "收复日": [dstr(v) for v in ep_tbl["end"]],
        "深度": [pct(v) for v in ep_tbl["depth"]],
        "下跌期数": [str(int(v)) for v in ep_tbl["down_periods"]],
        "恢复期数": ["未收复" if pd.isna(v) else str(int(v))
                     for v in ep_tbl["recovery_periods"]],
        "水下期数": [str(int(v)) for v in ep_tbl["total_periods"]],
        "已收复": ["是" if bool(v) else "否" for v in ep_tbl["recovered"]],
    }, index=ep_tbl.index)
    tables["drawdown_episodes"] = md_table(ep_tbl, nd=4, index_name="峰值日")
    dur_cn = dur.rename(index={"count": "样本数", "min": "最小", "p25": "P25",
                               "median": "中位", "p75": "P75", "max": "最大", "mean": "均值"},
                        columns={"depth": "深度", "down_periods": "下跌期数",
                                 "recovery_periods": "恢复期数", "total_periods": "水下期数"})
    tables["drawdown_duration"] = md_table(dur_cn, nd=2, index_name="统计量")

    uw = kr.underwater(port)
    monthly = uw.groupby(uw.index.to_period("M")).last()
    monthly.index = [str(p) for p in monthly.index]

    # ---- 5. 压力测试 ---------------------------------------------------- #
    beta_map = {str(k): float(v) for k, v in betas.items()}
    worst_day = returns.index[int(np.argmin(port.to_numpy()))]
    scenarios = [
        kr.Scenario.market(beta_map, shock=float(cfg.market_shock),
                           vol_multiplier=float(cfg.vol_multiplier), name="market_crash"),
        kr.Scenario.idiosyncratic({str(pct_s.index[0]): float(cfg.idio_shock)},
                                  name=f"idio_{pct_s.index[0]}"),
        kr.Scenario.vol_spike(float(cfg.vol_multiplier)),
        kr.Scenario.from_history(returns, worst_day, name="worst_history_day"),
    ]
    sc_tbl = kr.scenario_table(weights, scenarios, value=value, cov=cov, confidence=c95)
    replay_1d = kr.historical_replay(returns, weights, n_worst=int(cfg.n_worst),
                                     window=1, value=value)
    replay_win = kr.historical_replay(returns, weights, n_worst=int(cfg.n_worst),
                                      window=int(cfg.replay_window), value=value,
                                      non_overlapping=True)
    target_loss = value * float(cfg.reverse_target_pct)
    rev_pnl = kr.reverse_stress_test(weights, scenarios[0], target_loss=target_loss,
                                     value=value, metric="pnl")
    rev_var = kr.reverse_stress_test(weights, scenarios[0], target_loss=target_loss,
                                     value=value, cov=cov, confidence=c95, metric="var")
    stress = {
        "market_shock": float(cfg.market_shock),
        "vol_multiplier": float(cfg.vol_multiplier),
        "idio_shock": float(cfg.idio_shock),
        "idio_target_asset": str(pct_s.index[0]),
        "worst_history_day": dstr(worst_day),
        "portfolio_value": value,
        "n_worst": int(cfg.n_worst),
        "reverse_target_pct": float(cfg.reverse_target_pct),
        "reverse_target_loss": finite(target_loss),
        "scenario_table": frame_records(sc_tbl),
        "replay_1d": {
            "window": 1, "n_windows": int(replay_1d.n_windows),
            "worst_pnl": finite(replay_1d.worst_pnl),
            "mean_pnl": finite(replay_1d.mean_pnl),
            "total_pnl": finite(replay_1d.total_pnl),
            "hit_dates": [dstr(d) for d in replay_1d.hit_dates],
            "hit_returns": [finite(v) for v in replay_1d.frame["portfolio_return"].tolist()],
            "worst_asset_contrib": {
                str(k): finite(v) for k, v in replay_1d.frame.iloc[0].drop(
                    ["portfolio_return", "pnl"]).sort_values().head(3).items()},
        },
        "replay_multi_day": {
            "window": int(cfg.replay_window), "n_windows": int(replay_win.n_windows),
            "non_overlapping": True,
            "worst_pnl": finite(replay_win.worst_pnl),
            "mean_pnl": finite(replay_win.mean_pnl),
            "total_pnl": finite(replay_win.total_pnl),
            "hit_dates": [dstr(d) for d in replay_win.hit_dates],
            "hit_returns": [finite(v) for v in replay_win.frame["portfolio_return"].tolist()],
        },
        "reverse_pnl": {"metric": "pnl", "multiplier": finite(rev_pnl.multiplier),
                        "achieved_loss": finite(rev_pnl.achieved_loss),
                        "implied_market_shock": finite(rev_pnl.multiplier * cfg.market_shock),
                        "converged": bool(rev_pnl.converged),
                        "iterations": int(rev_pnl.iterations)},
        "reverse_var": {"metric": "var", "multiplier": finite(rev_var.multiplier),
                        "achieved_loss": finite(rev_var.achieved_loss),
                        "implied_market_shock": finite(rev_var.multiplier * cfg.market_shock),
                        "converged": bool(rev_var.converged),
                        "iterations": int(rev_var.iterations)},
    }
    M["stress"] = stress
    sc_show = pd.DataFrame({
        "损益率": [pct(r["pnl_pct"]) for _, r in sc_tbl.iterrows()],
        "损益": [money(r["pnl"]) for _, r in sc_tbl.iterrows()],
        "冲击后市值": [money(r["value_after"]) for _, r in sc_tbl.iterrows()],
        "最惨标的冲击": [pct(r["worst_asset_shock"]) for _, r in sc_tbl.iterrows()],
        "冲击后波动(每期)": [num(r["vol_stressed"], 5) for _, r in sc_tbl.iterrows()],
        "冲击后VaR95%": [pct(r["var_stressed"]) for _, r in sc_tbl.iterrows()],
    }, index=pd.Index([str(i) for i in sc_tbl.index], name="情景"))
    tables["scenarios"] = md_table(sc_show, nd=4)
    tables["replay_1d"] = md_table(pd.DataFrame({
        "组合收益": [finite(v) for v in replay_1d.frame["portfolio_return"].tolist()],
        "损益": [money(v) for v in replay_1d.frame["pnl"].tolist()],
    }, index=pd.Index([dstr(d) for d in replay_1d.hit_dates], name="日期")), nd=4)
    tables["replay_window"] = md_table(pd.DataFrame({
        f"{int(cfg.replay_window)}日累计收益": [
            finite(v) for v in replay_win.frame["portfolio_return"].tolist()],
        "损益": [money(v) for v in replay_win.frame["pnl"].tolist()],
    }, index=pd.Index([dstr(d) for d in replay_win.hit_dates], name="窗口结束日")), nd=4)

    # ---- 6. EVT 尾部 ---------------------------------------------------- #
    hill = kr.hill_estimator(port, k=(int(cfg.hill_k) if cfg.hill_k else None),
                             tail_quantile=float(cfg.hill_tail_quantile))
    hpath = kr.hill_path(port, k_min=int(cfg.hill_k_min), n_points=15)
    fit = kr.fit_gpd(port, tail_quantile=float(cfg.tail_quantile))
    tail: Dict[str, Any] = {
        "hill_xi": finite(hill.xi), "hill_se": finite(hill.se), "hill_k": int(hill.k),
        "hill_threshold": finite(hill.threshold),
        "hill_tail_index_alpha": finite(hill.tail_index),
        "hill_n_exceed": int(hill.n_exceed), "hill_n_obs": int(hill.n_obs),
        "hill_path_k_min": int(hpath.index.min()), "hill_path_k_max": int(hpath.index.max()),
        "hill_path_xi_min": finite(float(hpath["xi"].min())),
        "hill_path_xi_median": finite(float(hpath["xi"].median())),
        "hill_path_xi_max": finite(float(hpath["xi"].max())),
        "gpd_method": fit.method, "gpd_xi": finite(fit.xi), "gpd_scale": finite(fit.scale),
        "gpd_threshold": finite(fit.threshold), "gpd_n_exceed": int(fit.n_exceed),
        "gpd_n_obs": int(fit.n_obs),
        "gpd_exceedance_prob": finite(fit.exceedance_prob),
        "gpd_mean_excess": finite(fit.mean_excess()),
        "tail_ratio_95": finite(kr.tail_ratio(port, c95)),
        "pot": {},
    }
    for tag, c in (("99", c99), ("999", c999)):
        pot_var, pot_es = kr.pot_var(fit, c), kr.pot_es(fit, c)
        tail["pot"][tag] = {
            "confidence": c,
            "pot_var": finite(pot_var), "pot_es": finite(pot_es),
            "pot_var_in_money": finite(pot_var * value), "pot_es_in_money": finite(pot_es * value),
            "historical_var": finite(kr.historical_var(port, c)),
            "historical_es": finite(kr.expected_shortfall(port, c, method="historical")),
            "parametric_normal_var": finite(kr.parametric_var(port, c, distribution="normal")),
            "parametric_student_t_var": finite(
                kr.parametric_var(port, c, distribution="student_t")),
        }
    tail["tail_summary"] = {str(i): finite(v) for i, v in
                            kr.tail_summary(port, confidence=c99,
                                            tail_quantile=float(cfg.tail_quantile))["value"].items()}
    M["tail"] = tail
    tables["tail"] = md_table(pd.DataFrame({
        "POT/GPD VaR": [pct(tail["pot"]["99"]["pot_var"]), pct(tail["pot"]["999"]["pot_var"])],
        "POT/GPD ES": [pct(tail["pot"]["99"]["pot_es"]), pct(tail["pot"]["999"]["pot_es"])],
        "历史 VaR": [pct(tail["pot"]["99"]["historical_var"]),
                     pct(tail["pot"]["999"]["historical_var"])],
        "历史 ES": [pct(tail["pot"]["99"]["historical_es"]),
                    pct(tail["pot"]["999"]["historical_es"])],
        "正态 VaR": [pct(tail["pot"]["99"]["parametric_normal_var"]),
                     pct(tail["pot"]["999"]["parametric_normal_var"])],
        "学生t VaR": [pct(tail["pot"]["99"]["parametric_student_t_var"]),
                      pct(tail["pot"]["999"]["parametric_student_t_var"])],
        "POT VaR 金额": [money(tail["pot"]["99"]["pot_var_in_money"]),
                         money(tail["pot"]["999"]["pot_var_in_money"])],
    }, index=pd.Index([f"{c99:.1%}", f"{c999:.1%}"], name="置信度")), nd=4)

    # ---- 7. 回测风险（PSR / DSR） -------------------------------------- #
    sr = kr.per_period_sharpe(port)
    wci = kr.win_rate_ci(port)
    bt: Dict[str, Any] = {
        "n_obs": int(port.size),
        "sharpe_per_period": finite(sr),
        "sharpe_annualized": finite(sr * ann),
        "psr": finite(kr.psr_from_returns(port)),
        "dsr": {},
        "min_track_length_95": finite(kr.minimum_track_length(sr, 0.95)) if sr > 0 else None,
        "trials_budget_for_dsr95": int(kr.trials_for_target_dsr(sr, int(port.size),
                                                               target_dsr=0.95)) if sr > 0 else 0,
        "win_rate": finite(wci.point),
        "win_rate_ci": {"lower": finite(wci.lower), "upper": finite(wci.upper),
                        "confidence": finite(wci.confidence), "method": wci.method},
        "loss_probability": finite(kr.loss_probability(port, 0.0)),
        "probability_of_min_loss_21d": finite(kr.probability_of_min_loss(port, horizon=21)),
        "probability_of_min_loss_252d": finite(kr.probability_of_min_loss(port, horizon=252)),
        "backtest_report": {str(i): finite(r["value"]) for i, r in
                            kr.backtest_risk_report(port, n_trials=int(cfg.n_trials),
                                                    horizon=21).iterrows()},
    }
    for n in (1, int(returns.shape[1]), int(cfg.n_trials)):
        d = kr.dsr_from_returns(port, n_trials=n)
        bt["dsr"][str(n)] = {"dsr": finite(d.dsr), "psr": finite(d.psr), "sr0": finite(d.sr0),
                             "haircut": finite(d.haircut), "sharpe": finite(d.sharpe),
                             "sharpe_variance": finite(d.sharpe_variance),
                             "skew": finite(d.skew), "kurtosis": finite(d.kurtosis)}
    M["backtest"] = bt
    tables["dsr"] = md_table(pd.DataFrame(bt["dsr"]).T[["dsr", "psr", "sr0", "haircut",
                                                        "sharpe_variance"]].rename(columns={
        "dsr": "DSR", "psr": "PSR", "sr0": "门槛SR0", "haircut": "缩水幅度",
        "sharpe_variance": "夏普方差V"}), nd=4, index_name="试验次数N")

    # ---- 8. 一致性校验（自证） ----------------------------------------- #
    checks = {
        "euler_component_sum_equals_vol": decomp["additivity_abs_error"] < 1e-12,
        "component_es_sum_equals_es": decomp["component_es_additivity_error"] < 1e-12,
        "component_es_matches_portfolio_es": abs(
            ce95.value - kr.expected_shortfall(port, c95, method="historical")) < 1e-12,
        "component_var_sum_equals_var": decomp["component_var_additivity_error"] < 1e-9,
        "factor_plus_idio_equals_total": factor["factor_additivity_error"] < 1e-18,
        "model_vol_close_to_sample_vol": abs(factor["model_vs_sample_vol_ratio"] - 1.0) < 0.25,
        "es_ge_var_95": e95["historical"] >= v95["historical"],
        "es_ge_var_99": e99["historical"] >= v99["historical"],
        "var99_ge_var95": v99["historical"] >= v95["historical"],
        "student_t_var99_ge_normal_var99": v99["parametric_student_t"] >= v99["parametric_normal"],
        "pot_var99_above_gpd_threshold": tail["pot"]["99"]["pot_var"] > tail["gpd_threshold"],
        "pot_es99_ge_pot_var99": tail["pot"]["99"]["pot_es"] >= tail["pot"]["99"]["pot_var"],
        "pot_es999_ge_pot_var999": tail["pot"]["999"]["pot_es"] >= tail["pot"]["999"]["pot_var"],
        "pot_var999_ge_pot_var99": tail["pot"]["999"]["pot_var"] >= tail["pot"]["99"]["pot_var"],
        "pot_var999_ge_var95_historical": tail["pot"]["999"]["pot_var"] >= v95["historical"],
        "diversification_ratio_ge_one": decomp["diversification_ratio"] >= 1.0,
        "risk_effective_n_within_assets": 1.0 <= decomp["risk_effective_n"] <= returns.shape[1] + 1e-9,
        "weights_sum_to_one": abs(float(weights.sum()) - 1.0) < 1e-12,
        "weights_all_positive": bool(float(np.min(np.asarray(weights, dtype="float64"))) > 0),
        "all_returns_within_price_limit": bool(
            float(np.nanmax(np.abs(returns.to_numpy(dtype="float64")))) <= 0.31),
        "dsr_le_psr": all(v["dsr"] <= v["psr"] + 1e-12 for v in bt["dsr"].values()),
        "dsr_monotone_in_trials": bt["dsr"][str(int(cfg.n_trials))]["dsr"]
                                  <= bt["dsr"]["1"]["dsr"] + 1e-12,
        "hill_xi_positive_fat_tail": bool(tail["hill_xi"] > 0),
        "drawdown_within_unit": 0.0 <= float(ddr.max_drawdown) <= 1.0,
        "json_strictly_valid": _json_strictly_valid(M),
    }
    M["checks"] = {k: bool(v) for k, v in checks.items()}
    M["_tables"] = tables
    M["_monthly_underwater"] = monthly
    return M


def _json_strictly_valid(M: Dict[str, Any]) -> bool:
    """校验指标字典能被 ``json.dump(allow_nan=False)`` 严格序列化（无 NaN/Inf）。"""
    payload = json_safe({k: v for k, v in M.items() if not k.startswith("_")})
    try:
        json.dumps(payload, ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError):
        return False
    return True


# --------------------------------------------------------------------------- #
# 报告渲染
# --------------------------------------------------------------------------- #
def render_report(M: Dict[str, Any]) -> str:
    """把指标字典渲染成中文 markdown 报告（分节：数据 → 度量 → 分解 → 因子 → 回撤 →
    压测 → 尾部 → 回测 → 校验 → 结论）。"""
    meta, data = M["meta"], M["data"]
    m, dcp, fac = M["measures"], M["decomposition"], M["factor"]
    dd, st, tl, bt = M["drawdown"], M["stress"], M["tail"], M["backtest"]
    T = M["_tables"]
    v95, v99 = m["var"]["95"], m["var"]["99"]
    e95, e99 = m["es"]["95"], m["es"]["99"]
    p99, p999 = tl["pot"]["99"], tl["pot"]["999"]
    repair, sani, summ = data["repair"], data["sanitize"], data["summary"]
    mk = st["scenario_table"]["market_crash"]
    r1, rw = st["replay_1d"], st["replay_multi_day"]
    L: List[str] = []
    A = L.append

    A("# Kairos Risk · 真实数据组合风险报告（real_risk）")
    A("")
    A(f"- 生成时间：{meta['generated_at']}")
    A(f"- 数据来源：`{meta['data_dir']}`（{meta['n_files']} 个标的的真实前复权日线 CSV，**只读**）")
    A(f"- 样本区间：**{meta['sample_start']} ~ {meta['sample_end']}**"
      f"（{meta['n_obs']} 个交易日 × {meta['n_assets']} 个标的）")
    A(f"- 组合方案：**{meta['scheme']}**（静态权重，样本内一次性确定，不含调仓与交易成本），"
      f"组合市值 {money(meta['portfolio_value'])}")
    A(f"- 置信度口径：主口径 {meta['confidence']:.0%}，尾部 {meta['confidence99']:.0%} / "
      f"{meta['confidence999']:.1%}；年化期数 {int(meta['periods_per_year'])}")
    A(f"- 依赖版本：kairos_risk {meta['kairos_risk_version']} / numpy {meta['numpy_version']} / "
      f"pandas {meta['pandas_version']} / scipy "
      f"{'可用（GPD 走极大似然）' if meta['scipy_available'] else '缺失（走自研 PWM 回退）'}")
    A(f"- 复现命令：`{meta['command']}`")
    A("")

    # ---------------- 0. 数据口径 ----------------
    A("## 0. 数据口径与质量（先说清楚数字从哪来）")
    A("")
    A("本仓库不生产数据，只读同系列 `kairos-data` 已提交的真实日线。真实数据有五个必须处理的"
      "口径问题，全部由 `kairos_risk.realdata` 完成，且每一步都留痕：")
    A("")
    dropped_txt = ("、".join(
        f"`{k}`（压缩倍数 k̂={num(repair['compression_max'].get(k), 1)}）"
        for k in repair["dropped_symbols"]) or "无")
    A(f"1. **前复权价可能为负**：{summ['masked_prices_total']} 条价格 ≤0 或非法（长期高分红标的的"
      f"减法复权会把早期价格压到 0 以下），已记为缺失；样本裁剪到「全体标的均处于严格正价区间」"
      f"的公共窗口，丢弃 {summ['dropped_prices']} 条早期价格，因此区间起点是 "
      f"{data['panel_window'][0]} 而非数据最早的 2016 年。")
    A(f"2. **停牌/缺口**：窗口内 {summ['n_missing_prices']} 个缺失单元格，用前向填充"
      "（当日无成交则沿用前收盘价，即当日收益 0），只用历史信息、无未来函数。")
    A(f"3. **等差（减法）前复权放大收益**：该数据源的复权是 `P_adj = P_raw − D_t`"
      f"（D_t 为 t 日之后的累计分红），于是 `pct_change` 得到的是被放大 "
      f"k=P_raw/P_adj 倍的伪收益。修复前共 **{repair['violations_before']} 个交易日的收益超出"
      "交易所涨跌停幅度**（物理上不可能），高分红标的的年化波动甚至被放大到 8000%。"
      "修复方式：用涨跌停硬约束反解每标的每日的偏移**下界** D̂，令 `P_repaired = P_adj + D̂`；"
      f"因 D̂ ≤ D_真，修复只会把伪收益拉回真实口径、**不会把风险修小**。修复后残余越界 "
      f"{repair['violations_after']} 个。")
    A(f"4. **压缩过度、无法修复的标的**：{len(repair['dropped_symbols'])} 只被整只剔除"
      f"（{dropped_txt}，闸门 `max_compression={num(repair['max_compression'], 1)}`）——"
      "它们的复权价被压到接近 0（个别日期仅几分钱），源数据 2~3 位小数的舍入误差已主导收益，"
      "下界修复无法挽回；剔除原因与 k̂ 值同时写入 `risk_metrics.json`。")
    A(f"5. **残余越界观测**：{sani['bad_cells']} 个单元格仍越界，按 `on_bad={sani['on_bad']}` "
      "处理（取同号的涨跌停边界值：真实收益必在 ±L 内，取边界是**不低估尾部**的可辨识替代，"
      f"且不牵连同日其它标的的真实行情）；连带删除的交易日 {sani['dropped_days']} 天。")
    A("")
    A(f"最终分析样本：**{sani['n_assets_out']} 个标的 × {sani['n_obs']} 个交易日**，"
      f"单日最大绝对收益 {pct(sani['max_abs_return'])}（已落在涨跌停物理约束内）。")
    A("")
    A("| 口径 | 数值 |")
    A("|---|---|")
    A(f"| CSV 文件数 / 原始标的数 | {data['n_files']} / {repair['n_assets_in']} |")
    A(f"| 收盘价面板（裁剪后） | {data['panel_shape'][0]} 日 × {data['panel_shape'][1]} 标的 |")
    A(f"| 面板区间 | {data['panel_window'][0]} ~ {data['panel_window'][1]} |")
    A(f"| 前复权无效价（≤0/NaN）条数 | {summ['masked_prices_total']} |")
    A(f"| 裁剪丢弃的早期有效价条数 | {summ['dropped_prices']} |")
    A(f"| 窗口内缺失单元格（停牌/缺口，已前填） | {summ['n_missing_prices']} |")
    A(f"| 涨跌停越界：修复前 / 修复后 / 最终处理 | {repair['violations_before']} / "
      f"{repair['violations_after']} / {sani['bad_cells']} |")
    A(f"| 剔除标的数（压缩过度 / 残余越界） | {len(repair['dropped_symbols'])} / "
      f"{len(sani['dropped_symbols'])} |")
    A(f"| 年化波动：横截面中位 / 最小 / 最大 | {num(summ['annualized_vol_median'], 4)} / "
      f"{num(summ['annualized_vol_min'], 4)} / {num(summ['annualized_vol_max'], 4)} |")
    A(f"| 全横截面单日最差 / 最好 | {pct(summ['worst_single_day'])} / "
      f"{pct(summ['best_single_day'])} |")
    A("")
    A("> 残余偏差声明：D̂ 是**下界**估计，因此高分红标的早期（2019-2021）的收益仍可能被小幅高估，"
      "其波动/VaR 偏保守（偏大）；越接近样本末端（D→0）越精确。全部数字仅用于**方法演示**，"
      "不构成任何投资建议。")
    A("")
    A("**逐资产真实概览**（按年化波动升序；收益/波动为年化，VaR/ES/最大回撤为每期口径）")
    A("")
    A(T["per_asset"])
    A("")

    # ---------------- 1. 风险度量 ----------------
    A("## 1. 组合风险度量")
    A("")
    A(f"**组合权重（{meta['scheme']}，前 8 大）**")
    A("")
    A(T["weights"])
    A("")
    A(f"权重最大 {pct(M['portfolio']['weight_max'])}、最小 {pct(M['portfolio']['weight_min'])}、"
      f"中位 {pct(M['portfolio']['weight_median'])}；权重赫芬达尔 "
      f"{num(M['portfolio']['weight_hhi'], 4)}（权重口径有效资产数 "
      f"{num(M['portfolio']['effective_n_by_weight'], 2)} 只）。")
    A("")
    A("| 指标 | 数值 | 说明 |")
    A("|---|---|---|")
    A(f"| 年化收益 | {pct(m['annualized_return'])} | 几何（复利）年化 |")
    A(f"| 年化波动 | {pct(m['annualized_volatility'])} | 每期 σ={num(m['daily_volatility'], 5)}"
      f" × √{int(meta['periods_per_year'])} |")
    A(f"| 夏普 / Sortino | {num(m['sharpe'], 3)} / {num(m['sortino'], 3)} | rf=0 |")
    A(f"| 偏度 | {num(m['skewness'], 4)} | <0 表示左尾更长 |")
    A(f"| 超额峰度 | {num(m['excess_kurtosis'], 4)} | >0 即肥尾（正态为 0） |")
    A(f"| 下行标准差（年化） | {pct(m['downside_deviation_annualized'])} | 只计 MAR=0 以下的偏差 |")
    A(f"| 最好 / 最差单日 | {pct(m['best_day'])} / {pct(m['worst_day'])} | 真实历史极值 |")
    A(f"| 胜率 / 单期亏损概率 | {pct(m['win_rate'])} / {pct(bt['loss_probability'])} | 经验频率 |")
    A("")
    A("**VaR / ES 多口径对比**（每期损失率，正数口径）")
    A("")
    A(T["var_es"])
    A("")
    A(f"解读：{meta['confidence']:.0%} 档四种 VaR 口径差异不大（历史 {pct(v95['historical'])}、"
      f"正态 {pct(v95['parametric_normal'])}、学生 t {pct(v95['parametric_student_t'])}、"
      f"Cornish-Fisher {pct(v95['modified_cornish_fisher'])}），中位尾部主要由波动驱动；"
      f"到 {meta['confidence99']:.0%} 档，**学生 t（{pct(v99['parametric_student_t'])}）与 "
      f"CF 修正（{pct(v99['modified_cornish_fisher'])}）明显高于正态法"
      f"（{pct(v99['parametric_normal'])}）**，这正是超额峰度 {num(m['excess_kurtosis'], 2)} "
      f"带来的肥尾效应。ES 恒 ≥ 同置信度 VaR（{meta['confidence99']:.0%} 档："
      f"{pct(e99['historical'])} vs {pct(v99['historical'])}），满足一致性公理。")
    A("")
    A(f"换算成金额（市值 {money(meta['portfolio_value'])}）："
      f"{meta['confidence']:.0%} 单日历史 VaR ≈ {money(v95['historical'] * meta['portfolio_value'])}、"
      f"ES ≈ {money(e95['historical'] * meta['portfolio_value'])}；"
      f"{meta['confidence99']:.0%} VaR ≈ {money(v99['historical'] * meta['portfolio_value'])}、"
      f"ES ≈ {money(e99['historical'] * meta['portfolio_value'])}。"
      f"按 √T 规则年化，{meta['confidence99']:.0%} 历史 VaR ≈ "
      f"{pct(m['var_annualized_sqrt_rule']['99']['historical'])}。")
    A("")

    # ---------------- 2. 成分风险 ----------------
    A("## 2. 成分风险分解（欧拉可加性）")
    A("")
    A(f"组合每期波动 σp = **{num(dcp['portfolio_volatility_daily'], 6)}**"
      f"（年化 {pct(dcp['portfolio_volatility_annualized'])}）；成分风险贡献之和 Σ CRC = "
      f"{num(dcp['component_sum'], 6)}，**绝对误差 {dcp['additivity_abs_error']:.3e}**"
      "（欧拉定理要求精确可加，此处达到浮点精度级）。")
    A("")
    A("| 指标 | 数值 | 含义 |")
    A("|---|---|---|")
    A(f"| 加权个体波动 Σ&#124;w_i&#124;σ_i | {num(dcp['weighted_volatility'], 6)} | "
      "完全不分散时的波动 |")
    A(f"| **分散化比率 DR** | {num(dcp['diversification_ratio'], 4)} | ≥1，越大越分散 |")
    A(f"| DR 的「零相关」上界 √n | {num(dcp['diversification_ratio_independent_bound'], 4)} | "
      f"实际只达到其 {pct(dcp['diversification_realized_ratio'], 1)} |")
    A(f"| 风险赫芬达尔 | {num(dcp['risk_herfindahl'], 6)} | 完全均等时为 1/n = "
      f"{num(dcp['risk_herfindahl_equal_share'], 6)} |")
    A(f"| **风险有效资产数** | {num(dcp['risk_effective_n'], 2)} | 名义 {meta['n_assets']} 只 |")
    A(f"| 权重有效资产数 1/Σw² | {num(M['portfolio']['effective_n_by_weight'], 2)} | 与风险有效数对照 |")
    ewr = dcp["equal_weight_reference"]
    A(f"| 等权组合对照：DR / 风险有效N / 年化波动 | {num(ewr['diversification_ratio'], 4)} / "
      f"{num(ewr['risk_effective_n'], 2)} / {pct(ewr['portfolio_volatility_annualized'])} | "
      "换权重方案的敏感度 |")
    A(f"| 成分 ES 之和 / 组合 ES | {num(dcp['component_es_sum'], 6)} / "
      f"{num(dcp['component_es_value'], 6)} | 误差 {dcp['component_es_additivity_error']:.3e} |")
    A(f"| 成分 VaR 之和 / 组合 VaR | {num(dcp['component_var_sum'], 6)} / "
      f"{num(dcp['component_var_value'], 6)} | 误差 {dcp['component_var_additivity_error']:.3e} |")
    A(f"| 尾部门槛 / 尾部样本数（{meta['confidence']:.0%}） | "
      f"{num(dcp['tail_threshold_95'], 5)} / {dcp['n_tail_95']} | ES 的条件期望样本 |")
    A("")
    A("**风险贡献 Top12**（CRC 为每期波动口径，风险占比之和 = 1）")
    A("")
    A(T["decomposition"])
    A("")
    top3 = list(dcp["top_contributors"].items())[:3]
    A(f"解读：**逆波动权重确实把风险贡献拉平了**——风险有效资产数 "
      f"{num(dcp['risk_effective_n'], 1)}（名义 {meta['n_assets']} 只），风险赫芬达尔 "
      f"{num(dcp['risk_herfindahl'], 4)} 已逼近完全均等的 1/n="
      f"{num(dcp['risk_herfindahl_equal_share'], 4)}，风险贡献最大的 "
      + "、".join(f"`{k}`（{pct(v, 1)}）" for k, v in top3)
      + f" 也只占 {pct(top3[0][1], 1)}（前三合计 {pct(sum(v for _, v in top3), 1)}）。"
      f"但**分散化比率只有 {num(dcp['diversification_ratio'], 3)}**：若这 "
      f"{meta['n_assets']} 只标的两两不相关，逆波动组合的 DR 上界是 √n = "
      f"{num(dcp['diversification_ratio_independent_bound'], 2)}，实际只达到其 "
      f"{pct(dcp['diversification_realized_ratio'], 0)}——**约 "
      f"{pct(1 - dcp['diversification_realized_ratio'], 0)} 的分散化收益被 A 股的高相关性吃掉了**"
      f"（第 3 节将看到：单一市场因子解释了 {pct(fac['factor_share'])} 的组合方差）。"
      "结论是「权重分散、风险贡献也分散，但**风险本身并没有分散**」——组合波动仍主要由"
      "共同市场因子驱动，而非个股特质。")
    A("")

    # ---------------- 3. 因子风险 ----------------
    A("## 3. 因子风险分解（等权组合作市场代理）")
    A("")
    A(f"市场因子取**同样本内 {meta['n_assets']} 只标的的等权组合收益**"
      f"（年化收益 {pct(fac['market_annualized_return'])}、年化波动 "
      f"{pct(fac['market_annualized_volatility'])}），用 `market_model`（最小二乘）估计暴露，"
      "再用 `factor_risk_decomposition` 拆成系统性/特质两部分：")
    A("")
    A("| 指标 | 数值 |")
    A("|---|---|")
    A(f"| 组合市场暴露 b_p | **{num(fac['portfolio_exposure'], 4)}** |")
    A(f"| **系统性（因子）方差占比** | **{pct(fac['factor_share'])}** |")
    A(f"| 特质方差占比 | {pct(fac['idio_share'])} |")
    A(f"| 因子 / 特质 / 总方差（每期²） | {fac['factor_variance']:.4e} / "
      f"{fac['idio_variance']:.4e} / {fac['total_variance']:.4e} |")
    A(f"| 可加性误差（因子+特质−总） | {fac['factor_additivity_error']:.3e} |")
    A(f"| 模型隐含波动 / 样本波动 | {num(fac['model_volatility_daily'], 6)} / "
      f"{num(fac['sample_volatility_daily'], 6)}（比值 {num(fac['model_vs_sample_vol_ratio'], 4)}） |")
    A(f"| 个股 beta：均值 / 中位 / 最小 / 最大 | {num(fac['beta_mean'], 3)} / "
      f"{num(fac['beta_median'], 3)} / {num(fac['beta_min'], 3)} / {num(fac['beta_max'], 3)} |")
    A(f"| 单因子 R²：均值 / 中位 | {num(fac['r2_mean'], 4)} / {num(fac['r2_median'], 4)} |")
    A("")
    A("**个股暴露 / R² / α / 特质波动（按 beta 降序 Top12）**")
    A("")
    A(T["factor"])
    A("")
    A(f"解读：组合方差的 **{pct(fac['factor_share'])} 来自单一市场因子**，特质部分仅 "
      f"{pct(fac['idio_share'])}，与第 2 节「DR 只达到零相关上界的 "
      f"{pct(dcp['diversification_realized_ratio'], 0)}」互相印证：这样的多头分散组合本质上是一个 "
      f"**beta≈{num(fac['portfolio_exposure'], 2)} 的市场敞口**。单因子 R² 中位数 "
      f"{num(fac['r2_median'], 3)}（即等权市场一个因子就解释了个股约 "
      f"{pct(fac['r2_median'], 0)} 的方差）；特质风险贡献（占组合**总方差**的比例）最大的是 "
      + "、".join(f"`{k}`（{pct(v, 2)}）"
                  for k, v in list(fac["top_idio_contributors"].items())[:3])
      + "。模型隐含波动与样本波动之比 "
      f"{num(fac['model_vs_sample_vol_ratio'], 3)}，说明单因子结构已能解释组合波动的绝大部分。")
    A("")
    A("> 口径说明：市场因子由**同一篮子标的**等权构成（组合自身也含在因子内），因此是"
      "「代理市场」而非独立基准；若改用 `--scheme equal`，组合与因子重合，beta 恒为 1、"
      "特质恒为 0，分解退化——这正是默认用逆波动权重的原因。")
    A("")

    # ---------------- 4. 回撤 ----------------
    A("## 4. 回撤分析（真实历史水下曲线）")
    A("")
    A("| 指标 | 数值 |")
    A("|---|---|")
    A(f"| **最大回撤 MDD** | **{pct(dd['max_drawdown'])}** |")
    A(f"| 当前回撤 | {pct(dd['current_drawdown'])} |")
    A(f"| 最长水下持续期 | {dd['longest_drawdown_periods']} 个交易日"
      f"（≈{dd['longest_drawdown_periods'] / meta['periods_per_year']:.1f} 年） |")
    A(f"| 水下时间占比 | {pct(dd['time_underwater_ratio'])} |")
    A(f"| Calmar（年化收益 / MDD） | {num(dd['calmar_ratio'], 4)} |")
    A(f"| 回撤区间数 | {dd['n_episodes']} |")
    A(f"| 期末净值 / 历史峰值净值 | {num(dd['wealth_final'], 4)} / {num(dd['wealth_peak'], 4)} |")
    A("")
    we = dd["worst_episode"]
    if we:
        A(f"最深回撤区间：峰值 **{we['peak_date']}** → 谷底 **{we['trough_date']}**"
          f"（下跌 {we['down_periods']} 个交易日，深度 {pct(we['depth'])}）→ "
          + (f"**{we['recovery_date']}** 收复（恢复期 {int(we['recovery_periods'])} 个交易日，"
             f"水下共 {we['total_periods']} 个交易日）。" if we["recovered"]
             else f"至样本末端**仍未收复**（已水下 {we['total_periods']} 个交易日）。"))
        A("")
    A(f"**Top{len(dd['top_episodes'])} 深度回撤区间**")
    A("")
    A(T["drawdown_episodes"])
    A("")
    A("**回撤持续期分布**（全部区间；`recovery_periods` 不含未收复区间）")
    A("")
    A(T["drawdown_duration"])
    A("")
    A(f"解读：真实历史上该组合最大回撤 **{pct(dd['max_drawdown'])}**，"
      f"且 {pct(dd['time_underwater_ratio'])} 的时间处于水下——**回撤是常态而非例外**；"
      f"Calmar {num(dd['calmar_ratio'], 2)} 意味着每承担 1 单位最大回撤换来约 "
      f"{num(dd['calmar_ratio'], 2)} 单位年化收益。最长水下 "
      f"{dd['longest_drawdown_periods']} 个交易日（≈"
      f"{dd['longest_drawdown_periods'] / meta['periods_per_year']:.1f} 年）是比年化收益更硬的"
      "约束：它决定了资金的**期限匹配**与持有体验。月末水下曲线见 `drawdown.csv`。")
    A("")

    # ---------------- 5. 压力测试 ----------------
    A("## 5. 压力测试（假设情景 / 历史重放 / 反向压力）")
    A("")
    A(f"组合市值 {money(st['portfolio_value'])}，beta 取自第 3 节的真实市场模型估计。")
    A("")
    A("**① 假设情景**（`Scenario` + `apply_scenario` / `scenario_table`，一阶线性近似；"
      f"市场 {pct(st['market_shock'], 0)} 且波动×{num(st['vol_multiplier'], 2)}；"
      f"个股情景冲击风险贡献最大的 `{st['idio_target_asset']}` {pct(st['idio_shock'], 0)}；"
      f"历史情景重放最差单日 {st['worst_history_day']}）")
    A("")
    A(T["scenarios"])
    A("")
    A(f"解读：市场整体下跌 {abs(st['market_shock']):.0%}（按真实 beta 传导 + 波动放大 "
      f"{num(st['vol_multiplier'], 1)} 倍）会让组合损失 **{money(-mk['pnl'])}**"
      f"（损益率 {pct(mk['pnl_pct'])}），市值从 {money(st['portfolio_value'])} "
      f"跌到 {money(mk['value_after'])}；冲击后 {meta['confidence']:.0%} 单日 VaR 升至 "
      f"{pct(mk['var_stressed'])}（≈ {money(mk['var_stressed'] * st['portfolio_value'])}），"
      "即「暴跌 + 波动放大」之后，次日再亏一个 VaR 的量级仍在这个数附近——"
      "**压力情景与日常 VaR 必须叠加看**。")
    A("")
    A("**② 历史重放**（`historical_replay`：把真实历史上组合最差的窗口重放到当前权重）")
    A("")
    A(f"单日最差 {len(r1['hit_dates'])} 天（共 {r1['n_windows']} 个可选窗口）：")
    A("")
    A(T["replay_1d"])
    A("")
    A(f"{rw['window']} 个交易日的**非重叠**最差窗口（共 {rw['n_windows']} 个可选窗口）：")
    A("")
    A(T["replay_window"])
    A("")
    ratio_var = ((-r1["worst_pnl"] / st["portfolio_value"]) / v99["historical"]
                 if v99["historical"] else float("nan"))
    A(f"解读：历史上最惨的单日重放会亏 **{money(-r1['worst_pnl'])}**"
      f"（{pct(r1['hit_returns'][0])}，{r1['hit_dates'][0]}），"
      f"最惨的 {rw['window']} 日窗口亏 **{money(-rw['worst_pnl'])}**"
      f"（{pct(rw['hit_returns'][0])}，截至 {rw['hit_dates'][0]}）。单日最差损失约为 "
      f"{meta['confidence99']:.0%} 历史 VaR 的 **{num(ratio_var, 2)} 倍**——"
      "**VaR 不是损失上界**，这正是需要压力测试与 EVT 外推的原因。")
    A("")
    A(f"**③ 反向压力测试**（`reverse_stress_test`：目标损失 {money(st['reverse_target_loss'])}"
      f" = 市值的 {pct(st['reverse_target_pct'], 0)}，情景=市场 {pct(st['market_shock'], 0)}"
      f" + 波动×{num(st['vol_multiplier'], 2)}，二分搜索冲击倍数 λ）")
    A("")
    A("| 损失口径 | 所需 λ | 等价市场跌幅 | 达成损失 | 收敛 | 迭代次数 |")
    A("|---|---|---|---|---|---|")
    for key in ("reverse_pnl", "reverse_var"):
        rv = st[key]
        A(f"| {rv['metric']} | {num(rv['multiplier'], 4)} | "
          f"{pct(rv['implied_market_shock'], 1)} | {money(rv['achieved_loss'])} | "
          f"{'是' if rv['converged'] else '否'} | {rv['iterations']} |")
    A("")
    A(f"解读：要亏掉 {money(st['reverse_target_loss'])}，线性损益口径下市场需下跌 "
      f"{pct(-st['reverse_pnl']['implied_market_shock'], 1)}"
      f"（λ={num(st['reverse_pnl']['multiplier'], 2)}）；若按「冲击后 "
      f"{meta['confidence']:.0%} VaR」口径，只需 λ={num(st['reverse_var']['multiplier'], 2)}"
      f"（市场跌 {pct(-st['reverse_var']['implied_market_shock'], 1)}），"
      "因为波动放大项已提前把损失推过目标。**同一目标损失在不同风险口径下的『距离』差别很大**，"
      "风控上应同时盯两条。")
    A("")

    # ---------------- 6. 尾部 ----------------
    A("## 6. EVT 极值尾部（真实肥尾）")
    A("")
    A("| 指标 | 数值 |")
    A("|---|---|")
    A(f"| **Hill 尾指数 ξ̂** | **{num(tl['hill_xi'], 4)}** ± {num(tl['hill_se'], 4)}"
      f"（k={tl['hill_k']}，超出 {tl['hill_n_exceed']} 个 / 共 {tl['hill_n_obs']} 个观测） |")
    A(f"| Hill 阈值 X_(k+1) | {num(tl['hill_threshold'], 5)} |")
    A(f"| 帕累托尾指数 α = 1/ξ | {num(tl['hill_tail_index_alpha'], 3)} |")
    A(f"| ξ̂ 在 k∈[{tl['hill_path_k_min']}, {tl['hill_path_k_max']}] 上的范围 | "
      f"{num(tl['hill_path_xi_min'], 3)} ~ {num(tl['hill_path_xi_max'], 3)}"
      f"（中位 {num(tl['hill_path_xi_median'], 3)}） |")
    A(f"| GPD 拟合方法 | {tl['gpd_method']} |")
    A(f"| GPD ξ / β | {num(tl['gpd_xi'], 4)} / {num(tl['gpd_scale'], 5)} |")
    A(f"| GPD 阈值 u / 超出数 Nu / 超出概率 ζ | {num(tl['gpd_threshold'], 5)} / "
      f"{tl['gpd_n_exceed']} / {num(tl['gpd_exceedance_prob'], 4)} |")
    A(f"| 平均超出 E[X−u&#124;X>u] | {num(tl['gpd_mean_excess'], 5)} |")
    A(f"| 尾部比率 ES/VaR（{meta['confidence']:.0%}） | {num(tl['tail_ratio_95'], 4)}"
      "（正态约 1.25） |")
    A("")
    A("**POT/GPD 外推 vs 经验口径**（每期损失率；最后一列为 POT VaR 折算金额）")
    A("")
    A(T["tail"])
    A("")
    pot_vs_hist99 = ("高于" if (p99["pot_var"] or 0) >= (p99["historical_var"] or 0)
                     else "略低于")
    A(f"解读：Hill ξ̂ = {num(tl['hill_xi'], 3)} > 0，且在 k 从 {tl['hill_path_k_min']} 到 "
      f"{tl['hill_path_k_max']} 的整个网格上保持正号（{num(tl['hill_path_xi_min'], 2)} ~ "
      f"{num(tl['hill_path_xi_max'], 2)}），对应帕累托尾指数 α ≈ "
      f"{num(tl['hill_tail_index_alpha'], 1)}——**真实 A 股组合收益确为肥尾**"
      f"（尾部衰减远慢于正态）；GPD 拟合的 ξ = {num(tl['gpd_xi'], 3)} 与 Hill 同号同量级，"
      "两条独立路径互为印证。")
    A("")
    A(f"在 {meta['confidence99']:.0%} 档，POT VaR = {pct(p99['pot_var'])} "
      f"{pot_vs_hist99}历史法（{pct(p99['historical_var'])}）：GPD 是对 "
      f"{tl['gpd_n_exceed']} 个超出量做的**平滑参数拟合**，与「第 "
      f"{int(round(meta['n_obs'] * (1 - meta['confidence99'])))} 差的顺序统计量」互有高低属正常；"
      f"但它同时远高于正态法（{pct(p99['parametric_normal_var'])}，"
      f"{num(p99['pot_var'] / p99['parametric_normal_var'], 2)} 倍）。"
      f"外推到样本外的 {meta['confidence999']:.1%} 档（{meta['n_obs']} 期里平均只有 "
      f"{meta['n_obs'] * (1 - meta['confidence999']):.1f} 天，历史法已几乎无观测可用）："
      f"POT VaR = {pct(p999['pot_var'])}（≈ {money(p999['pot_var_in_money'])}），"
      f"是正态法（{pct(p999['parametric_normal_var'])}）的 "
      f"**{num(p999['pot_var'] / p999['parametric_normal_var'], 1)} 倍**、历史法"
      f"（{pct(p999['historical_var'])}）的 "
      f"{num(p999['pot_var'] / p999['historical_var'], 2)} 倍；对应 POT ES = "
      f"{pct(p999['pot_es'])}（≈ {money(p999['pot_es_in_money'])}）。"
      "**越往尾部走，正态法越低估风险，而历史法受限于样本长度——这正是 EVT 的价值所在**。")
    A("")

    # ---------------- 7. 回测风险 ----------------
    A("## 7. 回测风险与防过拟合（PSR / DSR）")
    A("")
    A("| 指标 | 数值 |")
    A("|---|---|")
    A(f"| 每期夏普 SR̂ / 年化夏普 | {num(bt['sharpe_per_period'], 5)} / "
      f"{num(bt['sharpe_annualized'], 4)} |")
    A(f"| **PSR**（真实夏普 > 0 的概率） | **{num(bt['psr'], 4)}** |")
    A(f"| 最小跟踪长度（使 PSR≥95%） | {num(bt['min_track_length_95'], 0)} 期"
      f"（实际 {bt['n_obs']} 期） |")
    A(f"| 维持 DSR≥95% 的试验预算 | N ≤ {bt['trials_budget_for_dsr95']} |")
    A(f"| 胜率（Wilson {pct(bt['win_rate_ci']['confidence'], 0)} CI） | {pct(bt['win_rate'])} "
      f"[{pct(bt['win_rate_ci']['lower'])}, {pct(bt['win_rate_ci']['upper'])}] |")
    A(f"| 单期亏损概率 | {pct(bt['loss_probability'])} |")
    A(f"| 21 / 252 日内重演历史最差单日的概率 | {pct(bt['probability_of_min_loss_21d'])} / "
      f"{pct(bt['probability_of_min_loss_252d'])} |")
    A("")
    A("**DSR 随试验次数 N 的变化**（N=1 表示不做多重检验修正；"
      f"N={meta['n_assets']} 相当于「在这篮子标的里挑出这一只组合」；"
      f"N={meta['n_trials']} 相当于更大范围的策略搜索）")
    A("")
    A(T["dsr"])
    A("")
    d1 = bt["dsr"]["1"]
    dn = bt["dsr"][str(int(meta["n_trials"]))]
    da = bt["dsr"][str(int(meta["n_assets"]))]
    A(f"解读：该静态组合年化夏普 {num(bt['sharpe_annualized'], 2)}、样本 {bt['n_obs']} 期，"
      f"PSR = {num(bt['psr'], 3)}——**不做多重检验修正时，跑赢零夏普的概率很高**；"
      f"但一旦承认它是在 {meta['n_assets']} 只标的中挑出来的（N={meta['n_assets']}），"
      f"DSR 降到 {num(da['dsr'], 3)}；若搜索预算达 N={meta['n_trials']}，DSR 进一步降到 "
      f"{num(dn['dsr'], 3)}（门槛 SR₀ 从 {num(d1['sr0'], 4)} 抬到 {num(dn['sr0'], 4)}，"
      f"缩水 {num(dn['haircut'], 4)}）。**这就是回测过拟合的代价**：同样的历史夏普，"
      "试验次数越多、可信度越低；本报告的组合是逆波动权重（几乎无参数搜索），"
      "因此 N=1 与 N=标的数 两档更接近其真实显著性。")
    A("")

    # ---------------- 8. 校验 ----------------
    A("## 8. 一致性校验（脚本自证）")
    A("")
    labels = {
        "euler_component_sum_equals_vol": "欧拉可加：Σ 成分风险 = 组合波动 σp",
        "component_es_sum_equals_es": "尾部可加：Σ 成分 ES = 组合 ES",
        "component_es_matches_portfolio_es": "成分 ES 口径 = 组合历史 ES",
        "component_var_sum_equals_var": "成分 VaR 缩放后可加 = 组合 VaR",
        "factor_plus_idio_equals_total": "因子方差 + 特质方差 = 总方差",
        "model_vol_close_to_sample_vol": "因子模型隐含波动 ≈ 样本波动（±25%）",
        "es_ge_var_95": "ES(95%) ≥ VaR(95%)",
        "es_ge_var_99": "ES(99%) ≥ VaR(99%)",
        "var99_ge_var95": "VaR(99%) ≥ VaR(95%)（分位单调）",
        "student_t_var99_ge_normal_var99": "学生 t VaR(99%) ≥ 正态 VaR(99%)（肥尾更保守）",
        "pot_var99_above_gpd_threshold": "POT VaR(99%) > GPD 阈值 u（外推方向正确）",
        "pot_es99_ge_pot_var99": "POT ES(99%) ≥ POT VaR(99%)",
        "pot_es999_ge_pot_var999": "POT ES(99.9%) ≥ POT VaR(99.9%)",
        "pot_var999_ge_pot_var99": "POT VaR(99.9%) ≥ POT VaR(99%)（置信度单调）",
        "pot_var999_ge_var95_historical": "POT VaR(99.9%) ≥ 历史 VaR(95%)",
        "diversification_ratio_ge_one": "分散化比率 DR ≥ 1",
        "risk_effective_n_within_assets": "风险有效资产数落在 [1, 资产数]",
        "weights_sum_to_one": "组合权重之和 = 1",
        "weights_all_positive": "权重全为正（多头组合）",
        "all_returns_within_price_limit": "全部收益落在涨跌停物理约束内（数据已清洗）",
        "dsr_le_psr": "DSR ≤ PSR（多重检验只会降低显著性）",
        "dsr_monotone_in_trials": "DSR 随试验次数单调不增",
        "hill_xi_positive_fat_tail": "Hill ξ > 0（肥尾）",
        "drawdown_within_unit": "最大回撤落在 [0, 1]",
        "json_strictly_valid": "JSON 产物严格合法（无 NaN/Inf 字面量）",
    }
    A("| 校验项 | 结果 |")
    A("|---|---|")
    for key, label in labels.items():
        A(f"| {label} | {'通过' if M['checks'].get(key) else '**失败**'} |")
    A("")
    A(f"关键误差量级：欧拉可加 {dcp['additivity_abs_error']:.3e}、"
      f"成分 ES {dcp['component_es_additivity_error']:.3e}、"
      f"成分 VaR {dcp['component_var_additivity_error']:.3e}、"
      f"因子分解 {fac['factor_additivity_error']:.3e}。")
    A("")

    # ---------------- 9. 结论 ----------------
    A("## 9. 结论速览")
    A("")
    A(f"1. **风险水平**：{meta['n_assets']} 只 A 股的 {meta['scheme']} 组合在 "
      f"{meta['sample_start']} ~ {meta['sample_end']} 的年化波动 "
      f"{pct(m['annualized_volatility'])}、年化收益 {pct(m['annualized_return'])}；"
      f"{meta['confidence']:.0%} 单日历史 VaR {pct(v95['historical'])}"
      f"（{money(v95['historical'] * meta['portfolio_value'])}），"
      f"{meta['confidence99']:.0%} ES {pct(e99['historical'])}"
      f"（{money(e99['historical'] * meta['portfolio_value'])}）。")
    A(f"2. **分散化被相关性吃掉**：逆波动权重把风险贡献拉平（风险有效 "
      f"{num(dcp['risk_effective_n'], 1)} / 名义 {meta['n_assets']} 只），但 DR 只有 "
      f"{num(dcp['diversification_ratio'], 2)}（零相关上界 √n = "
      f"{num(dcp['diversification_ratio_independent_bound'], 2)}）；"
      f"{pct(fac['factor_share'])} 的方差来自单一市场因子（组合 beta "
      f"{num(fac['portfolio_exposure'], 2)}）——多头分散本质仍是市场敞口。")
    A(f"3. **回撤是主约束**：MDD {pct(dd['max_drawdown'])}、水下时间 "
      f"{pct(dd['time_underwater_ratio'])}、最长水下 {dd['longest_drawdown_periods']} 个交易日、"
      f"Calmar {num(dd['calmar_ratio'], 2)}。")
    A(f"4. **尾部厚于正态**：Hill ξ={num(tl['hill_xi'], 3)}、GPD ξ={num(tl['gpd_xi'], 3)}，"
      f"{meta['confidence999']:.1%} POT VaR {pct(p999['pot_var'])}"
      f"（{money(p999['pot_var_in_money'])}）= 正态法的 "
      f"{num(p999['pot_var'] / p999['parametric_normal_var'], 1)} 倍。")
    A(f"5. **压力视角**：市场 {pct(st['market_shock'], 0)} 情景亏 {money(-mk['pnl'])}；"
      f"历史最差单日重放亏 {money(-r1['worst_pnl'])}（= {meta['confidence99']:.0%} VaR 的 "
      f"{num(ratio_var, 1)} 倍）；亏 {pct(st['reverse_target_pct'], 0)} 市值需市场跌 "
      f"{pct(-st['reverse_pnl']['implied_market_shock'], 1)}。")
    A(f"6. **显著性**：PSR {num(bt['psr'], 3)}，但在 N={meta['n_trials']} 的搜索预算下 "
      f"DSR 仅 {num(dn['dsr'], 3)}——真实历史收益的可信度必须按多重检验打折。")
    A("")
    A("### 数据声明")
    A("")
    A("- 数据为 `kairos-data` 仓库已提交的**真实 A 股前复权日线**（来源：腾讯/新浪公开行情接口，"
      "抓取代码见该仓库 `kairos_data/ashare.py`），本脚本**只读、不联网、不改写**原始文件。")
    A("- 数据版权归原作者/数据源所有，仅在研究、学习、演示目的下少量使用，不用于任何商业用途，"
      "不主张对数据本身的所有权；数据**不保证准确、完整或及时**，前复权口径以数据源为准"
      "（第 0 节已量化其偏差并做保守修复）。")
    A("- 本报告全部数字由 `kairos_risk` 的公开 API 在真实数据上计算，可离线复现；"
      "**不构成任何投资建议**。")
    A("- 产物：`REPORT.md`（本文件）、`risk_metrics.json`（全部指标，严格合法 JSON）、"
      "`drawdown.csv`（月末水下曲线）。")
    A("")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# 产物落盘与打印
# --------------------------------------------------------------------------- #
def write_outputs(out_dir: str, M: Dict[str, Any]) -> Dict[str, str]:
    """写出 REPORT.md / risk_metrics.json / drawdown.csv，返回文件路径字典。"""
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "report": os.path.join(out_dir, "REPORT.md"),
        "metrics": os.path.join(out_dir, "risk_metrics.json"),
        "drawdown": os.path.join(out_dir, "drawdown.csv"),
    }
    with open(paths["report"], "w", encoding="utf-8") as fh:
        fh.write(render_report(M))
    payload = json_safe({k: v for k, v in M.items() if not k.startswith("_")})
    with open(paths["metrics"], "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, allow_nan=False)
        fh.write("\n")
    monthly: pd.DataFrame = M["_monthly_underwater"].copy()
    monthly.index.name = "month_end"
    monthly["in_drawdown"] = monthly["in_drawdown"].astype(int)
    monthly.round(6).to_csv(paths["drawdown"], index=True, encoding="utf-8")
    return paths


def print_summary(M: Dict[str, Any]) -> None:
    """把关键风险数字打印到 stdout（脚本可直接跑通并自检）。"""
    meta, m, dcp, fac = M["meta"], M["measures"], M["decomposition"], M["factor"]
    dd, st, tl, bt = M["drawdown"], M["stress"], M["tail"], M["backtest"]
    v95, v99 = m["var"]["95"], m["var"]["99"]
    e95, e99 = m["es"]["95"], m["es"]["99"]
    p999 = tl["pot"]["999"]
    mk = st["scenario_table"]["market_crash"]
    line = "=" * 78
    print(line)
    print(f"真实数据组合风险报告 · {meta['sample_start']} ~ {meta['sample_end']} · "
          f"{meta['n_assets']} 标的 × {meta['n_obs']} 日 · 方案 {meta['scheme']} · "
          f"市值 {money(meta['portfolio_value'])}")
    print(line)
    repair, sani = M["data"]["repair"], M["data"]["sanitize"]
    print(f"[数据] 等差前复权修复：涨跌停越界 {repair['violations_before']} -> "
          f"{repair['violations_after']}；剔除压缩过度标的 "
          f"{repair['n_assets_in'] - repair['n_assets_out']} 只"
          f"（{'、'.join(repair['dropped_symbols']) or '无'}）；残余越界 {sani['bad_cells']} 个按 "
          f"{sani['on_bad']} 处理")
    print(f"[度量] 年化收益 {pct(m['annualized_return'])}  年化波动 "
          f"{pct(m['annualized_volatility'])}  夏普 {num(m['sharpe'], 3)}  "
          f"偏度 {num(m['skewness'], 3)}  超额峰度 {num(m['excess_kurtosis'], 3)}  "
          f"下行标准差(年化) {pct(m['downside_deviation_annualized'])}")
    print(f"[VaR ] {meta['confidence']:.0%}: 历史 {pct(v95['historical'])} | 正态 "
          f"{pct(v95['parametric_normal'])} | 学生t {pct(v95['parametric_student_t'])} | "
          f"CF修正 {pct(v95['modified_cornish_fisher'])}")
    print(f"[VaR ] {meta['confidence99']:.0%}: 历史 {pct(v99['historical'])} | 正态 "
          f"{pct(v99['parametric_normal'])} | 学生t {pct(v99['parametric_student_t'])} | "
          f"CF修正 {pct(v99['modified_cornish_fisher'])}")
    print(f"[ES  ] {meta['confidence']:.0%}: 历史 {pct(e95['historical'])} 正态 "
          f"{pct(e95['normal'])} 学生t {pct(e95['student_t'])} | "
          f"{meta['confidence99']:.0%}: 历史 {pct(e99['historical'])} 正态 {pct(e99['normal'])} "
          f"学生t {pct(e99['student_t'])}")
    print(f"[金额] {meta['confidence99']:.0%} 单日历史 VaR ≈ "
          f"{money(v99['historical'] * meta['portfolio_value'])}，ES ≈ "
          f"{money(e99['historical'] * meta['portfolio_value'])}")
    print(f"[成分] σp={num(dcp['portfolio_volatility_daily'], 6)}  Σ CRC="
          f"{num(dcp['component_sum'], 6)}（误差 {dcp['additivity_abs_error']:.2e}）  "
          f"DR={num(dcp['diversification_ratio'], 3)}  风险有效N="
          f"{num(dcp['risk_effective_n'], 2)}  Top1={list(dcp['top_contributors'])[0]}"
          f"({pct(list(dcp['top_contributors'].values())[0], 1)})")
    print(f"[因子] 市场暴露 {num(fac['portfolio_exposure'], 3)}  系统性占比 "
          f"{pct(fac['factor_share'])}  特质占比 {pct(fac['idio_share'])}  "
          f"R²中位 {num(fac['r2_median'], 3)}  beta中位 {num(fac['beta_median'], 3)}")
    print(f"[回撤] MDD {pct(dd['max_drawdown'])}  当前 {pct(dd['current_drawdown'])}  "
          f"最长水下 {dd['longest_drawdown_periods']} 日  水下占比 "
          f"{pct(dd['time_underwater_ratio'])}  Calmar {num(dd['calmar_ratio'], 3)}  "
          f"区间数 {dd['n_episodes']}")
    print(f"[压测] 市场{pct(st['market_shock'], 0)}情景损益 {money(mk['pnl'])}"
          f"（冲击后VaR{meta['confidence']:.0%} {pct(mk['var_stressed'])}）  "
          f"历史最差单日 {money(st['replay_1d']['worst_pnl'])}"
          f"（{st['replay_1d']['hit_dates'][0]}）  最差{st['replay_multi_day']['window']}日 "
          f"{money(st['replay_multi_day']['worst_pnl'])}")
    print(f"[反压] 目标 {money(st['reverse_target_loss'])}（市值的 "
          f"{pct(st['reverse_target_pct'], 0)}）：pnl 口径 λ="
          f"{num(st['reverse_pnl']['multiplier'], 3)}（等价市场跌 "
          f"{pct(-st['reverse_pnl']['implied_market_shock'], 1)}），VaR 口径 λ="
          f"{num(st['reverse_var']['multiplier'], 3)}")
    print(f"[尾部] Hill ξ={num(tl['hill_xi'], 4)}±{num(tl['hill_se'], 4)} (k={tl['hill_k']})  "
          f"GPD[{tl['gpd_method']}] ξ={num(tl['gpd_xi'], 4)} β={num(tl['gpd_scale'], 5)} "
          f"u={num(tl['gpd_threshold'], 4)} ζ={num(tl['gpd_exceedance_prob'], 4)}  "
          f"尾部比率 {num(tl['tail_ratio_95'], 3)}")
    print(f"[尾部] POT VaR {meta['confidence99']:.0%}={pct(tl['pot']['99']['pot_var'])} "
          f"{meta['confidence999']:.1%}={pct(p999['pot_var'])}"
          f"（≈{money(p999['pot_var_in_money'])}）  POT ES "
          f"{meta['confidence999']:.1%}={pct(p999['pot_es'])}（≈{money(p999['pot_es_in_money'])}）")
    print(f"[回测] 每期夏普 {num(bt['sharpe_per_period'], 5)}  PSR={num(bt['psr'], 4)}  "
          f"DSR(N=1)={num(bt['dsr']['1']['dsr'], 4)}  "
          f"DSR(N={meta['n_assets']})={num(bt['dsr'][str(meta['n_assets'])]['dsr'], 4)}  "
          f"DSR(N={meta['n_trials']})={num(bt['dsr'][str(meta['n_trials'])]['dsr'], 4)}  "
          f"胜率 {pct(bt['win_rate'])}")
    ok = sum(1 for v in M["checks"].values() if v)
    print(f"[校验] {ok}/{len(M['checks'])} 项通过"
          + ("" if ok == len(M["checks"])
             else f"；未通过：{[k for k, v in M['checks'].items() if not v]}"))
    print(line)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """解析命令行参数（全部有默认值，直接运行即可复现报告）。"""
    ap = argparse.ArgumentParser(
        description="用真实 A 股日线生成 kairos-risk 的完整组合风险报告（离线、只读）")
    ap.add_argument("--data-dir", default=None,
                    help="真实行情 CSV 目录（默认 <kairos-data>/data/ashare）")
    ap.add_argument("--out", default=DEFAULT_OUT, help="产物目录（默认 research/real_risk）")
    ap.add_argument("--scheme", default="inverse_vol", choices=list(WEIGHT_SCHEMES),
                    help="组合权重方案：inverse_vol（默认，逆波动）或 equal（等权）")
    ap.add_argument("--value", type=float, default=1e8, help="组合市值（元），默认 1 亿")
    ap.add_argument("--start", default=None, help="样本起始日期（含）")
    ap.add_argument("--end", default=None, help="样本结束日期（含）")
    ap.add_argument("--periods-per-year", type=float, default=252.0, help="年化期数，默认 252")
    ap.add_argument("--confidence", type=float, default=0.95, help="VaR/ES 主置信度，默认 0.95")
    ap.add_argument("--confidence99", type=float, default=0.99, help="尾部置信度，默认 0.99")
    ap.add_argument("--confidence999", type=float, default=0.999, help="极尾置信度，默认 0.999")
    ap.add_argument("--tol", type=float, default=0.01, help="涨跌停判定容差，默认 0.01")
    ap.add_argument("--max-compression", type=float, default=3.0,
                    help="复权压缩倍数闸门，超过则整只剔除该标的，默认 3.0")
    ap.add_argument("--max-bad-days", type=int, default=10,
                    help="残余越界天数上限，超过则剔除该标的，默认 10")
    ap.add_argument("--on-bad", default="clip", choices=("clip", "zero", "drop"),
                    help="残余越界观测的处理方式，默认 clip（取涨跌停边界）")
    ap.add_argument("--market-shock", type=float, default=-0.20, help="情景：市场冲击，默认 -0.20")
    ap.add_argument("--vol-multiplier", type=float, default=1.8, help="情景：波动放大倍数，默认 1.8")
    ap.add_argument("--idio-shock", type=float, default=-0.35, help="情景：个股冲击，默认 -0.35")
    ap.add_argument("--reverse-target-pct", type=float, default=0.10,
                    help="反向压力目标损失占市值比例，默认 0.10")
    ap.add_argument("--n-worst", type=int, default=5, help="历史重放取最差窗口数，默认 5")
    ap.add_argument("--replay-window", type=int, default=21,
                    help="多日重放窗口长度（交易日），默认 21")
    ap.add_argument("--top-drawdowns", type=int, default=6, help="报告展示的回撤区间数，默认 6")
    ap.add_argument("--hill-k", type=int, default=None, help="Hill 估计的 k（默认按尾部分位自动）")
    ap.add_argument("--hill-tail-quantile", type=float, default=0.90,
                    help="Hill 自动选 k 的尾部分位，默认 0.90")
    ap.add_argument("--hill-k-min", type=int, default=20, help="Hill 路径图的最小 k，默认 20")
    ap.add_argument("--tail-quantile", type=float, default=0.95, help="POT/GPD 阈值分位，默认 0.95")
    ap.add_argument("--n-trials", type=int, default=100, help="DSR 的试验次数上限，默认 100")
    ap.add_argument("--no-write", action="store_true", help="只打印不写文件（自检用）")
    args = ap.parse_args(argv)
    if args.data_dir is None:
        args.data_dir = rd.default_data_dir("ashare")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    """入口：加载真实数据 → 复权口径修复 → 构造组合 → 全链路风险计算 → 落盘 + 打印。"""
    cfg = parse_args(argv)
    if not rd.has_local_data(cfg.data_dir):
        print(f"[错误] 未找到真实行情 CSV：{cfg.data_dir}\n"
              "       请用 --data-dir 指定 <kairos-data>/data/ashare（本脚本不联网）。",
              file=sys.stderr)
        return 2

    prices_raw = rd.load_close_panel(cfg.data_dir, drop_incomplete=True,
                                     start=cfg.start, end=cfg.end)
    prices, repair = rd.repair_close_panel(prices_raw, tol=cfg.tol,
                                           max_compression=cfg.max_compression)
    returns, sani = rd.sanitize_returns(rd.simple_returns(prices), tol=cfg.tol,
                                        max_bad_days=cfg.max_bad_days, on_bad=cfg.on_bad)
    returns = returns.dropna(how="any")
    if returns.shape[0] < 120 or returns.shape[1] < 3:
        print(f"[错误] 清洗后样本过小（{returns.shape}），无法做风险分解", file=sys.stderr)
        return 3
    summary = rd.panel_summary(prices, returns, periods_per_year=cfg.periods_per_year)
    data_diag = {
        "n_files": int(prices_raw.attrs.get("n_files", prices_raw.shape[1])),
        "panel_shape": [int(prices_raw.shape[0]), int(prices_raw.shape[1])],
        "panel_window": list(prices_raw.attrs.get("window", ("", ""))),
        "summary": summary,
        "repair": repair,
        "sanitize": sani,
        "note": "等差前复权 -> 涨跌停硬约束下的下界修复 -> 残余越界按 on_bad 处理",
    }

    weights = build_weights(returns, cfg.scheme)
    M = collect(returns, prices_raw, weights, cfg, data_diag)
    print_summary(M)
    if not cfg.no_write:
        paths = write_outputs(cfg.out, M)
        print("[产物] " + "  ".join(f"{k}={v}" for k, v in paths.items()))
        print(f"[产物] drawdown.csv {os.path.getsize(paths['drawdown']) / 1024:.1f} KB"
              f"（{len(M['_monthly_underwater'])} 行月末水下曲线）")
    failed = [k for k, v in M["checks"].items() if not v]
    if failed:
        print(f"[警告] 以下一致性校验未通过：{failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
