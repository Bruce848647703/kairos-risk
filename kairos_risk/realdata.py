"""真实行情数据接入模块：把本地 CSV 日线目录读成风险分析可直接使用的收盘价面板。

用途
----
本库的度量/分解/压测/尾部函数都吃「收益面板」（DataFrame(index=期次, columns=资产)），
而真实数据通常以「一标的一文件」的 OHLCV CSV 存放。本模块负责这层落地转换：
:func:`load_close_panel` 读出对齐的收盘价面板，:func:`simple_returns` 转成收益面板，
:func:`panel_summary` 给出口径摘要（写进报告头部与 JSON 产物）。

数据清洗口径（真实数据的四个坑）
--------------------------------
1. **前复权可能为负**：长期高分红标的的前复权价在早期会变成 0 甚至负数，
   此时 ``pct_change`` 会产出 −100%、+1000% 这类**伪收益**，把波动/VaR/尾部全部污染。
   本模块把「非有限或非正」的价格一律记为缺失，并在 ``drop_incomplete=True`` 时
   把样本裁剪到「所有标的都已进入严格正价区间」的公共窗口（每标的取其**最后一个**
   非正价之后的第一行，再对全体取最大值），从根上排除零穿越。
2. **停牌/未上市造成的空洞**：公共窗口内的残余缺失用**前向填充**（当日无成交则沿用
   前收盘价，即当日收益 0），只用历史信息，不引入未来函数；各标的上市前的头部缺失
   在 ``drop_incomplete=False`` 时保留为 NaN。
3. **重复/乱序日期**：同一标的内重复日期保留最后一条，随后按日期升序排列。
4. **复权因子断裂 / 近零价噪声**：即使价格全为正，数据源在不同抓取分页上的前复权
   锚点不一致时会产生「单日跳变」；价格被压到接近 0 时，小数位四舍五入本身就会造成
   数十个百分点的伪波动。:func:`sanitize_returns` 用**交易所涨跌停幅度**作为物理约束
   识别并剔除这些观测（详见其文档），并把剔除明细写进诊断字典，绝不静默降级。

约定：仅依赖 numpy/pandas 与标准库，**不联网**；数据版权归原作者/数据源所有，
只用于研究与演示，任何基于真实数据的输出都不构成投资建议。
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

#: 收盘价面板的默认列/索引名约定
DATE_COL = "date"
PRICE_COL = "close"


def default_data_dir(dataset: str = "ashare") -> str:
    """返回同系列 ``kairos-data`` 仓库中真实行情数据集的默认目录（可能不存在）。

    按「兄弟仓库并排 checkout」的相对位置解析：
    ``<本仓库父目录>/kairos-data/data/<dataset>``，因此不写死任何绝对路径。
    """
    here = os.path.dirname(os.path.abspath(__file__))          # .../kairos_risk
    repo_root = os.path.dirname(here)                          # .../kairos-risk
    series_root = os.path.dirname(repo_root)                   # .../kairos
    return os.path.join(series_root, "kairos-data", "data", dataset)


def has_local_data(data_dir: Optional[str] = None, dataset: str = "ashare") -> bool:
    """判断数据目录是否存在且至少含一个 CSV（离线可用性探测）。"""
    d = data_dir or default_data_dir(dataset)
    if not os.path.isdir(d):
        return False
    return any(f.lower().endswith(".csv") for f in os.listdir(d))


def _read_one_close(path: str, date_col: str = DATE_COL,
                    price_col: str = PRICE_COL):
    """读单个 CSV 的收盘价序列：解析日期、去重排序、非正/非有限价记为缺失。

    返回 ``(clean_price, recorded)``：``clean_price`` 为已把「非正/非有限」价格置为
    NaN 的 Series；``recorded`` 为同索引的布尔 Series（该日在文件中**确有记录**）。
    区分二者很关键：文件中缺失的日期属于停牌/未上市（可前向填充），而「有记录但
    价格非正」属于前复权口径损坏（必须裁掉，否则 pct_change 出现零穿越伪收益）。
    """
    df = pd.read_csv(path)
    missing = [c for c in (date_col, price_col) if c not in df.columns]
    if missing:
        raise ValueError(f"{os.path.basename(path)} 缺少必需列 {missing}"
                         f"（实际列：{list(df.columns)}）")
    idx = pd.to_datetime(df[date_col], errors="coerce")
    px = pd.to_numeric(df[price_col], errors="coerce").to_numpy(dtype="float64")
    name = os.path.splitext(os.path.basename(path))[0]
    s = pd.Series(px, index=idx, name=name)
    s = s[s.index.notna()]                       # 丢弃无法解析的日期行
    s = s[~s.index.duplicated(keep="last")]      # 同日多条保留最后一条
    s = s.sort_index()
    arr = s.to_numpy(dtype="float64")
    clean = np.where(np.isfinite(arr) & (arr > 0.0), arr, np.nan)   # 前复权负价 -> 缺失
    return pd.Series(clean, index=s.index, name=name), pd.Series(True, index=s.index)


def _usable_start(price_col: np.ndarray, recorded_col: np.ndarray) -> int:
    """返回该标的的可用起点位置：max(最后一个非正价之后, 首个有记录的日期)。

    前复权负价集中在样本早期，取其**最后一次**出现的位置 +1，可保证之后的
    ``pct_change`` 不会因价格零穿越而产生伪收益；文件中缺失的日期（停牌/未上市）
    不算「口径损坏」，由前向填充处理，因此不参与该起点的判定。
    """
    broken = recorded_col & ~np.isfinite(price_col)
    start = 0
    if broken.any():
        start = int(price_col.size - int(np.argmax(broken[::-1])))
    first = np.flatnonzero(recorded_col)
    return int(max(start, first[0] if first.size else price_col.size))


def load_close_panel(data_dir: Optional[str] = None, drop_incomplete: bool = True,
                     start: Optional[str] = None, end: Optional[str] = None,
                     dataset: str = "ashare", date_col: str = DATE_COL,
                     price_col: str = PRICE_COL, ffill: bool = True) -> pd.DataFrame:
    """从 CSV 目录加载**对齐的收盘价面板**（index=DatetimeIndex, columns=标的代码）。

    参数
    ----
    data_dir:        CSV 目录；省略时用 :func:`default_data_dir`（同系列 kairos-data）。
    drop_incomplete: True（默认）时把样本裁剪到「全体标的均处于严格正价区间」的公共窗口，
                     得到无缺失的稠密面板，适合直接算收益与风险；
                     False 时保留全体日期的并集，各标的自身缺失按 ``ffill`` 前填，
                     上市/进入正价区间之前的头部保留 NaN。
    start / end:     可选日期窗口（含端点，字符串或 Timestamp），在裁剪之后生效。
    dataset:         仅在 ``data_dir`` 省略时用于推断默认目录。
    date_col / price_col: CSV 中的日期列与价格列名（默认 ``date`` / ``close``）。
    ffill:           是否对窗口内残余缺失（停牌）做前向填充，默认 True。

    返回面板的 ``attrs`` 附带口径诊断：``source_dir`` / ``n_files`` / ``symbols`` /
    ``masked_prices``（各标的被判为无效的价格条数）/ ``dropped_prices``（因裁剪而丢弃的
    有效价格条数）/ ``drop_incomplete`` / ``window``。
    """
    d = data_dir or default_data_dir(dataset)
    if not os.path.isdir(d):
        raise FileNotFoundError(
            f"数据目录不存在：{d}\n请通过 --data-dir 指定真实行情 CSV 目录"
            f"（形如 <kairos-data>/data/{dataset}，每标的一份 {date_col},...,{price_col},... 的 CSV）。"
        )
    files = sorted(f for f in os.listdir(d) if f.lower().endswith(".csv"))
    if not files:
        raise FileNotFoundError(f"{d} 下没有 CSV 文件，无法构造收盘价面板")

    series: Dict[str, pd.Series] = {}
    recorded: Dict[str, pd.Series] = {}
    masked: Dict[str, int] = {}
    for f in files:
        s, rec = _read_one_close(os.path.join(d, f), date_col=date_col, price_col=price_col)
        sym = str(s.name)
        if sym in series:
            raise ValueError(f"标的代码重复：{sym}（目录内存在同名 CSV）")
        if s.empty:
            raise ValueError(f"{f} 未解析出任何带日期的价格行")
        series[sym] = s
        recorded[sym] = rec
        masked[sym] = int(np.count_nonzero(~np.isfinite(s.to_numpy(dtype="float64"))))
    panel = pd.DataFrame(series).sort_index()
    if panel.shape[1] == 0:
        raise ValueError(f"{d} 下的 CSV 未解析出任何标的")
    # 有记录掩码：按并集日期对齐，缺失日期填 False（bool dtype，不产生 NaN）
    rec_panel = pd.DataFrame({
        sym: recorded[sym].reindex(panel.index, fill_value=False) for sym in panel.columns
    })
    raw_valid = int(np.count_nonzero(np.isfinite(panel.to_numpy(dtype="float64"))))

    arr = panel.to_numpy(dtype="float64")
    rec_arr = rec_panel.to_numpy()
    starts = [_usable_start(arr[:, j], rec_arr[:, j]) for j in range(arr.shape[1])]
    pos0 = max(starts) if drop_incomplete else 0
    if pos0 >= arr.shape[0]:
        raise ValueError("全体标的没有共同的有效价格区间，请检查数据或改用 drop_incomplete=False")
    panel = panel.iloc[pos0:]
    if ffill:
        panel = panel.ffill()
    if start is not None or end is not None:
        panel = panel.loc[start:end]
    panel = panel.dropna(axis=1, how="all").dropna(how="all")
    if panel.empty:
        raise ValueError("裁剪后收盘价面板为空，请放宽 start/end 或 drop_incomplete")

    kept_valid = int(np.count_nonzero(np.isfinite(panel.to_numpy(dtype="float64"))))
    panel.attrs.update({
        "source_dir": os.path.abspath(d),
        "n_files": len(files),
        "symbols": [str(c) for c in panel.columns],
        "masked_prices": masked,
        "dropped_prices": max(raw_valid - kept_valid, 0),
        "drop_incomplete": bool(drop_incomplete),
        "window": (str(pd.Timestamp(panel.index[0]).date()),
                   str(pd.Timestamp(panel.index[-1]).date())),
    })
    return panel


def simple_returns(prices: pd.DataFrame, drop_first: bool = True) -> pd.DataFrame:
    """收盘价面板 -> 每期**简单收益**面板 r_t = P_t/P_{t−1} − 1。

    ``drop_first=True``（默认）时剔除首行（无前收，收益无定义）；价格含缺失时
    对应收益为 NaN，交由下游 ``as_returns_frame`` 按行剔除。
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("simple_returns 需要收盘价 DataFrame 面板")
    out = prices.astype("float64").pct_change()
    return out.iloc[1:] if drop_first else out


def log_returns(prices: pd.DataFrame, drop_first: bool = True) -> pd.DataFrame:
    """收盘价面板 -> 每期**对数收益**面板 ln(P_t/P_{t−1})（可跨期可加）。"""
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("log_returns 需要收盘价 DataFrame 面板")
    out = np.log(prices.astype("float64")).diff()
    return out.iloc[1:] if drop_first else out


#: 板块日涨跌停幅度（研究口径）：创业板/科创板 20%，沪深主板 10%
LIMIT_GEM_STAR = 0.20
LIMIT_MAIN = 0.10
_GEM_STAR_PREFIX = ("sz300", "sz301", "sz302", "sh688", "sh689")


def daily_price_limit(symbol: object, tol: float = 0.01) -> float:
    """按代码前缀推断标的的**日涨跌停幅度**（加容差 ``tol``），用于识别口径断裂。

    A 股非 ST 个股的日涨跌幅有硬性限制：创业板 ``sz300/sz301`` 与科创板 ``sh688``
    为 ±20%，沪深主板为 ±10%。因此**超出该幅度的单日收益必定是数据问题**
    （前复权因子断裂、近零价的四舍五入噪声、复权口径切换等），而不是真实行情。
    ``tol`` 用于吸收价格四舍五入带来的边界误差。
    """
    s = str(symbol).lower()
    base = LIMIT_GEM_STAR if s.startswith(_GEM_STAR_PREFIX) else LIMIT_MAIN
    t = float(tol)
    if t < 0:
        raise ValueError("tol 不能为负")
    return base + t


def estimate_dividend_offset(prices: pd.DataFrame, tol: float = 0.01) -> pd.DataFrame:
    """估计「等差（减法）前复权」在每个标的、每个日期上的复权偏移量 D̂_t。

    背景
    ----
    部分公开行情源的前复权采用**减法口径** ``P_adj = P_raw − D_t``，其中
    ``D_t`` 为 t 日之后（不含 t）全部现金分红之和：越靠近样本末端 D 越小（末日为 0），
    越往早期 D 越大，高分红标的甚至会变成**负价**。此时直接 ``pct_change`` 得到的是
    被放大的伪收益：``r_adj = k_t·r_raw``，放大倍数 ``k_t = P_raw/(P_raw − D_t) ≥ 1``，
    对低价高分红标的可达 2 倍以上（波动、VaR、尾部指数全部被同比例放大）。

    估计量（用交易所涨跌停这一**硬约束**反解）
    -------------------------------------------
    真实收益满足 ``|r_raw| ≤ L``（L 为该板块涨跌停幅度 + ``tol``）。把
    ``P_raw = P_adj + D`` 代入 ``r_raw = (P_raw,t/P_raw,t−1) − 1`` 并解出 D，
    可得每个交易日的**必要条件**：

        上涨越界：D ≥ (P_t − (1+L)·P_{t−1}) / L
        下跌越界：D ≥ ((1−L)·P_{t−1} − P_t) / L

    对全体日期取 ``D̂_t = max_{s ≥ t} required_s``（反向累计最大值），既满足每日约束，
    又符合「D 随时间单调不增」的经济含义。可以证明 ``required_s ≤ D_s ≤ D_t``
    （等号仅在该日恰好涨停/跌停时成立），因此 **D̂ 是真实偏移的下界**：
    修复后的收益只会向真实口径靠拢，**不会把风险修小**（保守、可用于风控）。

    返回与 ``prices`` 同形状的非负偏移面板（末日附近趋于 0）。
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("estimate_dividend_offset 需要收盘价 DataFrame 面板")
    frame = prices.astype("float64")
    out: Dict[str, np.ndarray] = {}
    for col in frame.columns:
        limit = daily_price_limit(col, tol)
        a = frame[col].to_numpy(dtype="float64")
        prev, cur = a[:-1], a[1:]
        with np.errstate(invalid="ignore"):
            up = (cur - (1.0 + limit) * prev) / limit
            dn = ((1.0 - limit) * prev - cur) / limit
        req = np.zeros(a.size, dtype="float64")
        req[1:] = np.fmax(np.fmax(up, dn), 0.0)          # NaN（停牌缺口）视为无信息
        req = np.nan_to_num(req, nan=0.0, posinf=0.0, neginf=0.0)
        out[str(col)] = np.maximum.accumulate(req[::-1])[::-1]   # 单调不增包络
    return pd.DataFrame(out, index=frame.index)


def repair_close_panel(prices: pd.DataFrame, tol: float = 0.01,
                       max_compression: float = 3.0) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """把「等差前复权」价格面板修复回**限价一致**的价格口径，返回 ``(修复面板, 诊断)``。

    做法：``P_repaired = P_adj + D̂``，其中 D̂ 由 :func:`estimate_dividend_offset`
    用涨跌停硬约束反解得到。因为 D̂ 是真实偏移的**下界**，修复只会把被放大的伪收益
    往真实口径拉回，**不会把风险修小**（保守，可用于风控）。修复后 ``P_repaired``
    近似还原原始价格水平，其 ``pct_change`` 即近似真实价格收益。

    ``max_compression`` 为质量闸门：压缩倍数 ``k̂ = (P_adj + D̂)/P_adj`` 表示该标的的
    复权价曾被压到真实价的 1/k̂。当 k̂ 超过该阈值（默认 3，即复权价不足真实价的 1/3、
    甚至逼近 0）时，源数据 2~3 位小数的舍入误差本身就已主导收益，下界修复无法挽回，
    该标的被**整只剔除**（原因写入诊断的 ``dropped_symbols``）。

    诊断字段：``n_assets_in`` / ``n_assets_out`` / ``dropped_symbols`` /
    ``compression_max``（各标的 k̂ 上界）/ ``offset_max`` / ``anchor_days`` /
    ``violations_before`` / ``violations_after`` / ``tol`` / ``max_compression``。
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("repair_close_panel 需要收盘价 DataFrame 面板")
    frame = prices.astype("float64")
    offset = estimate_dividend_offset(frame, tol)
    with np.errstate(divide="ignore", invalid="ignore"):
        compression = ((frame.abs() + offset) / frame.abs()).max()
    compression = compression.replace([np.inf, -np.inf], np.nan).fillna(1.0)

    dropped: Dict[str, str] = {}
    for col in frame.columns:
        k = float(compression[col])
        if k > float(max_compression):
            dropped[str(col)] = (f"复权压缩倍数 k̂={k:.1f} > max_compression="
                                 f"{float(max_compression):g}（复权价被压到近零，"
                                 "舍入误差主导收益，下界修复不可信）")
    kept = [c for c in frame.columns if str(c) not in dropped]
    if not kept:
        raise ValueError("全部标的的复权压缩都超出 max_compression，请放宽阈值或检查数据")
    repaired = frame[kept] + offset[kept]

    def _violations(px: pd.DataFrame) -> int:
        r = px.pct_change().abs()
        lim = pd.Series({c: daily_price_limit(c, tol) for c in r.columns})
        return int((r > lim).sum().sum())

    diag: Dict[str, object] = {
        "n_assets_in": int(frame.shape[1]),
        "n_assets_out": int(repaired.shape[1]),
        "dropped_symbols": dropped,
        "compression_max": {str(c): float(compression[c]) for c in frame.columns},
        "offset_max": {str(c): float(offset[c].max()) for c in repaired.columns},
        "offset_first": {str(c): float(offset[c].iloc[0]) for c in repaired.columns},
        "anchor_days": {str(c): int(np.count_nonzero(offset[c].to_numpy(dtype="float64") > 0))
                        for c in repaired.columns},
        "violations_before": _violations(frame[kept]),
        "violations_after": _violations(repaired),
        "tol": float(tol),
        "max_compression": float(max_compression),
    }
    repaired.attrs.update(dict(frame.attrs))
    repaired.attrs["repair"] = diag
    return repaired, diag


def sanitize_returns(returns: pd.DataFrame, tol: float = 0.01, max_bad_days: int = 10,
                     min_obs: int = 60, on_bad: str = "clip",
                     ) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """清洗真实收益面板中**物理上不可能的观测**，返回 ``(清洁面板, 诊断字典)``。

    规则（确定性、可复现，绝不静默降级——所有处理都写进诊断字典）：

    1. **标的级**：``|r_it| >`` :func:`daily_price_limit`（板块涨跌停 + ``tol``）的天数
       超过 ``max_bad_days`` 的标的整只剔除（说明其复权口径整体失真，修复未能挽回）；
       有效观测数 ``< min_obs`` 的标的一并剔除；
    2. **单元格级**：残余的越界观测按 ``on_bad`` 处理——
       ``"clip"``（默认）取同号的涨跌停边界值：真实收益必在 ±L 内，取边界是**不低估
       尾部风险**的可辨识替代，且不牵连同日其它标的的真实行情；
       ``"zero"`` 置 0（中性假设，会略微低估尾部）；
       ``"drop"`` 置 NaN 后 ``dropna(how="any")``（最严格，但会连带删掉当日全市场样本，
       而被删的往往正是极端行情日，会系统性低估组合尾部风险）。

    诊断字段：``n_assets_in`` / ``n_assets_out`` / ``dropped_symbols``（含原因）/
    ``bad_cells`` / ``bad_by_symbol`` / ``dropped_days`` / ``n_obs`` /
    ``max_abs_return`` / ``on_bad`` / ``tol`` / ``max_bad_days`` / ``min_obs``。
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("sanitize_returns 需要收益 DataFrame 面板")
    if on_bad not in ("clip", "zero", "drop"):
        raise ValueError("on_bad 只支持 'clip' / 'zero' / 'drop'")
    frame = returns.astype("float64")
    limits = pd.Series({c: daily_price_limit(c, tol) for c in frame.columns})
    bad = frame.abs() > limits
    bad_counts = bad.sum()

    dropped: Dict[str, str] = {}
    for sym in frame.columns:
        if int(bad_counts[sym]) > int(max_bad_days):
            dropped[str(sym)] = (f"超出涨跌停幅度的观测 {int(bad_counts[sym])} 天 > "
                                 f"max_bad_days={int(max_bad_days)}（复权口径整体失真）")
    kept = [c for c in frame.columns if str(c) not in dropped]
    if not kept:
        raise ValueError("全部标的都被判定为口径损坏，请放宽 max_bad_days 或检查数据")
    frame, bad, bad_counts = frame[kept], bad[kept], bad_counts[kept]

    clean = frame.mask(bad)
    n_obs_col = clean.notna().sum()
    for sym in list(clean.columns):
        if int(n_obs_col[sym]) < int(min_obs):
            dropped[str(sym)] = f"有效观测 {int(n_obs_col[sym])} < min_obs={int(min_obs)}"
    kept2 = [c for c in clean.columns if str(c) not in dropped]
    if not kept2:
        raise ValueError("清洗后没有满足 min_obs 的标的，请放宽 min_obs 或检查数据")
    clean, bad = clean[kept2], bad[kept2]

    if on_bad == "clip":
        base = pd.Series({c: daily_price_limit(c, 0.0) for c in clean.columns})
        clean = clean.where(~bad, np.sign(frame.reindex(columns=kept2)) * base)
    elif on_bad == "zero":
        clean = clean.fillna(0.0)

    n_rows_in = int(clean.shape[0])
    clean = clean.dropna(how="any")
    if clean.empty:
        raise ValueError("清洗后收益面板为空（每天都存在不可信观测），请放宽 tol/max_bad_days")
    bad_by_symbol = {str(k): int(v) for k, v in bad_counts.items() if int(v) > 0}
    diag: Dict[str, object] = {
        "n_assets_in": int(returns.shape[1]),
        "n_assets_out": int(clean.shape[1]),
        "dropped_symbols": dropped,
        "bad_cells": int(sum(bad_by_symbol.values())),
        "bad_by_symbol": bad_by_symbol,
        "dropped_days": n_rows_in - int(clean.shape[0]),
        "n_obs": int(clean.shape[0]),
        "max_abs_return": float(clean.abs().to_numpy().max()),
        "on_bad": on_bad,
        "tol": float(tol),
        "max_bad_days": int(max_bad_days),
        "min_obs": int(min_obs),
    }
    clean.attrs.update({"sanitize": diag})
    return clean, diag



def panel_summary(prices: pd.DataFrame, returns: Optional[pd.DataFrame] = None,
                  periods_per_year: float = 252.0) -> Dict[str, object]:
    """收盘价/收益面板的口径摘要（写进报告头部与 JSON 产物）。

    含资产数、期数、起止日期、缺失与前复权无效价统计，以及收益面板的横截面
    年化波动中位数与区间累计收益中位数（真实数据的「体检」信息）。
    """
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("prices 面板为空，无法生成摘要")
    idx = prices.index
    info: Dict[str, object] = {
        "n_assets": int(prices.shape[1]),
        "n_periods": int(prices.shape[0]),
        "start": str(pd.Timestamp(idx[0]).date()),
        "end": str(pd.Timestamp(idx[-1]).date()),
        "periods_per_year": float(periods_per_year),
        "n_missing_prices": int(prices.isna().sum().sum()),
        "masked_prices_total": int(sum(prices.attrs.get("masked_prices", {}).values())),
        "dropped_prices": int(prices.attrs.get("dropped_prices", 0)),
        "source_dir": str(prices.attrs.get("source_dir", "")),
        "symbols": list(map(str, prices.columns)),
    }
    r = returns if returns is not None else simple_returns(prices)
    ann = float(np.sqrt(periods_per_year))
    vol = r.std(ddof=1) * ann
    cum = (1.0 + r).prod() - 1.0
    info.update({
        "n_return_periods": int(r.shape[0]),
        "annualized_vol_median": float(vol.median()),
        "annualized_vol_min": float(vol.min()),
        "annualized_vol_max": float(vol.max()),
        "cumulative_return_median": float(cum.median()),
        "worst_single_day": float(r.min().min()),
        "best_single_day": float(r.max().max()),
    })
    return info


def symbols_of(panel: Union[pd.DataFrame, List[str]]) -> List[str]:
    """取出面板的标的代码列表（也接受直接的列表输入），便于报告展示。"""
    if isinstance(panel, pd.DataFrame):
        return [str(c) for c in panel.columns]
    return [str(c) for c in panel]


__all__ = [
    "DATE_COL", "PRICE_COL", "LIMIT_GEM_STAR", "LIMIT_MAIN", "default_data_dir",
    "has_local_data", "load_close_panel", "simple_returns", "log_returns",
    "daily_price_limit", "estimate_dividend_offset", "repair_close_panel",
    "sanitize_returns", "panel_summary", "symbols_of",
]
