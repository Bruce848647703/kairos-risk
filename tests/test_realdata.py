"""realdata 模块测试：真实 CSV 行情接入、等差前复权修复与收益清洗（全部离线、tmp_path 造数）。

不联网、不读仓库外的任何文件：所有 CSV 都在 ``tmp_path`` 下现造，日期与价格均为
确定性构造（固定 seed），因此断言可以精确到数值。
"""
from __future__ import annotations

import json
import math
import os
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest

import kairos_risk as kr
from kairos_risk import realdata as rd


# --------------------------------------------------------------------------- #
# 造数工具（离线、确定性）
# --------------------------------------------------------------------------- #
def _dates(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    """生成 n 个工作日期索引。"""
    return pd.bdate_range(start=start, periods=n)


def _write_csv(dirpath, symbol: str, closes, dates: Optional[pd.DatetimeIndex] = None,
               volume: float = 1.0e5, shuffle: bool = False,
               duplicate_last: bool = False) -> str:
    """把一条收盘价序列写成 ``date,open,high,low,close,volume`` 的 CSV（与真实数据同构）。"""
    arr = np.asarray(closes, dtype="float64")
    idx = _dates(arr.size) if dates is None else pd.DatetimeIndex(dates)
    df = pd.DataFrame({"date": idx.strftime("%Y-%m-%d"), "open": arr, "high": arr,
                       "low": arr, "close": arr, "volume": volume})
    if duplicate_last:                      # 同日再写一条（close 翻倍），验证「保留最后一条」
        extra = df.iloc[[-1]].copy()
        extra["close"] = arr[-1] * 2.0
        df = pd.concat([df, extra], ignore_index=True)
    if shuffle:
        df = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    path = os.path.join(str(dirpath), f"{symbol}.csv")
    df.to_csv(path, index=False)
    return path


def _walk(n: int, seed: int = 3, start: float = 20.0, vol: float = 0.02) -> np.ndarray:
    """生成一条「日收益不超过 ±8%」的确定性价格路径。"""
    rng = np.random.default_rng(seed)
    rets = np.clip(rng.normal(0.0005, vol, n), -0.08, 0.08)
    rets[0] = 0.0
    return start * np.cumprod(1.0 + rets)


def _make_qfq(n: int = 120, seed: int = 7, start: float = 40.0, vol: float = 0.02,
              divs: Sequence[Tuple[int, float]] = ((40, 2.0), (80, 2.0)),
              limit_days: Sequence[int] = ()) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造「等差（减法）前复权」样本，返回 ``(raw, d_true, adj)``。

    - ``raw``：原始价格路径，除息日按现金分红除权（价格下跌），涨停锚点日为精确 +10%；
    - ``d_true``：每日「未来累计分红」D_t（单调不增、末日为 0），即减法复权的偏移量；
    - ``adj``：数据源给出的前复权价 ``raw − D``（在除息日**连续**，与真实数据一致）。

    由 ``P_raw = P_adj + D`` 可知：``adj`` 的 ``pct_change`` 是被放大 k=raw/adj 倍的伪收益，
    而 ``estimate_dividend_offset`` 应当从涨跌停硬约束里把 D 反解出来。
    """
    rng = np.random.default_rng(seed)
    m = np.clip(rng.normal(0.0004, vol, n), -0.08, 0.08)
    m[0] = 0.0
    for t in limit_days:
        m[t] = 0.10                                    # 精确涨停：可辨识的锚点
    div = np.zeros(n)
    for t, amt in divs:
        div[t] = amt
        if t not in limit_days:
            m[t] = 0.0                                 # 除息日不叠加市场波动，便于精确断言
    raw = np.empty(n)
    raw[0] = float(start)
    for t in range(1, n):
        raw[t] = raw[t - 1] * (1.0 + m[t]) - div[t]
    d_true = np.cumsum(div[::-1])[::-1] - div          # d[t] = Σ_{s>t} div_s
    adj = raw - d_true
    if adj.min() <= 0:
        raise AssertionError("测试数据构造失败：前复权价出现非正值")
    return raw, d_true, adj


# --------------------------------------------------------------------------- #
# load_close_panel
# --------------------------------------------------------------------------- #
class TestLoadClosePanel:
    def test_basic_shape_and_alignment(self, tmp_path):
        """三只标的、同样日期：面板形状/列名/索引都应正确对齐。"""
        for i, sym in enumerate(["sh600000", "sh600001", "sz000002"]):
            _write_csv(tmp_path, sym, _walk(30, seed=i))
        panel = rd.load_close_panel(str(tmp_path))
        assert isinstance(panel, pd.DataFrame)
        assert panel.shape == (30, 3)
        assert list(panel.columns) == ["sh600000", "sh600001", "sz000002"]   # 按代码排序
        assert isinstance(panel.index, pd.DatetimeIndex)
        assert panel.index.is_monotonic_increasing
        assert not panel.isna().any().any()
        assert panel.attrs["n_files"] == 3
        assert panel.attrs["symbols"] == list(panel.columns)
        assert panel.attrs["drop_incomplete"] is True
        assert panel.attrs["window"] == (str(panel.index[0].date()), str(panel.index[-1].date()))

    def test_values_match_source_csv(self, tmp_path):
        """面板数值应与 CSV 中的 close 列逐位一致。"""
        closes = _walk(12, seed=1)
        _write_csv(tmp_path, "sh600000", closes)
        _write_csv(tmp_path, "sh600001", closes * 2.0)
        panel = rd.load_close_panel(str(tmp_path))
        assert np.allclose(panel["sh600000"].to_numpy(), closes)
        assert np.allclose(panel["sh600001"].to_numpy(), closes * 2.0)

    def test_unsorted_rows_are_sorted(self, tmp_path):
        """CSV 行序乱掉也要按日期升序对齐，数值不错位。"""
        closes = _walk(10, seed=2)
        _write_csv(tmp_path, "sh600000", closes, shuffle=True)
        panel = rd.load_close_panel(str(tmp_path))
        assert panel.shape[0] == 10
        assert panel.index.is_monotonic_increasing
        assert np.allclose(panel["sh600000"].to_numpy(), closes)

    def test_duplicate_date_keeps_last(self, tmp_path):
        """同日多条记录保留文件中最后一条（避免静默重复计数）。"""
        closes = _walk(10, seed=2)
        _write_csv(tmp_path, "sh600000", closes, duplicate_last=True)
        panel = rd.load_close_panel(str(tmp_path))
        assert panel.shape[0] == 10
        assert float(panel["sh600000"].iloc[-1]) == pytest.approx(closes[-1] * 2.0)

    def test_non_positive_prices_are_masked(self, tmp_path):
        """前复权负价/零价必须被记为缺失，绝不能进入收益计算。"""
        closes = _walk(20, seed=4)
        bad = closes.copy()
        bad[:5] = [-1.0, 0.0, -3.5, 2.0, 1.0]          # 前 3 天非正价
        _write_csv(tmp_path, "sh600000", bad)
        _write_csv(tmp_path, "sh600001", closes)
        panel = rd.load_close_panel(str(tmp_path), drop_incomplete=False)
        arr = panel["sh600000"].to_numpy()
        assert np.isnan(arr[:3]).all()                  # 非正 -> NaN
        assert np.isfinite(arr[3:]).all()
        assert panel.attrs["masked_prices"]["sh600000"] == 3

    def test_drop_incomplete_trims_to_common_positive_window(self, tmp_path):
        """drop_incomplete=True：起点 = max(每标的最后一个非正价之后, 首个有记录日)。"""
        n = 30
        dates = _dates(n)
        clean = _walk(n, seed=5)
        broken = clean.copy()
        broken[8] = -0.5                                # 第 9 天出现负价（最后一个坏点）
        late = np.full(n, np.nan)
        late[12:] = clean[12:]                          # 第 13 天才上市
        _write_csv(tmp_path, "sh600000", broken, dates=dates)
        _write_csv(tmp_path, "sh600001", clean, dates=dates)
        _write_csv(tmp_path, "sh600002", late, dates=dates)
        panel = rd.load_close_panel(str(tmp_path), drop_incomplete=True)
        assert panel.index[0] == dates[12]              # max(8+1, 0, 12) = 12
        assert panel.shape == (n - 12, 3)
        assert not panel.isna().any().any()
        assert panel.attrs["dropped_prices"] > 0

    def test_drop_incomplete_false_keeps_union_and_leading_nan(self, tmp_path):
        """drop_incomplete=False：保留并集日期，头部缺失不裁剪。"""
        n = 20
        dates = _dates(n)
        clean = _walk(n, seed=6)
        late = np.full(n, np.nan)
        late[5:] = clean[5:]
        _write_csv(tmp_path, "sh600000", clean, dates=dates)
        _write_csv(tmp_path, "sh600001", late, dates=dates)
        panel = rd.load_close_panel(str(tmp_path), drop_incomplete=False)
        assert panel.shape == (n, 2)
        assert panel.index[0] == dates[0]
        assert panel["sh600001"].isna().sum() == 5       # 上市前保留 NaN
        assert np.allclose(panel["sh600001"].dropna().to_numpy(), clean[5:])

    def test_suspension_gap_is_forward_filled(self, tmp_path):
        """窗口内的停牌缺口用前收盘价前填（当日收益 0），不引入未来函数。"""
        n = 15
        dates = _dates(n)
        a = _walk(n, seed=8)
        b = _walk(n, seed=9)
        _write_csv(tmp_path, "sh600000", a, dates=dates)
        _write_csv(tmp_path, "sh600001", np.delete(b, 7), dates=dates.delete(7))   # 缺一天
        panel = rd.load_close_panel(str(tmp_path))
        assert panel.shape == (n, 2)
        assert float(panel["sh600001"].iloc[7]) == pytest.approx(b[6])    # 前填而非后填
        rets = rd.simple_returns(panel)
        assert float(rets["sh600001"].iloc[6]) == pytest.approx(0.0, abs=1e-15)

    def test_start_end_window(self, tmp_path):
        """start/end 在裁剪之后生效，且含端点。"""
        n = 40
        dates = _dates(n, start="2021-01-01")
        _write_csv(tmp_path, "sh600000", _walk(n, seed=10), dates=dates)
        panel = rd.load_close_panel(str(tmp_path), start="2021-02-01", end="2021-03-31")
        assert panel.index[0] >= pd.Timestamp("2021-02-01")
        assert panel.index[-1] <= pd.Timestamp("2021-03-31")
        assert len(panel) < n

    def test_missing_dir_raises(self, tmp_path):
        """目录不存在时给出可操作的中文报错，而不是静默返回空面板。"""
        with pytest.raises(FileNotFoundError, match="数据目录不存在"):
            rd.load_close_panel(str(tmp_path / "not_here"))

    def test_empty_dir_raises(self, tmp_path):
        """目录里没有 CSV 时报错。"""
        (tmp_path / "readme.md").write_text("no csv here", encoding="utf-8")
        with pytest.raises(FileNotFoundError, match="没有 CSV"):
            rd.load_close_panel(str(tmp_path))

    def test_missing_column_raises(self, tmp_path):
        """缺少 close/date 列时报错并指出实际列名。"""
        pd.DataFrame({"date": ["2020-01-01"], "px": [10.0]}).to_csv(
            tmp_path / "sh600000.csv", index=False)
        with pytest.raises(ValueError, match="缺少必需列"):
            rd.load_close_panel(str(tmp_path))

    def test_all_bad_prices_raises(self, tmp_path):
        """全体标的都没有有效价格区间时应报错。"""
        _write_csv(tmp_path, "sh600000", [-1.0] * 5)
        with pytest.raises(ValueError):
            rd.load_close_panel(str(tmp_path), drop_incomplete=True)

    def test_custom_column_names(self, tmp_path):
        """支持自定义日期列/价格列名。"""
        closes = _walk(8, seed=11)
        pd.DataFrame({"d": _dates(8).strftime("%Y-%m-%d"), "px": closes}).to_csv(
            tmp_path / "sh600000.csv", index=False)
        panel = rd.load_close_panel(str(tmp_path), date_col="d", price_col="px")
        assert np.allclose(panel["sh600000"].to_numpy(), closes)

    def test_default_data_dir_and_has_local_data(self, tmp_path):
        """默认目录按「兄弟仓库」相对定位；has_local_data 只做本地探测（不联网）。"""
        path = rd.default_data_dir("ashare")
        assert path.endswith(os.path.join("kairos-data", "data", "ashare"))
        assert rd.has_local_data(str(tmp_path)) is False           # 空目录
        _write_csv(tmp_path, "sh600000", _walk(5, seed=12))
        assert rd.has_local_data(str(tmp_path)) is True
        assert rd.has_local_data(str(tmp_path / "nope")) is False
        assert rd.has_local_data() in (True, False)                # 无参数也不报错


# --------------------------------------------------------------------------- #
# 收益转换与摘要
# --------------------------------------------------------------------------- #
class TestReturns:
    def test_simple_returns_values(self, tmp_path):
        """简单收益 = P_t/P_{t−1} − 1，首行默认剔除。"""
        closes = np.array([10.0, 11.0, 10.5, 12.0])
        _write_csv(tmp_path, "sh600000", closes)
        panel = rd.load_close_panel(str(tmp_path))
        r = rd.simple_returns(panel)
        assert r.shape == (3, 1)
        assert np.allclose(r["sh600000"].to_numpy(),
                           [0.1, 10.5 / 11.0 - 1.0, 12.0 / 10.5 - 1.0])
        assert rd.simple_returns(panel, drop_first=False).shape == (4, 1)

    def test_log_returns_are_additive(self, tmp_path):
        """对数收益跨期可加：全区间之和 = ln(P_T/P_0)。"""
        closes = _walk(25, seed=13)
        _write_csv(tmp_path, "sh600000", closes)
        panel = rd.load_close_panel(str(tmp_path))
        lr = rd.log_returns(panel)
        assert float(lr["sh600000"].sum()) == pytest.approx(math.log(closes[-1] / closes[0]))
        assert rd.log_returns(panel, drop_first=False).iloc[0].isna().all()

    def test_returns_reject_non_frame(self):
        """输入必须是 DataFrame 面板（Series/ndarray 直接报错，避免静默误用）。"""
        with pytest.raises(TypeError):
            rd.simple_returns(pd.Series([1.0, 2.0]))
        with pytest.raises(TypeError):
            rd.log_returns(np.array([1.0, 2.0]))

    def test_panel_summary_fields(self, tmp_path):
        """摘要包含口径、区间、波动与极值等字段，且数值可核对。"""
        closes = _walk(40, seed=14)
        _write_csv(tmp_path, "sh600000", closes)
        _write_csv(tmp_path, "sh600001", closes * 1.5)
        panel = rd.load_close_panel(str(tmp_path))
        rets = rd.simple_returns(panel)
        info = rd.panel_summary(panel, rets, periods_per_year=252)
        for key in ("n_assets", "n_periods", "start", "end", "n_missing_prices",
                    "masked_prices_total", "dropped_prices", "symbols", "n_return_periods",
                    "annualized_vol_median", "cumulative_return_median",
                    "worst_single_day", "best_single_day"):
            assert key in info, key
        assert info["n_assets"] == 2 and info["n_periods"] == 40
        assert info["start"] == str(panel.index[0].date())
        assert info["n_return_periods"] == 39
        assert info["annualized_vol_median"] == pytest.approx(
            float(rets.std(ddof=1).median() * math.sqrt(252.0)))
        assert info["worst_single_day"] <= 0.0 < info["best_single_day"]
        with pytest.raises(ValueError):
            rd.panel_summary(pd.DataFrame())

    def test_symbols_of(self, tmp_path):
        """symbols_of 同时接受面板与列表。"""
        _write_csv(tmp_path, "sh600000", _walk(5, seed=15))
        panel = rd.load_close_panel(str(tmp_path))
        assert rd.symbols_of(panel) == ["sh600000"]
        assert rd.symbols_of(["a", "b"]) == ["a", "b"]


# --------------------------------------------------------------------------- #
# 涨跌停约束与等差前复权修复
# --------------------------------------------------------------------------- #
class TestPriceLimitAndRepair:
    def test_daily_price_limit_by_board(self):
        """主板 ±10%、创业板/科创板 ±20%，容差可加、不可为负。"""
        assert rd.daily_price_limit("sh600000") == pytest.approx(0.11)
        assert rd.daily_price_limit("sz000001") == pytest.approx(0.11)
        assert rd.daily_price_limit("sz300750") == pytest.approx(0.21)
        assert rd.daily_price_limit("sh688001") == pytest.approx(0.21)
        assert rd.daily_price_limit("sh600000", tol=0.0) == pytest.approx(0.10)
        assert rd.daily_price_limit("sh600000", tol=0.05) == pytest.approx(0.15)
        with pytest.raises(ValueError):
            rd.daily_price_limit("sh600000", tol=-0.01)

    def test_offset_recovers_true_additive_adjustment(self):
        """白盒：段末涨停锚点 + tol=0 -> D̂ 精确还原真实复权偏移，修复后收益逐位相等。

        构造 ``adj = raw − D``（D 分段常数、除息日连续），并在每段最后一天放一个精确
        +10% 涨停：此时约束反解出的 required 恰等于该段 D，反向累计最大值把它传播到
        整段，因此 D̂ 应逐位等于 D_真，``adj + D̂`` 应逐位还原 raw。
        """
        raw, d_true, adj = _make_qfq(n=120, seed=7, start=30.0,
                                     divs=((40, 2.0), (80, 2.0)), limit_days=(39, 79))
        prices = pd.DataFrame({"sh600000": adj}, index=_dates(120))
        off = rd.estimate_dividend_offset(prices, tol=0.0)
        assert np.allclose(off["sh600000"].to_numpy(), d_true, atol=1e-9)
        repaired = prices + off
        assert np.allclose(repaired["sh600000"].to_numpy(), raw, atol=1e-8)
        r_true = pd.Series(raw, index=_dates(120)).pct_change().dropna()
        r_rep = repaired["sh600000"].pct_change().dropna()
        assert np.allclose(r_rep.to_numpy(), r_true.to_numpy(), atol=1e-12)

    def test_offset_is_lower_bound_and_monotone(self):
        """带容差时 D̂ ≤ D_真（保守下界），且沿时间单调不增、末端为 0。"""
        raw, d_true, adj = _make_qfq(n=100, seed=8, start=25.0,
                                     divs=((50, 3.0),), limit_days=(49,))
        prices = pd.DataFrame({"sh600000": adj}, index=_dates(100))
        off = rd.estimate_dividend_offset(prices, tol=0.01)["sh600000"].to_numpy()
        assert np.all(off <= d_true + 1e-12)               # 下界：绝不超调（不会把风险修小）
        assert np.all(np.diff(off) <= 1e-12)               # 单调不增
        assert off[0] > 0.0
        assert off[-1] == pytest.approx(0.0, abs=1e-12)
        exact = rd.estimate_dividend_offset(prices, tol=0.0)["sh600000"].to_numpy()
        assert np.all(off <= exact + 1e-12)                # 容差越大越保守

    def test_repair_shrinks_inflated_returns(self, tmp_path):
        """修复后被放大的伪收益回落：波动下降、涨跌停越界数下降、偏移不超调。"""
        raw, d_true, adj = _make_qfq(n=200, seed=9, start=40.0,
                                     divs=((40, 2.0), (80, 2.0), (120, 2.0), (160, 2.0)),
                                     limit_days=(39, 79, 119, 159))
        _write_csv(tmp_path, "sh600000", adj)
        panel = rd.load_close_panel(str(tmp_path))
        before = rd.simple_returns(panel)["sh600000"]
        repaired, diag = rd.repair_close_panel(panel, tol=0.01)
        after = rd.simple_returns(repaired)["sh600000"]
        assert float(after.std()) < float(before.std())                    # 伪波动被压回
        assert diag["violations_after"] <= diag["violations_before"]
        assert diag["n_assets_out"] == 1 and diag["dropped_symbols"] == {}
        assert diag["offset_max"]["sh600000"] <= float(d_true.max()) + 1e-9
        assert repaired.attrs["source_dir"] == panel.attrs["source_dir"]   # 口径信息保留
        assert float(before.abs().max()) > rd.daily_price_limit("sh600000")

    def test_repair_drops_over_compressed_symbol(self, tmp_path):
        """复权价被压到近零（压缩倍数超闸门）的标的整只剔除，并留下原因。"""
        raw = _walk(200, seed=24, start=20.0)
        crushed = raw - raw.min() + 0.05             # 复权价被压到 0.05 附近 -> k̂ 巨大
        _write_csv(tmp_path, "sh600000", raw)
        _write_csv(tmp_path, "sh600001", crushed)
        panel = rd.load_close_panel(str(tmp_path))
        repaired, diag = rd.repair_close_panel(panel, max_compression=3.0)
        assert list(repaired.columns) == ["sh600000"]
        assert diag["n_assets_in"] == 2 and diag["n_assets_out"] == 1
        assert "sh600001" in diag["dropped_symbols"]
        assert "压缩倍数" in diag["dropped_symbols"]["sh600001"]
        assert diag["compression_max"]["sh600001"] > 3.0
        assert diag["compression_max"]["sh600000"] == pytest.approx(1.0, abs=1e-9)

    def test_repair_raises_when_all_compressed(self, tmp_path):
        """全部标的都压缩过度时报错，而不是返回空面板。"""
        raw = _walk(80, seed=25, start=20.0)
        _write_csv(tmp_path, "sh600000", raw - raw.min() + 0.01)
        panel = rd.load_close_panel(str(tmp_path))
        with pytest.raises(ValueError, match="max_compression"):
            rd.repair_close_panel(panel, max_compression=1.0)

    def test_repair_rejects_non_frame(self):
        """输入必须是面板。"""
        with pytest.raises(TypeError):
            rd.repair_close_panel(pd.Series([1.0, 2.0]))
        with pytest.raises(TypeError):
            rd.estimate_dividend_offset(np.array([[1.0, 2.0]]))


# --------------------------------------------------------------------------- #
# sanitize_returns
# --------------------------------------------------------------------------- #
class TestSanitizeReturns:
    @staticmethod
    def _panel() -> pd.DataFrame:
        """3 标的 × 100 期：sh600000 有 2 个越界、sh600001 有 12 个越界（触发标的级剔除）。"""
        rng = np.random.default_rng(31)
        n = 100
        base = pd.DataFrame(rng.normal(0.0, 0.012, (n, 3)), index=_dates(n),
                            columns=["sh600000", "sh600001", "sz300750"])
        base.iloc[10, 0] = 0.25                        # 主板越界（>11%）
        base.iloc[50, 0] = -0.18                       # 主板越界
        for i in range(12):
            base.iloc[20 + i, 1] = 0.30                # 12 天越界 > max_bad_days=10
        base.iloc[5, 2] = 0.19                         # 创业板 19%：±20% 内，属合法观测
        return base

    def test_clip_keeps_shape_and_bounds(self):
        """默认 clip：形状不变，越界值被压到同号的涨跌停边界，越界标的被剔除。"""
        frame = self._panel()
        clean, diag = rd.sanitize_returns(frame)
        assert clean.shape == (frame.shape[0], 2)      # sh600001 被整只剔除
        assert diag["n_assets_out"] == 2
        assert "sh600001" in diag["dropped_symbols"]
        assert "max_bad_days" in diag["dropped_symbols"]["sh600001"]
        assert float(clean["sh600000"].iloc[10]) == pytest.approx(0.10)
        assert float(clean["sh600000"].iloc[50]) == pytest.approx(-0.10)
        assert float(clean["sz300750"].iloc[5]) == pytest.approx(0.19)   # 合法值原样保留
        assert float(clean.abs().to_numpy().max()) <= 0.20 + 1e-12
        assert diag["dropped_days"] == 0 and diag["on_bad"] == "clip"
        assert diag["bad_cells"] == 2
        assert clean.attrs["sanitize"] is diag

    def test_zero_and_drop_policies(self):
        """zero 置 0；drop 连带删除当日全截面样本（最严格）。"""
        frame = self._panel()
        clean_z, diag_z = rd.sanitize_returns(frame, on_bad="zero")
        assert float(clean_z["sh600000"].iloc[10]) == 0.0
        assert diag_z["dropped_days"] == 0
        clean_d, diag_d = rd.sanitize_returns(frame, on_bad="drop")
        assert diag_d["dropped_days"] == 2                # sh600000 的两个越界日
        assert clean_d.shape[0] == frame.shape[0] - 2
        assert not clean_d.isna().any().any()

    def test_max_bad_days_and_min_obs(self):
        """标的级闸门可调：放宽 max_bad_days 则保留；min_obs 过高则全部剔除并报错。"""
        frame = self._panel()
        clean, diag = rd.sanitize_returns(frame, max_bad_days=20)
        assert diag["n_assets_out"] == 3 and diag["dropped_symbols"] == {}
        assert clean.shape[1] == 3 and diag["bad_cells"] == 14
        with pytest.raises(ValueError, match="min_obs"):
            rd.sanitize_returns(frame, min_obs=1000)

    def test_invalid_inputs(self):
        """非法 on_bad / 非 DataFrame 输入要报错。"""
        with pytest.raises(ValueError, match="on_bad"):
            rd.sanitize_returns(self._panel(), on_bad="interp")
        with pytest.raises(TypeError):
            rd.sanitize_returns(np.zeros((3, 3)))

    def test_all_symbols_broken_raises(self):
        """全体标的都越界时报错，而不是返回空面板。"""
        frame = pd.DataFrame(0.5, index=_dates(10), columns=["sh600000", "sh600001"])
        with pytest.raises(ValueError):
            rd.sanitize_returns(frame, max_bad_days=1)

    def test_clean_panel_passes_through(self):
        """本就越界为零的面板应原样通过（不误伤）。"""
        rng = np.random.default_rng(33)
        frame = pd.DataFrame(rng.normal(0.0, 0.01, (80, 2)), index=_dates(80),
                             columns=["sh600000", "sh600001"])
        clean, diag = rd.sanitize_returns(frame)
        assert diag["bad_cells"] == 0 and diag["dropped_days"] == 0
        assert np.allclose(clean.to_numpy(), frame.to_numpy())


# --------------------------------------------------------------------------- #
# 端到端：真实数据接入 -> kairos_risk 风险链路
# --------------------------------------------------------------------------- #
class TestEndToEnd:
    def test_pipeline_feeds_kairos_risk(self, tmp_path):
        """CSV -> 面板 -> 复权修复 -> 收益 -> 清洗 -> 成分风险，欧拉可加性必须精确成立。"""
        n = 250
        for i, sym in enumerate(["sh600000", "sh600001", "sh600002", "sz300750"]):
            _, _, adj = _make_qfq(n=n, seed=40 + i, start=40.0 + 5.0 * i,
                                  divs=((60, 2.0), (120, 2.0), (180, 2.0)),
                                  limit_days=(59, 119, 179))
            _write_csv(tmp_path, sym, adj)
        panel = rd.load_close_panel(str(tmp_path))
        repaired, rep = rd.repair_close_panel(panel)
        assert rep["violations_after"] <= rep["violations_before"]
        rets, diag = rd.sanitize_returns(rd.simple_returns(repaired))
        assert rets.shape == (n - 1, 4) and not rets.isna().any().any()
        assert diag["bad_cells"] == rep["violations_after"]          # 两级诊断口径一致
        for col in rets.columns:                                     # 全链路后无越界收益
            assert float(rets[col].abs().max()) <= rd.daily_price_limit(col) + 1e-12

        w = pd.Series(0.25, index=rets.columns)
        dec = kr.component_risk(w, kr.covariance_matrix(rets))
        assert float(dec.component.sum()) == pytest.approx(dec.volatility, abs=1e-15)
        assert dec.diversification_ratio >= 1.0
        ce = kr.component_es(w, rets, 0.95)
        assert float(ce.component.sum()) == pytest.approx(ce.value, abs=1e-15)
        port = kr.portfolio_returns(rets, w)
        assert kr.max_drawdown(port) >= 0.0
        assert kr.historical_var(port, 0.95) > 0.0
        assert kr.expected_shortfall(port, 0.99) >= kr.historical_var(port, 0.99)
        assert kr.hill_estimator(port, k=20).n_obs == len(port)
        assert kr.psr_from_returns(port) > 0.0

    def test_no_network_and_no_optional_deps(self):
        """铁律：realdata 只依赖 numpy/pandas/标准库，源码里不得出现联网或 scipy/sklearn。"""
        import inspect

        src = inspect.getsource(rd)
        for forbidden in ("urllib", "requests", "http://", "https://", "socket",
                          "scipy", "sklearn", "statsmodels"):
            assert forbidden not in src, f"realdata 不应依赖 {forbidden}"

    def test_diagnostics_are_json_serializable(self, tmp_path):
        """诊断字典必须可直接 json.dump（报告产物 risk_metrics.json 的前提）。"""
        closes = _walk(200, seed=50)
        _write_csv(tmp_path, "sh600000", closes)
        _write_csv(tmp_path, "sh600001", closes * 1.2)
        panel = rd.load_close_panel(str(tmp_path))
        repaired, rep = rd.repair_close_panel(panel)
        rets, diag = rd.sanitize_returns(rd.simple_returns(repaired))
        info = rd.panel_summary(panel, rets)
        text = json.dumps({"repair": rep, "sanitize": diag, "summary": info},
                          ensure_ascii=False, allow_nan=False, default=float)
        loaded = json.loads(text)
        assert loaded["summary"]["n_assets"] == 2
        assert loaded["repair"]["n_assets_out"] == 2
        assert "NaN" not in text and "Infinity" not in text
