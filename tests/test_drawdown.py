"""drawdown 模块测试：构造序列上最大回撤/持续期/恢复期精确正确，区间切分与 Calmar。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import kairos_risk as kr


# --------------------------------------------------------------------------- #
# 基础曲线
# --------------------------------------------------------------------------- #
class TestCurves:
    def test_wealth_curve_exact(self):
        r = pd.Series([0.1, -0.05, 0.02])
        w = kr.wealth_curve(r)
        expected = np.array([1.1, 1.1 * 0.95, 1.1 * 0.95 * 1.02])
        assert np.allclose(w.to_numpy(), expected)
        assert w.index.equals(r.index)
        w2 = kr.wealth_curve(r, start=2.0)
        assert np.allclose(w2.to_numpy(), 2.0 * expected)

    def test_running_peak_monotone_and_ge_wealth(self):
        path = kr.make_drawdown_path()
        peak = kr.running_peak(path)
        w = kr.wealth_curve(path)
        assert (np.diff(peak.to_numpy()) >= -1e-15).all()
        assert (peak.to_numpy() >= w.to_numpy() - 1e-15).all()

    def test_drawdown_series_nonpositive(self):
        path = kr.make_drawdown_path()
        dd = kr.drawdown_series(path)
        assert (dd.to_numpy() <= 0.0).all()
        manual = kr.wealth_curve(path) / peak_of(path) - 1.0
        assert np.allclose(dd.to_numpy(), manual.to_numpy())

    def test_underwater_table(self):
        r = pd.Series([0.10, -0.05, -0.04, 0.02, 0.03, 0.05, -0.10, 0.02])
        uw = kr.underwater(r)
        assert list(uw.columns) == ["wealth", "peak", "drawdown",
                                    "in_drawdown", "periods_underwater"]
        # 水下计数：新高归零，水下逐期 +1
        assert list(uw["periods_underwater"]) == [0, 1, 2, 3, 4, 0, 1, 2]
        assert list(uw["in_drawdown"]) == [False, True, True, True, True,
                                           False, True, True]
        assert int(uw["periods_underwater"].max()) == kr.longest_drawdown(r)
        assert abs(kr.time_underwater_ratio(r) - 6.0 / 8.0) < 1e-15


def peak_of(r: pd.Series) -> pd.Series:
    """测试内部小工具：直接由净值算历史峰值。"""
    w = np.cumprod(1.0 + r.to_numpy())
    return pd.Series(np.maximum.accumulate(np.maximum(w, 1.0)), index=r.index)


# --------------------------------------------------------------------------- #
# 最大回撤 / 当前回撤（精确值）
# --------------------------------------------------------------------------- #
class TestMaxDrawdown:
    def test_exact_on_constructed_path(self):
        path = kr.make_drawdown_path(up_periods=5, down_periods=4, recover_periods=9,
                                     up=0.02, down=-0.03)
        expected = 1.0 - (1.0 - 0.03) ** 4   # 峰后连跌 4 期
        assert abs(kr.max_drawdown(path) - expected) < 1e-12

    def test_hand_series_exact(self):
        r = pd.Series([0.10, -0.05, -0.04, 0.02, 0.03, 0.05, -0.10, 0.02])
        # 第一段谷 = 1.1·0.95·0.96 = 1.0032（深度 0.088）；第二段谷 = 峰·0.9（深度 0.1）
        assert abs(kr.max_drawdown(r) - 0.1) < 1e-12
        peak = float(np.prod(1.0 + r.to_numpy()[:6]))      # 第 6 期创新高
        w_end = float(np.prod(1.0 + r.to_numpy()))
        assert abs(kr.current_drawdown(r) - (1.0 - w_end / peak)) < 1e-12

    def test_no_drawdown_returns_clean_zero(self):
        up = pd.Series([0.01, 0.02, 0.015])
        mdd = kr.max_drawdown(up)
        assert mdd == 0.0 and str(mdd) == "0.0"   # 不允许出现 -0.0
        assert kr.current_drawdown(up) == 0.0
        assert kr.longest_drawdown(up) == 0
        assert kr.time_underwater_ratio(up) == 0.0
        assert kr.calmar_ratio(up, periods_per_year=1) == float("inf")

    def test_current_drawdown_unrecovered(self):
        path = kr.make_drawdown_path(up_periods=3, down_periods=3, recover_periods=4,
                                     up=0.03, down=-0.04, recover=0.01)
        cur = kr.current_drawdown(path)
        mdd = kr.max_drawdown(path)
        assert 0.0 < cur < mdd + 1e-15
        expected_cur = 1.0 - (0.96 ** 3) * (1.01 ** 4)
        assert abs(cur - expected_cur) < 1e-12


# --------------------------------------------------------------------------- #
# 回撤区间：下跌期 / 恢复期 / 未收复
# --------------------------------------------------------------------------- #
class TestEpisodes:
    def test_recovered_episode_exact(self):
        path = kr.make_drawdown_path(up_periods=5, down_periods=4, recover_periods=9,
                                     up=0.02, down=-0.03)
        eps = kr.drawdown_episodes(path)
        assert len(eps) == 1
        e = eps[0]
        assert e.recovered and e.end is not None
        assert e.start_pos == 4 and e.trough_pos == 8 and e.end_pos == 15
        assert e.down_periods == 4          # 峰 -> 谷
        assert e.recovery_periods == 7      # (1.02)^7 恰好收复 (0.97)^4 的跌幅
        assert e.total_periods == 11        # 峰 -> 收复
        assert abs(e.depth - (1.0 - 0.97 ** 4)) < 1e-12
        assert abs(e.peak_value - 1.02 ** 5) < 1e-12
        assert abs(e.trough_value - 1.02 ** 5 * 0.97 ** 4) < 1e-12

    def test_unrecovered_episode(self):
        path = kr.make_drawdown_path(up_periods=3, down_periods=3, recover_periods=4,
                                     up=0.03, down=-0.04, recover=0.01)
        eps = kr.drawdown_episodes(path)
        assert len(eps) == 1
        e = eps[0]
        assert not e.recovered and e.end is None and e.end_pos == -1
        assert math.isnan(e.recovery_periods)
        assert abs(e.depth - (1.0 - 0.96 ** 3)) < 1e-12
        rep = kr.drawdown_report(path)
        assert rep.n_episodes == 1

    def test_two_episodes_order_and_depths(self):
        r = pd.Series([0.10, -0.05, -0.04, 0.02, 0.03, 0.05, -0.10, 0.02])
        eps = kr.drawdown_episodes(r)
        assert len(eps) == 2
        assert eps[0].recovered and not eps[1].recovered
        assert abs(eps[0].depth - 0.088) < 1e-12
        assert abs(eps[1].depth - 0.1) < 1e-12
        assert eps[0].start_pos == 0 and eps[1].start_pos == 5

    def test_top_drawdowns_sorted_desc(self):
        r = pd.Series([0.10, -0.05, -0.04, 0.02, 0.03, 0.05, -0.10, 0.02])
        top = kr.top_drawdowns(r, n=5)
        depths = [e.depth for e in top]
        assert depths == sorted(depths, reverse=True)
        assert len(top) == 2
        assert abs(top[0].depth - 0.1) < 1e-12
        top1 = kr.top_drawdowns(r, n=1)
        assert len(top1) == 1

    def test_min_depth_filter(self):
        r = pd.Series([0.10, -0.05, -0.04, 0.02, 0.03, 0.05, -0.10, 0.02])
        eps = kr.drawdown_episodes(r, min_depth=0.09)
        assert len(eps) == 1
        assert abs(eps[0].depth - 0.1) < 1e-12

    def test_episodes_frame_and_duration_summary(self):
        r = pd.Series([0.10, -0.05, -0.04, 0.02, 0.03, 0.05, -0.10, 0.02])
        eps = kr.drawdown_episodes(r)
        fr = kr.episodes_frame(eps)
        assert fr.index.name == "peak" and len(fr) == 2
        assert {"depth", "down_periods", "recovery_periods",
                "total_periods", "recovered"} <= set(fr.columns)
        ds = kr.duration_summary(eps)
        assert {"count", "min", "median", "max", "mean"} <= set(ds.index)
        assert set(ds.columns) == {"depth", "down_periods",
                                   "recovery_periods", "total_periods"}
        # 未收复区间的恢复期不参与统计：count 少 1
        assert ds.loc["count", "recovery_periods"] == 1.0

    def test_empty_episodes_frame(self):
        empty = kr.episodes_frame([])
        assert empty.empty and "depth" in empty.columns


# --------------------------------------------------------------------------- #
# Calmar 与报告
# --------------------------------------------------------------------------- #
class TestCalmarAndReport:
    def test_calmar_known(self):
        path = kr.make_drawdown_path(up_periods=5, down_periods=4, recover_periods=9,
                                     up=0.02, down=-0.03)
        ann = math.expm1(float(np.mean(np.log1p(path.to_numpy()))) * 12)
        expected = ann / kr.max_drawdown(path)
        assert abs(kr.calmar_ratio(path, periods_per_year=12) - expected) < 1e-12

    def test_calmar_zero_mdd_positive_return(self):
        assert kr.calmar_ratio(pd.Series([0.01] * 10), periods_per_year=1) == float("inf")

    def test_drawdown_report_summary(self):
        path = kr.make_drawdown_path()
        rep = kr.drawdown_report(path, periods_per_year=252)
        s = rep.summary()
        assert {"max_drawdown", "current_drawdown", "longest_drawdown_periods",
                "time_underwater_ratio", "calmar_ratio", "n_episodes"} <= set(s)
        assert abs(s["max_drawdown"] - kr.max_drawdown(path)) < 1e-15
        assert s["n_episodes"] == float(len(rep.episodes))
        assert isinstance(rep.frame(), pd.DataFrame)

    def test_validation(self):
        panel = pd.DataFrame({"A": [0.01, 0.02], "B": [0.01, 0.02]})
        with pytest.raises(ValueError):
            kr.max_drawdown(panel)               # 多列输入
        with pytest.raises(ValueError):
            kr.wealth_curve(pd.Series([-1.5]))   # ≤ −100%
        with pytest.raises(ValueError):
            kr.drawdown_episodes(pd.Series([0.01, -0.005]), min_depth=-0.1)
        with pytest.raises(ValueError):
            kr.wealth_curve(pd.Series([0.01]), start=0.0)
        with pytest.raises(ValueError):
            kr.calmar_ratio(pd.Series([], dtype="float64"))
