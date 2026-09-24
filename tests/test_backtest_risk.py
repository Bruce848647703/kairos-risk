"""backtest_risk 模块测试：PSR/DSR 行为性质、最小跟踪长度、损失概率与胜率区间。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import kairos_risk as kr
from kairos_risk.backtest_risk import _moments_of


# --------------------------------------------------------------------------- #
# 夏普与标准误
# --------------------------------------------------------------------------- #
class TestSharpeBasics:
    def test_per_period_sharpe_known(self):
        r = pd.Series([0.01, -0.01, 0.02, -0.02, 0.015])
        expected = float(r.mean() / r.std(ddof=1))
        assert abs(kr.per_period_sharpe(r) - expected) < 1e-15
        assert abs(kr.per_period_sharpe(r, risk_free=0.001)
                   - float(r.mean() - 0.001) / float(r.std(ddof=1))) < 1e-15

    def test_per_period_sharpe_validation(self):
        with pytest.raises(ValueError):
            kr.per_period_sharpe(pd.Series([0.01]))            # 观测不足
        with pytest.raises(ValueError):
            kr.per_period_sharpe(pd.Series([0.01, 0.01, 0.01]))  # 零方差

    def test_sharpe_std_error_formula(self):
        se = kr.sharpe_std_error(100, 0.1, skew=0.0, kurtosis=3.0)
        expected = math.sqrt((1.0 + 0.5 * 0.01) / 99.0)
        assert abs(se - expected) < 1e-15
        # 肥尾（γ₄>3）提高 SE -> 同样的夏普显著性更低
        assert kr.sharpe_std_error(100, 0.1, kurtosis=8.0) > se
        with pytest.raises(ValueError):
            kr.sharpe_std_error(1, 0.1)


# --------------------------------------------------------------------------- #
# PSR
# --------------------------------------------------------------------------- #
class TestPSR:
    def test_zero_sharpe_gives_half(self):
        rng = np.random.default_rng(6)
        z = rng.standard_normal(2000) * 0.01
        z = z - z.mean()                     # 精确零均值 -> SR=0 -> PSR=0.5
        assert abs(kr.psr_from_returns(z) - 0.5) < 1e-9

    def test_high_sharpe_near_one(self):
        rng = np.random.default_rng(4)
        r = pd.Series(rng.standard_normal(3000) * 0.01 + 0.003)   # 每期 SR≈0.3
        assert kr.psr_from_returns(r) > 0.999

    def test_analytic_value(self):
        sr, T = 0.1, 100
        z = sr * math.sqrt(T - 1) / math.sqrt(1.0 + (3.0 - 1.0) / 4.0 * sr * sr)
        expected = 0.5 * math.erfc(-z / math.sqrt(2.0))          # Φ(z)，纯 math
        assert abs(kr.probabilistic_sharpe_ratio(sr, T) - expected) < 1e-12

    def test_bounds_and_monotonicity(self):
        vals = [kr.probabilistic_sharpe_ratio(0.05, T) for T in (50, 200, 1000, 5000)]
        assert all(0.0 < v < 1.0 for v in vals)
        assert vals == sorted(vals)                              # T 越大越显著
        srs = [kr.probabilistic_sharpe_ratio(s, 1000) for s in (0.01, 0.05, 0.1)]
        assert srs == sorted(srs)                                # SR 越大越显著

    def test_kurtosis_penalty(self):
        base = kr.probabilistic_sharpe_ratio(0.08, 500, skew=0.0, kurtosis=3.0)
        fat = kr.probabilistic_sharpe_ratio(0.08, 500, skew=0.0, kurtosis=10.0)
        assert fat < base

    def test_benchmark_sharpe_lowers_psr(self):
        p0 = kr.probabilistic_sharpe_ratio(0.08, 500, benchmark_sharpe=0.0)
        p1 = kr.probabilistic_sharpe_ratio(0.08, 500, benchmark_sharpe=0.05)
        assert p1 < p0
        # SR 不高于基准时 PSR ≤ 0.5
        assert kr.probabilistic_sharpe_ratio(0.02, 500, benchmark_sharpe=0.02) == \
            pytest.approx(0.5)

    def test_minimum_track_length_roundtrip(self):
        tmin = kr.minimum_track_length(0.05, confidence=0.95)
        assert kr.probabilistic_sharpe_ratio(0.05, math.ceil(tmin)) >= 0.95 - 1e-9
        assert kr.probabilistic_sharpe_ratio(0.05, max(2, math.floor(tmin) - 1)) < 0.95
        # 置信度越高所需跟踪越长
        assert kr.minimum_track_length(0.05, 0.99) > tmin
        with pytest.raises(ValueError):
            kr.minimum_track_length(0.0, 0.95)                   # SR == 基准
        with pytest.raises(ValueError):
            kr.minimum_track_length(-0.1, 0.95, benchmark_sharpe=0.0)


# --------------------------------------------------------------------------- #
# DSR：随试验次数增加而下降
# --------------------------------------------------------------------------- #
class TestDSR:
    def test_decreasing_in_trials(self):
        dsrs = [kr.deflated_sharpe_ratio(0.05, 1000, n_trials=n).dsr
                for n in (1, 2, 5, 20, 100)]
        assert all(a > b for a, b in zip(dsrs, dsrs[1:]))
        assert all(0.0 < d < 1.0 for d in dsrs)

    def test_single_trial_equals_psr(self):
        d = kr.deflated_sharpe_ratio(0.05, 1000, n_trials=1)
        assert d.dsr == d.psr
        assert d.haircut == 0.0
        assert d.sr0 == 0.0

    def test_dsr_ne_above_psr(self):
        d = kr.deflated_sharpe_ratio(0.05, 1000, n_trials=50)
        assert d.dsr <= d.psr
        assert d.haircut >= 0.0

    def test_expected_max_sharpe(self):
        assert kr.expected_max_sharpe(1) == 0.0
        vals = [kr.expected_max_sharpe(n) for n in (2, 5, 10, 50, 100)]
        assert vals == sorted(vals)
        assert all(v > 0 for v in vals)
        # 按 √V 缩放
        assert abs(kr.expected_max_sharpe(20, 4.0)
                   - 2.0 * kr.expected_max_sharpe(20, 1.0)) < 1e-12
        with pytest.raises(ValueError):
            kr.expected_max_sharpe(0)
        with pytest.raises(ValueError):
            kr.expected_max_sharpe(5, sharpe_variance=-1.0)

    def test_trial_sharpes_drive_n_and_variance(self):
        trials = [0.01, 0.02, 0.05, 0.03, 0.04, 0.02]
        d = kr.deflated_sharpe_ratio(0.05, 1000, trial_sharpes=trials)
        assert d.n_trials == len(trials)
        assert abs(d.sharpe_variance - float(np.var(trials, ddof=1))) < 1e-15
        # N 至少与显式 n_trials 一致
        d2 = kr.deflated_sharpe_ratio(0.05, 1000, n_trials=20, trial_sharpes=trials)
        assert d2.n_trials == 20

    def test_benchmark_floor(self):
        d = kr.deflated_sharpe_ratio(0.05, 1000, n_trials=2, benchmark_sharpe=0.5)
        assert d.sr0 == 0.5            # 外部基准高于 E[max SR] 时取基准
        assert d.dsr < 0.5

    def test_dsr_from_returns_consistent(self):
        rng = np.random.default_rng(4)
        r = pd.Series(rng.standard_normal(2000) * 0.01 + 0.001)
        d = kr.dsr_from_returns(r, n_trials=10)
        sr = kr.per_period_sharpe(r)
        skew, kurt = _moments_of(r)
        d2 = kr.deflated_sharpe_ratio(sr, len(r), n_trials=10, skew=skew, kurtosis=kurt)
        assert abs(d.dsr - d2.dsr) < 1e-15
        assert abs(d.psr - d2.psr) < 1e-15
        assert 0.0 < d.dsr <= d.psr <= 1.0

    def test_trials_for_target_dsr(self):
        nt = kr.trials_for_target_dsr(0.1, 2000, target_dsr=0.95)
        assert nt >= 10
        assert kr.deflated_sharpe_ratio(0.1, 2000, n_trials=nt).dsr >= 0.95
        assert kr.deflated_sharpe_ratio(0.1, 2000, n_trials=nt * 2 + 10).dsr < 0.95
        # 夏普太低：N=1 都达不到目标 -> 预算为 0
        assert kr.trials_for_target_dsr(0.001, 50, target_dsr=0.95) == 0


# --------------------------------------------------------------------------- #
# 损失概率与胜率区间
# --------------------------------------------------------------------------- #
class TestLossAndWinRate:
    def test_loss_probability_known(self):
        r = pd.Series([0.01, -0.01, 0.02, -0.02])
        assert abs(kr.loss_probability(r, 0.0) - 0.5) < 1e-15
        assert abs(kr.loss_probability(r, 0.015) - 0.25) < 1e-15
        with pytest.raises(ValueError):
            kr.loss_probability(r, method="bootstrap")

    def test_loss_probability_normal_approx(self, norm_rets):
        p_emp = kr.loss_probability(norm_rets, 0.0)
        p_norm = kr.loss_probability(norm_rets, 0.0, method="normal")
        assert abs(p_emp - 0.5) < 0.02
        assert abs(p_norm - p_emp) < 0.02
        # 更深的损失水平 -> 概率更小
        assert kr.loss_probability(norm_rets, 0.03, method="normal") < p_norm

    def test_probability_of_min_loss(self, norm_rets):
        p1 = kr.probability_of_min_loss(norm_rets, horizon=1)
        p21 = kr.probability_of_min_loss(norm_rets, horizon=21)
        p200 = kr.probability_of_min_loss(norm_rets, horizon=200)
        assert 0.0 < p1 <= p21 <= p200 <= 1.0     # 对 horizon 单调不减
        n = len(norm_rets)
        assert abs(p1 - 1.0 / n) < 2.0 / n        # 经验口径 ≈ 1/n
        with pytest.raises(ValueError):
            kr.probability_of_min_loss(norm_rets, horizon=0)

    def test_win_rate(self):
        r = pd.Series([0.01, -0.01, 0.0, 0.02])
        assert abs(kr.win_rate(r) - 0.5) < 1e-15            # 严格大于 0：2/4
        assert abs(kr.win_rate(r, threshold=-0.005) - 0.75) < 1e-15  # 3/4（0.0 也算赢）
        assert abs(kr.win_rate(r, threshold=0.0) - 0.5) < 1e-15      # 0.0 不算赢

    def test_win_rate_ci_methods(self):
        ci_w = kr.win_rate_ci(wins=30, n=50, method="wilson")
        ci_n = kr.win_rate_ci(wins=30, n=50, method="normal")
        ci_c = kr.win_rate_ci(wins=30, n=50, method="clopper_pearson")
        for ci in (ci_w, ci_n, ci_c):
            assert 0.0 <= ci.lower <= ci.point <= ci.upper <= 1.0
            assert ci.contains(ci.point)
            assert abs(ci.point - 0.6) < 1e-15
            assert ci.width > 0
        # 精确区间（Clopper-Pearson）比 Wilson 更保守
        assert ci_c.lower <= ci_w.lower + 1e-9
        assert ci_c.upper >= ci_w.upper - 1e-9

    def test_win_rate_ci_edges(self):
        # Clopper-Pearson 在 k=0 / k=n 时区间精确贴边
        ci0 = kr.win_rate_ci(wins=0, n=50, method="clopper_pearson")
        cin = kr.win_rate_ci(wins=50, n=50, method="clopper_pearson")
        assert ci0.lower == 0.0 and 0.0 < ci0.upper < 0.2
        assert cin.upper == 1.0 and 0.8 < cin.lower < 1.0
        # Wilson 在 k=0 时下界恰为 0（center 与半宽解析相消），上界仍是正的小量
        w0 = kr.win_rate_ci(wins=0, n=50, method="wilson")
        assert 0.0 <= w0.lower < 1e-9 < w0.upper < 0.2

    def test_win_rate_ci_higher_confidence_wider(self):
        c90 = kr.win_rate_ci(wins=30, n=50, confidence=0.90)
        c99 = kr.win_rate_ci(wins=30, n=50, confidence=0.99)
        assert c99.width > c90.width

    def test_win_rate_ci_from_returns(self, t_rets):
        ci = kr.win_rate_ci(t_rets)
        assert ci.wins == int((t_rets > 0).sum())
        assert ci.n == len(t_rets)

    def test_win_rate_ci_validation(self):
        with pytest.raises(ValueError):
            kr.win_rate_ci(wins=5, n=0)
        with pytest.raises(ValueError):
            kr.win_rate_ci(wins=6, n=5)
        with pytest.raises(ValueError):
            kr.win_rate_ci(wins=1, n=5, method="bayes")
        with pytest.raises(ValueError):
            kr.win_rate_ci()                      # 既无 returns 也无 (wins, n)
        with pytest.raises(ValueError):
            kr.win_rate_ci(wins=1, n=5, confidence=1.5)

    def test_report(self):
        rng = np.random.default_rng(4)
        r = pd.Series(rng.standard_normal(1500) * 0.01 + 0.0015)
        rep = kr.backtest_risk_report(r, n_trials=10, horizon=21)
        needed = {"n_obs", "sharpe_per_period", "skewness", "kurtosis", "psr", "dsr",
                  "dsr_haircut", "min_track_length", "win_rate", "win_rate_lower",
                  "win_rate_upper", "loss_probability", "probability_of_min_loss"}
        assert needed <= set(rep.index)
        assert set(rep.columns) == {"value", "detail"}
        assert rep.loc["n_obs", "value"] == 1500.0
        for key in ("psr", "dsr"):
            assert 0.0 <= rep.loc[key, "value"] <= 1.0
        assert rep.loc["dsr", "value"] <= rep.loc["psr", "value"] + 1e-12

    def test_report_negative_sharpe_inf_track(self):
        rng = np.random.default_rng(5)
        r = pd.Series(rng.standard_normal(500) * 0.01 - 0.002)   # 负夏普
        rep = kr.backtest_risk_report(r)
        assert rep.loc["min_track_length", "value"] == float("inf")
