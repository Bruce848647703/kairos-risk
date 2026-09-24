"""measures 模块测试：VaR/ES 单调性与已知分布数值校验、回归统计还原、相对风险指标。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import kairos_risk as kr

#: 标准正态 95% 分位（双侧常用常数）
Z95 = 1.6448536269514722


def _phi_cdf(x: float) -> float:
    """标准正态 CDF（纯 math.erf，用于交叉验证，不依赖被测代码的 scipy 路径）。"""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


# --------------------------------------------------------------------------- #
# 矩与波动
# --------------------------------------------------------------------------- #
class TestMoments:
    def test_volatility_matches_numpy(self, panel4):
        vol = kr.volatility(panel4)
        expected = panel4.to_numpy().std(axis=0, ddof=1)
        assert np.allclose(vol.to_numpy(), expected)

    def test_volatility_annualization_sqrt_t(self, t_rets):
        per = kr.volatility(t_rets)
        ann = kr.volatility(t_rets, periods_per_year=252)
        assert abs(ann - per * math.sqrt(252)) < 1e-12

    def test_same_shape_out(self, panel4, t_rets):
        vol = kr.volatility(panel4)
        assert isinstance(vol, pd.Series)
        assert list(vol.index) == list(panel4.columns)
        assert isinstance(kr.volatility(t_rets), float)
        assert isinstance(kr.volatility(t_rets.to_numpy()), float)

    def test_skew_kurtosis_normal_near_zero(self, norm_rets):
        assert abs(kr.skewness(norm_rets)) < 0.1
        assert abs(kr.kurtosis(norm_rets)) < 0.15  # 超额峰度，正态为 0

    def test_kurtosis_fat_tail_positive(self, t_rets):
        assert kr.kurtosis(t_rets) > 1.0

    def test_skewness_sign(self):
        left = pd.Series([-0.10, 0.01, 0.01, 0.01, 0.012, 0.008, 0.01, -0.06])
        right = -left
        assert kr.skewness(left) < 0 < kr.skewness(right)

    def test_correlation_matrix_properties(self, panel4):
        c = kr.correlation_matrix(panel4).to_numpy()
        assert np.allclose(np.diag(c), 1.0)
        assert np.allclose(c, c.T)
        off = c[~np.eye(len(c), dtype=bool)]
        assert off.min() >= -1.0 and off.max() <= 1.0

    def test_covariance_matrix_matches_pandas(self, panel4):
        cov = kr.covariance_matrix(panel4)
        assert np.allclose(cov.to_numpy(), panel4.cov().to_numpy(), atol=1e-18)
        assert list(cov.columns) == list(panel4.columns)

    def test_annualized_return_constant_growth(self):
        r = pd.Series([0.01] * 24)
        ann = kr.annualized_return(r, periods_per_year=12)
        assert abs(ann - (1.01 ** 12 - 1.0)) < 1e-12

    def test_annualized_return_rejects_total_loss(self):
        with pytest.raises(ValueError):
            kr.annualized_return(pd.Series([0.01, -1.0, 0.02]))


# --------------------------------------------------------------------------- #
# VaR / ES：单调性、符号约定与已知分布数值校验
# --------------------------------------------------------------------------- #
class TestVarEs:
    def test_historical_var_monotone_in_confidence(self, t_rets):
        vs = [kr.historical_var(t_rets, c) for c in (0.90, 0.95, 0.99)]
        assert vs[0] <= vs[1] <= vs[2]

    def test_positive_loss_convention(self, t_rets):
        assert kr.historical_var(t_rets, 0.95) > 0
        assert kr.expected_shortfall(t_rets, 0.95) > 0
        assert kr.parametric_var(t_rets, 0.95) > 0
        assert kr.modified_var(t_rets, 0.95) > 0

    def test_es_ge_var_same_confidence(self, t_rets):
        for c in (0.90, 0.95, 0.99):
            assert kr.expected_shortfall(t_rets, c) >= kr.historical_var(t_rets, c) - 1e-15

    def test_es_monotone_in_confidence(self, t_rets):
        es = [kr.expected_shortfall(t_rets, c) for c in (0.90, 0.95, 0.99)]
        assert es[0] <= es[1] <= es[2]

    def test_parametric_normal_known_value(self):
        # 直接给 (mean, std)：VaR = z·σ，与 scipy 无关的解析真值
        v = kr.parametric_var(mean=0.0, std=0.02, confidence=0.95, distribution="normal")
        assert abs(v - Z95 * 0.02) < 1e-10

    def test_parametric_normal_on_large_sample(self, norm_rets):
        sigma = float(norm_rets.std(ddof=1))
        pv = kr.parametric_var(norm_rets, 0.95, distribution="normal")
        hv = kr.historical_var(norm_rets, 0.95)
        assert abs(pv - Z95 * sigma) < 0.02 * sigma
        assert abs(hv - Z95 * sigma) < 0.05 * sigma  # 历史法含分位数抽样误差

    def test_normal_es_closed_form(self, norm_rets):
        mu, sigma = float(norm_rets.mean()), float(norm_rets.std(ddof=1))
        es = kr.expected_shortfall(norm_rets, 0.95, method="normal")
        alpha = 0.05
        phi = math.exp(-0.5 * Z95 * Z95) / math.sqrt(2.0 * math.pi)
        theory = -mu + sigma * phi / alpha
        assert abs(es - theory) < 1e-10
        assert es >= kr.parametric_var(norm_rets, 0.95) - 1e-12

    def test_sqrt_time_scaling_zero_mean(self):
        v1 = kr.parametric_var(mean=0.0, std=0.01, confidence=0.95, periods_per_year=1)
        v10 = kr.parametric_var(mean=0.0, std=0.01, confidence=0.95, periods_per_year=10)
        assert abs(v10 - v1 * math.sqrt(10)) < 1e-12

    def test_student_t_fatter_at_high_confidence(self):
        pn = kr.parametric_var(mean=0.0, std=0.01, confidence=0.999, distribution="normal")
        pt = kr.parametric_var(mean=0.0, std=0.01, confidence=0.999,
                               distribution="student_t", df=5.0)
        assert pt > pn
        # t(5) 0.1% 分位 ≈ −5.8945，尺度按 √((ν−2)/ν) 方差校准
        assert abs(pt - 5.8945 * 0.01 * math.sqrt(3.0 / 5.0)) < 1e-5

    def test_student_t_es_above_var_and_normal_es(self, t_rets):
        es_t = kr.expected_shortfall(t_rets, 0.95, method="student_t", df=5.0)
        es_n = kr.expected_shortfall(t_rets, 0.95, method="normal")
        pv_t = kr.parametric_var(t_rets, 0.95, distribution="student_t", df=5.0)
        assert es_t >= pv_t - 1e-12
        assert es_t > es_n  # 肥尾数据的 t-ES 高于正态 ES

    def test_implied_df_from_kurtosis(self, t_rets):
        # 不给 df 时由超额峰度反推，结果应与显式 df=5 同量级
        auto = kr.parametric_var(t_rets, 0.99, distribution="student_t")
        explicit = kr.parametric_var(t_rets, 0.99, distribution="student_t", df=5.0)
        assert abs(auto - explicit) / explicit < 0.25

    def test_modified_var_amplifies_left_tail(self):
        mix = kr.make_mixture_returns(n=20000, seed=13)  # 负偏 + 肥尾
        mv = kr.modified_var(mix, 0.95)
        pv = kr.parametric_var(mix, 0.95, distribution="normal")
        assert mv > pv

    def test_cornish_fisher_z(self):
        z95 = kr.cornish_fisher_z(0.95, 0.0, 0.0)
        assert abs(z95 - (-Z95)) < 1e-9                    # 零偏零超额峰 -> 正态分位
        assert kr.cornish_fisher_z(0.95, -1.0, 0.0) < z95  # 负偏 -> 左尾更负
        # 肥尾修正的拐点在 |z|=√3：99% 档（|z|>√3）正超额峰度放大左尾
        z99 = kr.cornish_fisher_z(0.99, 0.0, 0.0)
        assert kr.cornish_fisher_z(0.99, 0.0, 3.0) < z99
        assert kr.cornish_fisher_z(0.99, -1.0, 3.0) < kr.cornish_fisher_z(0.99, 0.0, 3.0)

    def test_cvar_alias(self):
        assert kr.cvar is kr.expected_shortfall

    def test_validation(self, t_rets):
        for bad in (0.0, 1.0, -0.1, 1.2):
            with pytest.raises(ValueError):
                kr.historical_var(t_rets, bad)
        with pytest.raises(ValueError):
            kr.parametric_var(t_rets, 0.95, distribution="laplace")
        with pytest.raises(ValueError):
            kr.parametric_var(mean=0.0, std=0.01, confidence=0.95,
                              distribution="student_t", df=2.0)
        with pytest.raises(ValueError):
            kr.expected_shortfall(t_rets, 0.95, method="empirical")
        with pytest.raises(ValueError):
            kr.modified_var(pd.Series([0.01, -0.01, 0.02]))  # < 4 个观测


# --------------------------------------------------------------------------- #
# 回归：beta / alpha / R² 还原
# --------------------------------------------------------------------------- #
class TestRegression:
    def test_beta_recovery_on_constructed_data(self):
        rng = np.random.default_rng(9)
        n = 4000
        bm = pd.Series(rng.standard_normal(n) * 0.01, name="bm")
        asset = pd.Series(1.3 * bm.to_numpy() + rng.standard_normal(n) * 0.004,
                          index=bm.index)
        b = kr.beta(asset, bm)
        assert abs(b - 1.3) < 0.05

    def test_exact_linear_identity(self):
        bm = pd.Series(np.linspace(-0.02, 0.02, 101), name="bm")
        asset = pd.Series(0.0004 + 2.0 * bm.to_numpy(), index=bm.index)
        st = kr.regression_stats(asset, bm, periods_per_year=1)
        assert abs(st.beta - 2.0) < 1e-9
        assert abs(st.alpha - 0.0004) < 1e-12
        assert abs(st.r2 - 1.0) < 1e-12
        assert st.resid_std < 1e-12
        assert st.n_obs == 101

    def test_alpha_recovery_and_annualization(self):
        rng = np.random.default_rng(10)
        n = 4000
        bm = pd.Series(rng.standard_normal(n) * 0.01)
        asset = pd.Series(0.0003 + 0.8 * bm.to_numpy() + rng.standard_normal(n) * 0.005)
        a_per = kr.jensen_alpha(asset, bm, periods_per_year=252, annualized=False)
        a_ann = kr.jensen_alpha(asset, bm, periods_per_year=252, annualized=True)
        assert abs(a_per - 0.0003) < 2.5e-4
        assert abs(a_ann - a_per * 252) < 1e-12

    def test_r2_equals_corr_squared(self):
        rng = np.random.default_rng(11)
        n = 2000
        bm = pd.Series(rng.standard_normal(n) * 0.01)
        asset = pd.Series(0.5 * bm.to_numpy() + rng.standard_normal(n) * 0.01)
        st = kr.regression_stats(asset, bm)
        corr = float(np.corrcoef(asset.to_numpy(), bm.to_numpy())[0, 1])
        assert abs(st.r2 - corr ** 2) < 1e-10
        assert abs(st.corr - corr) < 1e-10
        assert 0.0 < st.r2 < 1.0

    def test_beta_panel_returns_series(self, panel4):
        bm = panel4.mean(axis=1)
        betas = kr.beta(panel4, bm)
        assert isinstance(betas, pd.Series)
        assert list(betas.index) == list(panel4.columns)
        assert (betas > 0).all()

    def test_regression_table(self, panel4):
        tab = kr.regression_table(panel4, panel4.mean(axis=1))
        assert list(tab.index) == [str(c) for c in panel4.columns]
        assert {"beta", "alpha", "alpha_annualized", "r2", "corr",
                "resid_std", "n_obs"} <= set(tab.columns)
        assert (tab["r2"] >= 0).all() and (tab["r2"] <= 1).all()

    def test_index_alignment(self):
        idx_a = pd.date_range("2020-01-01", periods=10)
        idx_b = pd.date_range("2020-01-06", periods=10)
        a = pd.Series(np.arange(10.0) * 0.001, index=idx_a)
        b = pd.Series(np.arange(10.0) * 0.002, index=idx_b)
        overlap = a.index.intersection(b.index)
        st = kr.regression_stats(a, b)
        assert st.n_obs == len(overlap) == 5

    def test_validation(self):
        with pytest.raises(ValueError):
            kr.regression_stats(pd.Series([0.01, 0.02]), pd.Series([0.01, 0.02]))
        with pytest.raises(ValueError):
            kr.regression_stats(pd.Series([0.01, -0.02, 0.03]),
                                pd.Series([0.0, 0.0, 0.0]))  # 基准零方差


# --------------------------------------------------------------------------- #
# 跟踪误差 / 信息比率 / 下行风险
# --------------------------------------------------------------------------- #
class TestRelativeRisk:
    @pytest.fixture()
    def pair(self):
        rng = np.random.default_rng(12)
        bm = pd.Series(rng.standard_normal(1000) * 0.008, name="bm")
        r = bm + pd.Series(rng.standard_normal(1000) * 0.002 + 0.0004, index=bm.index)
        return r, bm

    def test_tracking_error_formula(self, pair):
        r, bm = pair
        expected = float((r - bm).std(ddof=1))
        assert abs(kr.tracking_error(r, bm) - expected) < 1e-12
        ann = kr.tracking_error(r, bm, periods_per_year=252)
        assert abs(ann - expected * math.sqrt(252)) < 1e-10

    def test_information_ratio_formula(self, pair):
        r, bm = pair
        active = r - bm
        expected = float(active.mean()) * math.sqrt(252) / float(active.std(ddof=1))
        assert abs(kr.information_ratio(r, bm, periods_per_year=252) - expected) < 1e-10

    def test_identical_series(self, pair):
        r, _ = pair
        assert kr.tracking_error(r, r) == 0.0
        with pytest.raises(ValueError):
            kr.information_ratio(r, r)  # TE=0 -> IR 无定义

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            kr.tracking_error(np.zeros(10), np.zeros(11))

    def test_downside_deviation_known_value(self):
        r = pd.Series([0.01, -0.02, 0.0, -0.01, 0.03])
        dd = kr.downside_deviation(r, mar=0.0)
        assert abs(dd - math.sqrt((0.0004 + 0.0001) / 5)) < 1e-15

    def test_downside_zero_when_all_above_mar(self):
        r = pd.Series([0.01, 0.02, 0.03])
        assert kr.downside_deviation(r, 0.0) == 0.0
        assert kr.sortino_ratio(r, 0.0) == float("inf")   # 无下行且均值为正
        assert kr.sortino_ratio(-r, 0.0) < 0              # 全下行 -> 负 Sortino
        assert kr.sortino_ratio(pd.Series([0.0, 0.0]), 0.0) == 0.0  # 无下行且均值非正

    def test_sortino_sign(self):
        neg = pd.Series([-0.01, 0.005, -0.02, -0.005])
        assert kr.sortino_ratio(neg, 0.0, periods_per_year=1) < 0

    def test_sharpe_sign_and_zero_vol(self, norm_rets):
        assert abs(kr.sharpe_ratio(norm_rets, periods_per_year=1)) < 0.1  # 零均值
        neg = pd.Series([-0.010, 0.005, -0.020, -0.008, 0.002, -0.015, 0.001, -0.012])
        assert kr.sharpe_ratio(neg, periods_per_year=1) < 0
        assert kr.sharpe_ratio(-neg, periods_per_year=1) > 0
        with pytest.raises(ValueError):
            kr.sharpe_ratio(pd.Series([0.01, 0.01, 0.01]))  # 零波动


# --------------------------------------------------------------------------- #
# 一键摘要
# --------------------------------------------------------------------------- #
class TestRiskReport:
    def test_report_shape_and_metrics(self, panel4):
        rep = kr.risk_report(panel4)
        assert list(rep.columns) == [str(c) for c in panel4.columns]
        needed = {"annualized_return", "volatility", "sharpe", "skewness",
                  "excess_kurtosis", "historical_var", "parametric_var_normal",
                  "parametric_var_student_t", "modified_var", "es_historical",
                  "es_normal", "es_student_t", "downside_deviation", "sortino"}
        assert needed <= set(rep.index)
        assert (rep.loc["es_historical"] >= rep.loc["historical_var"]).all()
        assert np.all(np.isfinite(rep.to_numpy()))

    def test_report_single_series(self, t_rets):
        rep = kr.risk_report(t_rets)
        assert rep.shape[1] == 1
        assert rep.loc["excess_kurtosis"].iloc[0] > 1.0
