"""decomposition 模块测试：欧拉可加性、分散化比率、成分 VaR/ES 与尾部样本。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import kairos_risk as kr


# --------------------------------------------------------------------------- #
# 波动分解（MRC / CRC）
# --------------------------------------------------------------------------- #
class TestComponentRisk:
    def test_euler_additivity(self, w4, cov4):
        rd = kr.component_risk(w4, cov4)
        assert abs(float(rd.component.sum()) - rd.volatility) < 1e-15
        assert abs(float(rd.pct.sum()) - 1.0) < 1e-12

    def test_volatility_quadratic_form(self, w4, cov4):
        rd = kr.component_risk(w4, cov4)
        expected = math.sqrt(float(w4.to_numpy() @ cov4.to_numpy() @ w4.to_numpy()))
        assert abs(rd.volatility - expected) < 1e-15

    def test_marginal_formula(self, w4, cov4):
        rd = kr.component_risk(w4, cov4)
        expected = cov4.to_numpy() @ w4.to_numpy() / rd.volatility
        assert np.allclose(np.asarray(rd.marginal, dtype="float64"), expected, atol=1e-15)
        # component_i = w_i · marginal_i
        assert np.allclose(np.asarray(rd.component, dtype="float64"),
                           w4.to_numpy() * expected, atol=1e-15)

    def test_two_asset_closed_form(self):
        sig = pd.DataFrame([[4e-4, 1e-4], [1e-4, 1e-4]], index=["A", "B"], columns=["A", "B"])
        w = pd.Series([0.5, 0.5], index=["A", "B"])
        rd = kr.component_risk(w, sig)
        vol = math.sqrt(0.25 * 4e-4 + 2 * 0.25 * 1e-4 + 0.25 * 1e-4)
        assert abs(rd.volatility - vol) < 1e-15
        assert abs(float(rd.component.sum()) - vol) < 1e-15

    def test_diversification_ratio_ge_one(self, w4, cov4):
        assert kr.diversification_ratio(w4, cov4) >= 1.0 - 1e-12

    def test_dr_equals_one_for_single_asset(self, cov4, panel4):
        w = pd.Series([1.0, 0.0, 0.0, 0.0], index=panel4.columns)
        assert abs(kr.diversification_ratio(w, cov4) - 1.0) < 1e-12

    def test_dr_equals_one_for_perfect_correlation(self):
        rng = np.random.default_rng(1)
        r1 = pd.Series(rng.standard_normal(500) * 0.01, name="A")
        r2 = pd.Series(2.0 * r1.to_numpy(), index=r1.index, name="B")
        cov = kr.covariance_matrix(pd.concat([r1, r2], axis=1))
        w = pd.Series([0.5, 0.5], index=["A", "B"])
        assert abs(kr.diversification_ratio(w, cov) - 1.0) < 1e-9

    def test_effective_n_equal_risk(self):
        sig = np.eye(3) * 1e-4
        w = np.full(3, 1.0 / 3.0)
        rd = kr.component_risk(w, sig)
        assert np.allclose(np.asarray(rd.pct), 1.0 / 3.0, atol=1e-12)
        assert abs(rd.herfindahl - 1.0 / 3.0) < 1e-12
        assert abs(kr.risk_effective_n(w, sig) - 3.0) < 1e-9

    def test_effective_n_bounds(self, w4, cov4):
        n_eff = kr.risk_effective_n(w4, cov4)
        assert 1.0 <= n_eff <= 4.0 + 1e-9

    def test_ndarray_in_ndarray_out(self, cov4):
        w = np.array([0.4, 0.3, 0.2, 0.1])
        rd = kr.component_risk(w, cov4.to_numpy())
        assert isinstance(rd.component, np.ndarray) and not isinstance(rd.component, pd.Series)
        assert isinstance(kr.marginal_risk(w, cov4.to_numpy()), np.ndarray)

    def test_series_weights_aligned_by_name(self, cov4, panel4):
        w_shuffled = pd.Series([0.1, 0.2, 0.3, 0.4], index=list(panel4.columns)[::-1])
        w_ordered = w_shuffled.reindex(panel4.columns)
        a = kr.component_risk(w_shuffled, cov4)
        b = kr.component_risk(w_ordered, cov4)
        assert abs(a.volatility - b.volatility) < 1e-18
        assert np.allclose(a.pct.reindex(panel4.columns).to_numpy(),
                           b.pct.to_numpy(), atol=1e-15)

    def test_validation(self, cov4):
        with pytest.raises(ValueError):
            kr.component_risk(np.array([0.5, 0.5]), cov4)          # 维度不符
        with pytest.raises(ValueError):
            kr.component_risk(pd.Series([0.5, 0.5], index=["A", "B"]), cov4)  # 缺资产
        with pytest.raises(ValueError):
            kr.component_risk(np.array([0.5, 0.5]), np.zeros((2, 2)))        # 零波动
        with pytest.raises(ValueError):
            kr.component_risk(np.array([0.5, np.nan]), np.eye(2))            # NaN 权重


# --------------------------------------------------------------------------- #
# 尾部分解（成分 VaR / 成分 ES）
# --------------------------------------------------------------------------- #
class TestTailDecomposition:
    def test_component_es_additive_and_matches_measures(self, w4, panel4):
        ce = kr.component_es(w4, panel4, 0.95)
        assert abs(float(ce.component.sum()) - ce.value) < 1e-12
        assert abs(float(ce.pct.sum()) - 1.0) < 1e-12
        port = kr.portfolio_returns(panel4, w4)
        assert abs(ce.value - kr.expected_shortfall(port, 0.95)) < 1e-12

    def test_component_var_additive_and_matches_measures(self, w4, panel4):
        cv = kr.component_var(w4, panel4, 0.95)
        assert abs(float(cv.component.sum()) - cv.value) < 1e-12
        port = kr.portfolio_returns(panel4, w4)
        assert abs(cv.value - kr.historical_var(port, 0.95)) < 1e-12
        ce = kr.component_es(w4, panel4, 0.95)
        assert cv.value <= ce.value + 1e-15  # VaR ≤ ES

    def test_tail_metadata(self, w4, panel4):
        ce = kr.component_es(w4, panel4, 0.95)
        assert ce.measure == "es"
        assert ce.n_obs == len(panel4)
        assert 0 < ce.n_tail <= math.ceil(0.05 * len(panel4)) + 1
        assert ce.confidence == 0.95
        assert ce.threshold < 0

    def test_es_value_monotone_in_confidence(self, w4, panel4):
        v90 = kr.component_es(w4, panel4, 0.90).value
        v99 = kr.component_es(w4, panel4, 0.99).value
        assert v99 >= v90

    def test_portfolio_returns_formula(self, panel4, w4):
        port = kr.portfolio_returns(panel4, w4)
        assert isinstance(port, pd.Series)
        assert port.index.equals(panel4.index)
        assert np.allclose(port.to_numpy(), panel4.to_numpy() @ w4.to_numpy())

    def test_worst_tail_sample_matches_mask(self, w4, panel4):
        sub = kr.worst_tail_sample(w4, panel4, 0.95)
        port = kr.portfolio_returns(panel4, w4)
        thr = float(np.quantile(port.to_numpy(), 0.05))
        expected_idx = set(port.index[port.to_numpy() <= thr])
        assert set(sub.index) == expected_idx
        assert len(sub) >= 1
        # 子面板中的组合收益全部不优于阈值
        assert float((sub.to_numpy() @ w4.to_numpy()).max()) <= thr + 1e-15

    def test_ndarray_io(self, panel4):
        w = np.array([0.4, 0.3, 0.2, 0.1])
        ce = kr.component_es(w, panel4.to_numpy(), 0.95)
        assert isinstance(ce.component, np.ndarray) and not isinstance(ce.component, pd.Series)

    def test_validation(self, panel4):
        with pytest.raises(ValueError):
            kr.component_es(np.array([0.5, 0.5]), panel4, 0.95)
        with pytest.raises(ValueError):
            kr.component_es(pd.Series(0.25, index=panel4.columns), panel4, 1.5)
