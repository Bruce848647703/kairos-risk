"""factor 模块测试：已知暴露数据上还原 beta/alpha，因子-特质风险分解精确可加。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import kairos_risk as kr


# --------------------------------------------------------------------------- #
# 模型估计与还原
# --------------------------------------------------------------------------- #
class TestFit:
    def test_exposure_recovery(self, factor_env):
        fp, b = factor_env
        fm = kr.fit_factor_model(fp.assets, fp.factors)
        assert float(np.abs(fm.exposures.to_numpy() - b).max()) < 0.03
        assert list(fm.exposures.index) == list(fp.assets.columns)
        assert list(fm.exposures.columns) == list(fp.factors.columns)

    def test_alpha_recovery(self, factor_env):
        fp, _ = factor_env
        fm = kr.fit_factor_model(fp.assets, fp.factors)
        assert float(np.abs(fm.alphas.to_numpy() - 0.0002).max()) < 4e-4

    def test_r2_and_idio_recovery(self, factor_env):
        fp, _ = factor_env
        fm = kr.fit_factor_model(fp.assets, fp.factors)
        assert ((fm.r2 > 0.5) & (fm.r2 <= 1.0)).all()
        assert float(np.abs(fm.resid_std.to_numpy() - 0.004).max()) < 8e-4
        assert (fm.idio_var >= 0).all()

    def test_factor_cov_recovery(self, factor_env):
        fp, _ = factor_env
        fm = kr.fit_factor_model(fp.assets, fp.factors)
        truth = np.eye(2) * 0.010 ** 2
        assert float(np.abs(fm.factor_cov.to_numpy() - truth).max()) < 0.15 * 0.010 ** 2

    def test_implied_cov_close_to_truth(self, factor_env):
        fp, _ = factor_env
        fm = kr.fit_factor_model(fp.assets, fp.factors)
        err = float(np.abs(fm.implied_cov().to_numpy() - fp.true_cov.to_numpy()).max())
        scale = float(np.abs(np.diag(fp.true_cov.to_numpy())).max())
        assert err / scale < 0.08
        assert fm.total_var() == pytest.approx(float(np.trace(fp.true_cov.to_numpy())), rel=0.08)

    def test_true_cov_algebra(self, factor_env):
        fp, b = factor_env
        f_cov = np.diag(fp.factor_std.to_numpy() ** 2)
        d = np.diag(fp.idio_std.to_numpy() ** 2)
        assert np.allclose(fp.true_cov.to_numpy(), b @ f_cov @ b.T + d)

    def test_fit_alpha_false(self, factor_env):
        fp, b = factor_env
        fm = kr.fit_factor_model(fp.assets, fp.factors, fit_alpha=False)
        assert not fm.has_alpha
        assert float(np.abs(fm.alphas.to_numpy()).max()) == 0.0
        # α 很小，强制过原点对暴露影响有限
        assert float(np.abs(fm.exposures.to_numpy() - b).max()) < 0.05

    def test_market_model_single_factor(self):
        rng = np.random.default_rng(21)
        n = 3000
        mkt = pd.Series(rng.standard_normal(n) * 0.01, name="MKT")
        assets = pd.DataFrame({
            "A": 1.5 * mkt.to_numpy() + rng.standard_normal(n) * 0.004,
            "B": -0.8 * mkt.to_numpy() + rng.standard_normal(n) * 0.006,
        })
        mm = kr.market_model(assets, mkt)
        assert mm.n_factors == 1
        assert abs(float(mm.exposures.loc["A", "MKT"]) - 1.5) < 0.1
        assert abs(float(mm.exposures.loc["B", "MKT"]) + 0.8) < 0.15

    def test_market_model_accepts_ndarray(self):
        rng = np.random.default_rng(22)
        n = 500
        mkt = rng.standard_normal(n) * 0.01
        assets = pd.DataFrame({"A": 1.2 * mkt + rng.standard_normal(n) * 0.004})
        mm = kr.market_model(assets, mkt)
        assert abs(float(mm.exposures.loc["A", "MKT"]) - 1.2) < 0.2

    def test_duplicate_column_raises(self, factor_env):
        fp, _ = factor_env
        renamed = fp.assets.rename(columns={fp.assets.columns[0]: fp.factors.columns[0]})
        with pytest.raises(ValueError):
            kr.fit_factor_model(renamed, fp.factors)

    def test_too_few_obs_raises(self, factor_env):
        fp, _ = factor_env
        with pytest.raises(ValueError):
            kr.fit_factor_model(fp.assets.iloc[:4], fp.factors.iloc[:4])

    def test_collinear_factors_raise(self):
        rng = np.random.default_rng(23)
        n = 200
        f1 = rng.standard_normal(n) * 0.01
        factors = pd.DataFrame({"F1": f1, "F2": 2.0 * f1})  # 完全共线
        assets = pd.DataFrame({"A": f1 + rng.standard_normal(n) * 0.002})
        with pytest.raises(ValueError):
            kr.fit_factor_model(assets, factors)


# --------------------------------------------------------------------------- #
# 组合层面的因子 / 特质风险分解
# --------------------------------------------------------------------------- #
class TestBreakdown:
    def test_variance_decomposition_additive(self, factor_env):
        fp, _ = factor_env
        fm = kr.fit_factor_model(fp.assets, fp.factors)
        w = pd.Series([0.4, 0.3, 0.2, 0.1], index=fp.assets.columns)
        brk = kr.factor_risk_decomposition(w, fm)
        # 因子方差 + 特质方差 = 总方差（代数恒等）
        assert abs(brk.factor_variance + brk.idio_variance - brk.total_variance) < 1e-22
        # 各自的欧拉贡献精确可加
        assert abs(float(brk.factor_contrib.sum()) - brk.factor_variance) < 1e-22
        assert abs(float(brk.idio_contrib.sum()) - brk.idio_variance) < 1e-22
        # 逐资产总贡献之和 = 组合波动（对隐含协方差的欧拉分解）
        assert abs(float(brk.asset_contrib.sum()) - brk.volatility) < 1e-14
        assert abs(brk.volatility - math.sqrt(brk.total_variance)) < 1e-18
        # 占比口径闭合
        assert abs(float(brk.factor_pct.sum()) + float(brk.idio_pct.sum()) - 1.0) < 1e-12
        assert abs(float(brk.asset_pct.sum()) - 1.0) < 1e-12
        assert 0.0 <= brk.factor_share <= 1.0

    def test_total_variance_equals_quadratic_form(self, factor_env):
        fp, _ = factor_env
        fm = kr.fit_factor_model(fp.assets, fp.factors)
        w = pd.Series([0.4, 0.3, 0.2, 0.1], index=fp.assets.columns)
        brk = kr.factor_risk_decomposition(w, fm)
        sigma = fm.implied_cov().to_numpy()
        wv = w.to_numpy()
        assert abs(brk.total_variance - float(wv @ sigma @ wv)) < 1e-20

    def test_portfolio_exposure_linear(self, factor_env):
        fp, _ = factor_env
        fm = kr.fit_factor_model(fp.assets, fp.factors)
        w = pd.Series([0.4, 0.3, 0.2, 0.1], index=fp.assets.columns)
        bp = kr.portfolio_exposure(w, fm)
        expected = fm.exposures.to_numpy().T @ w.to_numpy()
        assert np.allclose(bp.to_numpy(), expected)
        assert list(bp.index) == list(fp.factors.columns)

    def test_idio_dominant_when_exposure_near_zero(self):
        rng = np.random.default_rng(5)
        n = 800
        f = pd.DataFrame({"F": rng.standard_normal(n) * 0.01})
        y = pd.DataFrame({"A": rng.standard_normal(n) * 0.005,
                          "B": rng.standard_normal(n) * 0.007})
        fm = kr.fit_factor_model(y, f)
        brk = kr.factor_risk_decomposition(pd.Series([0.5, 0.5], index=y.columns), fm)
        assert brk.factor_share < 0.05

    def test_helper_functions_consistent(self, factor_env):
        fp, _ = factor_env
        fm = kr.fit_factor_model(fp.assets, fp.factors)
        w = pd.Series(0.25, index=fp.assets.columns)
        brk = kr.factor_risk_decomposition(w, fm)
        assert kr.factor_variance(w, fm) == brk.factor_variance
        assert kr.idiosyncratic_variance(w, fm) == brk.idio_variance

    def test_summary_tables(self, factor_env):
        fp, _ = factor_env
        fm = kr.fit_factor_model(fp.assets, fp.factors)
        tab = kr.factor_var_contributions(fm)
        assert list(tab.index) == list(fp.factors.columns)
        assert {"std", "variance", "mean_abs_corr"} <= set(tab.columns)
        summ = kr.exposure_summary(fm)
        assert list(summ.index) == list(fp.assets.columns)
        assert {"alpha", "r2", "idio_std"} <= set(summ.columns)

    def test_dimension_mismatch_raises(self, factor_env):
        fp, _ = factor_env
        fm = kr.fit_factor_model(fp.assets, fp.factors)
        with pytest.raises(ValueError):
            kr.factor_risk_decomposition(np.array([0.5, 0.5]), fm)
