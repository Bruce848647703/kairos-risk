"""sim 模块测试：固定 seed 可复现、肥尾性质、真值协方差代数与确定性路径。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import kairos_risk as kr


class TestReproducibility:
    def test_same_seed_same_output(self):
        pd.testing.assert_frame_equal(kr.make_multi_asset_panel(seed=3),
                                      kr.make_multi_asset_panel(seed=3))
        pd.testing.assert_series_equal(kr.make_student_t_returns(seed=5),
                                       kr.make_student_t_returns(seed=5))
        pd.testing.assert_series_equal(kr.make_drawdown_path(), kr.make_drawdown_path())

    def test_different_seed_differs(self):
        a = kr.make_multi_asset_panel(seed=1)
        b = kr.make_multi_asset_panel(seed=2)
        assert not np.allclose(a.to_numpy(), b.to_numpy())

    def test_business_day_index(self):
        p = kr.make_multi_asset_panel(n_periods=100, start="2020-01-01")
        assert isinstance(p.index, pd.DatetimeIndex)
        assert len(p.index) == 100
        assert (p.index.dayofweek < 5).all()


class TestFatTails:
    def test_student_t_moments(self):
        r = kr.make_student_t_returns(n=40000, seed=5, df=5.0, scale=0.01, loc=0.0002)
        assert kr.kurtosis(r) > 1.0                    # 超额峰度显著为正（峰度>3）
        assert abs(float(r.std(ddof=1)) - 0.01) < 0.001  # 方差归一校准
        assert abs(float(r.mean()) - 0.0002) < 3e-4

    def test_student_t_df_validation(self):
        with pytest.raises(ValueError):
            kr.make_student_t_returns(n=100, df=2.0)   # 方差不存在

    def test_mixture_skew_and_kurt(self):
        r = kr.make_mixture_returns(n=20000, seed=13)
        assert kr.skewness(r) < -0.5                   # 危机态 -> 负偏
        assert kr.kurtosis(r) > 3.0                    # 显著肥尾
        with pytest.raises(ValueError):
            kr.make_mixture_returns(crisis_prob=1.5)

    def test_pareto_losses(self):
        r = kr.make_pareto_losses(n=5000, seed=11, xi=0.3, threshold=0.02)
        assert (r.to_numpy() < -0.02).all()            # 收益口径 = 负的损失
        h = kr.hill_estimator(r, k=250)
        assert 0.15 < h.xi < 0.45                      # 还原 ξ≈0.3
        with pytest.raises(ValueError):
            kr.make_pareto_losses(xi=0.0)


class TestPanels:
    def test_shapes_and_default_names(self):
        p = kr.make_multi_asset_panel(n_assets=5, n_periods=100, seed=1)
        assert p.shape == (100, 5)
        assert list(p.columns) == ["ASSET_A", "ASSET_B", "ASSET_C",
                                   "ASSET_D", "ASSET_E"]
        p2 = kr.make_multi_asset_panel(n_assets=8, n_periods=60, seed=1)
        assert list(p2.columns) == [f"ASSET_{i}" for i in range(8)]
        p3 = kr.make_multi_asset_panel(n_assets=3, n_periods=60, seed=1,
                                       names=["X", "Y", "Z"])
        assert list(p3.columns) == ["X", "Y", "Z"]
        with pytest.raises(ValueError):
            kr.make_multi_asset_panel(n_assets=2, n_periods=60, names=["X"])

    def test_kinds_fat_tail_ordering(self):
        pn = kr.make_multi_asset_panel(n_assets=4, n_periods=4000, seed=3, kind="normal")
        pt = kr.make_multi_asset_panel(n_assets=4, n_periods=4000, seed=3,
                                       kind="student_t", df=4)
        assert float(kr.kurtosis(pt).mean()) > float(kr.kurtosis(pn).mean()) + 1.0
        with pytest.raises(ValueError):
            kr.make_multi_asset_panel(kind="laplace")

    def test_correlation_structure(self):
        p = kr.make_multi_asset_panel(n_assets=4, n_periods=3000, seed=3, rho=0.35)
        c = kr.correlation_matrix(p).to_numpy()
        off = c[~np.eye(4, dtype=bool)]
        assert np.abs(off - 0.35).max() < 0.05
        assert np.allclose(np.diag(c), 1.0)

    def test_custom_vols(self):
        vols = np.array([0.008, 0.012, 0.016, 0.020])
        p = kr.make_multi_asset_panel(n_assets=4, n_periods=4000, seed=9, vols=vols)
        sv = kr.volatility(p).to_numpy()
        assert np.abs(sv / vols - 1.0).max() < 0.10

    def test_corr_validation(self):
        not_psd = np.array([[1.0, 1.2], [1.2, 1.0]])
        with pytest.raises(ValueError):
            kr.make_multi_asset_panel(n_assets=2, n_periods=50, corr=not_psd)
        non_unit_diag = np.array([[1.0, 0.3], [0.3, 2.0]])
        with pytest.raises(ValueError):
            kr.make_multi_asset_panel(n_assets=2, n_periods=50, corr=non_unit_diag)
        with pytest.raises(ValueError):
            kr.make_multi_asset_panel(n_assets=3, n_periods=50, rho=1.5)
        with pytest.raises(ValueError):
            kr.make_multi_asset_panel(n_assets=2, n_periods=50, vols=[0.01, -0.02])

    def test_mixture_crisis_days(self):
        p = kr.make_multi_asset_panel(n_assets=4, n_periods=3000, seed=3,
                                      kind="mixture", crisis_prob=0.06,
                                      crisis_scale=3.5, crisis_shift=-0.02)
        worst_day_sum = float(p.sum(axis=1).min())
        assert worst_day_sum < -0.1        # 存在同步暴跌日


class TestCrashPanel:
    def test_crash_rows_match_betas(self):
        cp = kr.make_crash_panel(n_assets=4, n_periods=500, seed=17,
                                 crash_positions=(120, 260), crash_magnitude=-0.18)
        betas_true = np.linspace(1.3, 0.4, 4)
        for pos in (120, 260):
            row = cp.iloc[pos].to_numpy()
            assert np.abs(row - betas_true * -0.18).max() < 0.02  # 噪声 σ=0.004

    def test_crash_positions_out_of_range(self):
        with pytest.raises(ValueError):
            kr.make_crash_panel(n_periods=100, crash_positions=(500,))

    def test_custom_betas(self):
        cp = kr.make_crash_panel(n_assets=3, n_periods=200, seed=5,
                                 crash_positions=(10,), crash_magnitude=-0.10,
                                 betas=[1.0, 2.0, 3.0])
        row = cp.iloc[10].to_numpy()
        assert np.abs(row - np.array([1.0, 2.0, 3.0]) * -0.10).max() < 0.02


class TestFactorPanel:
    def test_true_cov_algebra(self):
        fp = kr.make_factor_panel(n_assets=4, n_factors=2, n_periods=200, seed=7)
        b = fp.exposures.to_numpy()
        f = np.diag(fp.factor_std.to_numpy() ** 2)
        d = np.diag(fp.idio_std.to_numpy() ** 2)
        assert np.allclose(fp.true_cov.to_numpy(), b @ f @ b.T + d)

    def test_shapes_and_alignment(self):
        fp = kr.make_factor_panel(n_assets=6, n_factors=3, n_periods=300, seed=2)
        assert fp.assets.shape == (300, 6)
        assert fp.factors.shape == (300, 3)
        assert fp.exposures.shape == (6, 3)
        assert fp.n_obs == 300
        assert fp.assets.index.equals(fp.factors.index)
        assert (fp.alphas.to_numpy() == 0.0002).all()

    def test_custom_exposures_used(self):
        b = np.array([[1.5, -0.5], [0.0, 1.0]])
        fp = kr.make_factor_panel(n_assets=2, n_factors=2, n_periods=200,
                                  seed=3, exposures=b)
        assert np.allclose(fp.exposures.to_numpy(), b)
        with pytest.raises(ValueError):
            kr.make_factor_panel(n_assets=2, n_factors=2, n_periods=200,
                                 exposures=np.zeros((3, 2)))

    def test_vector_vol_inputs(self):
        fp = kr.make_factor_panel(n_assets=3, n_factors=2, n_periods=200, seed=4,
                                  idio_std=[0.004, 0.005, 0.006],
                                  factor_std=[0.008, 0.012])
        assert np.allclose(fp.idio_std.to_numpy(), [0.004, 0.005, 0.006])
        assert np.allclose(fp.factor_std.to_numpy(), [0.008, 0.012])

    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            kr.make_factor_panel(n_assets=2, n_factors=2, n_periods=3)  # n < k+3


class TestDrawdownPath:
    def test_deterministic_values(self):
        p = kr.make_drawdown_path(up_periods=2, down_periods=2, recover_periods=2,
                                  up=0.02, down=-0.03, recover=0.01)
        assert np.allclose(p.to_numpy(), [0.02, 0.02, -0.03, -0.03, 0.01, 0.01])
        assert len(p) == 6

    def test_range_index_option(self):
        p = kr.make_drawdown_path(up_periods=1, down_periods=1, recover_periods=0,
                                  freq="range")
        assert isinstance(p.index, pd.RangeIndex)

    def test_validation(self):
        with pytest.raises(ValueError):
            kr.make_drawdown_path(down_periods=0)
        with pytest.raises(ValueError):
            kr.make_drawdown_path(up_periods=-1)
        with pytest.raises(ValueError):
            kr.make_drawdown_path(down=-1.2)      # 收益 ≤ −100%
