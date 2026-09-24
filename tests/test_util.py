"""_util 内部工具测试：分布函数（含无 scipy 回退）精度与输入归一化行为。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from kairos_risk import _util


# --------------------------------------------------------------------------- #
# 标准正态 / 学生 t / 贝塔函数
# --------------------------------------------------------------------------- #
class TestDistributions:
    def test_norm_cdf_known_values(self):
        assert abs(_util.norm_cdf(0.0) - 0.5) < 1e-15
        assert abs(_util.norm_cdf(1.96) - 0.975) < 1e-4
        x = np.array([-2.0, -0.5, 0.5, 2.0])
        out = _util.norm_cdf(x)
        assert isinstance(out, np.ndarray)
        assert np.allclose(out, [0.02275, 0.30854, 0.69146, 0.97725], atol=1e-4)
        assert isinstance(_util.norm_cdf(0.0), float)   # 标量进标量出

    def test_norm_pdf(self):
        assert abs(_util.norm_pdf(0.0) - 1.0 / math.sqrt(2.0 * math.pi)) < 1e-15
        # 密度在 0 处最大、对称
        assert _util.norm_pdf(0.0) > _util.norm_pdf(1.0)
        assert abs(_util.norm_pdf(1.0) - _util.norm_pdf(-1.0)) < 1e-15

    def test_norm_ppf_roundtrip(self):
        for p in (1e-4, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0 - 1e-4):
            z = _util.norm_ppf(p)
            assert abs(_util.norm_cdf(z) - p) < 1e-10
        with pytest.raises(ValueError):
            _util.norm_ppf(0.0)
        with pytest.raises(ValueError):
            _util.norm_ppf(1.0)

    def test_betainc_properties(self):
        assert abs(_util.betainc(2.5, 2.5, 0.5) - 0.5) < 1e-12
        for a, b, x in ((2.0, 3.0, 0.4), (0.5, 5.0, 0.1), (7.0, 1.5, 0.8)):
            # 对称恒等式 I_x(a,b) + I_{1−x}(b,a) = 1
            assert abs(_util.betainc(a, b, x) + _util.betainc(b, a, 1.0 - x) - 1.0) < 1e-12
        assert _util.betainc(2.0, 3.0, 0.0) == 0.0
        assert _util.betainc(2.0, 3.0, 1.0) == 1.0
        with pytest.raises(ValueError):
            _util.betainc(0.0, 1.0, 0.5)
        with pytest.raises(ValueError):
            _util.betainc(1.0, -2.0, 0.5)

    def test_betaincinv_roundtrip(self):
        for a, b in ((2.0, 3.0), (30.0, 71.0), (0.5, 0.5)):
            for p in (0.05, 0.5, 0.95):
                x = _util.betaincinv(a, b, p)
                assert abs(_util.betainc(a, b, x) - p) < 1e-8
        assert _util.betaincinv(2.0, 3.0, 0.0) == 0.0
        assert _util.betaincinv(2.0, 3.0, 1.0) == 1.0
        with pytest.raises(ValueError):
            _util.betaincinv(2.0, 3.0, 1.5)

    def test_student_t_pdf_integrates_to_one(self):
        grid = np.linspace(-300.0, 300.0, 300001)
        pdf = _util.student_t_pdf(grid, 5.0)
        integral = float(np.sum(pdf) * (grid[1] - grid[0]))   # 黎曼和，避免 trapz 弃用
        assert abs(integral - 1.0) < 1e-4
        with pytest.raises(ValueError):
            _util.student_t_pdf(0.0, 0.0)

    def test_t_cdf_symmetry(self):
        assert abs(_util.t_cdf(0.0, 5.0) - 0.5) < 1e-15
        for df in (1.0, 3.0, 10.0):
            for x in (0.7, 2.5):
                assert abs(_util.t_cdf(x, df) + _util.t_cdf(-x, df) - 1.0) < 1e-12
        # t(1)（柯西）在 x=1 处 CDF = 0.75
        assert abs(_util.t_cdf(1.0, 1.0) - 0.75) < 1e-12
        with pytest.raises(ValueError):
            _util.t_cdf(1.0, 0.0)

    def test_t_ppf_roundtrip(self):
        for df in (3.0, 5.0, 20.0):
            for p in (0.01, 0.25, 0.75, 0.99):
                x = _util.t_ppf(p, df)
                assert abs(_util.t_cdf(x, df) - p) < 1e-8
        assert _util.t_ppf(0.5, 7.0) == 0.0
        with pytest.raises(ValueError):
            _util.t_ppf(0.0, 5.0)
        with pytest.raises(ValueError):
            _util.t_ppf(0.95, -1.0)

    def test_no_scipy_fallback_matches_scipy(self, monkeypatch):
        """屏蔽 scipy 后，回退实现与 scipy 参考值一致（二分/连分式精度）。"""
        sstats = pytest.importorskip("scipy.stats")
        ref_ppf = {p: float(sstats.norm.ppf(p)) for p in (0.001, 0.05, 0.5, 0.95, 0.999)}
        ref_t = {(5.0, 0.025): float(sstats.t.ppf(0.025, 5)),
                 (10.0, 0.975): float(sstats.t.ppf(0.975, 10))}
        monkeypatch.setattr(_util, "_try_import_scipy", lambda: None)
        for p, v in ref_ppf.items():
            assert abs(_util.norm_ppf(p) - v) < 1e-9
        for (df, p), v in ref_t.items():
            assert abs(_util.t_ppf(p, df) - v) < 1e-6
        assert abs(_util.norm_cdf(1.0) - float(sstats.norm.cdf(1.0))) < 1e-12
        assert abs(_util.betaincinv(30.0, 71.0, 0.025)
                   - float(sstats.beta.ppf(0.025, 30.0, 71.0))) < 1e-8


# --------------------------------------------------------------------------- #
# 输入归一化
# --------------------------------------------------------------------------- #
class TestNormalization:
    def test_clean_1d(self):
        s = pd.Series([0.1, np.nan, 0.2, np.inf])
        out = _util.clean_1d(s)
        assert np.allclose(out, [0.1, 0.2])
        with pytest.raises(ValueError):
            _util.clean_1d(pd.Series([np.nan, np.nan]))
        with pytest.raises(ValueError):
            _util.clean_1d(pd.DataFrame({"a": [1.0], "b": [2.0]}))  # 多列
        single = pd.DataFrame({"a": [1.0, 2.0]})
        assert np.allclose(_util.clean_1d(single), [1.0, 2.0])     # 单列可

    def test_as_returns_frame(self):
        s = pd.Series([1.0, 2.0], name="x")
        assert list(_util.as_returns_frame(s).columns) == ["x"]
        assert list(_util.as_returns_frame(s.rename(None)).columns) == ["asset"]
        assert list(_util.as_returns_frame(np.array([1.0, 2.0])).columns) == ["asset"]
        f3 = _util.as_returns_frame(pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, 3.0]}))
        assert len(f3) == 1                       # 整行剔除 NaN
        with pytest.raises(ValueError):
            _util.as_returns_frame(np.zeros((2, 2, 2)))
        with pytest.raises(ValueError):
            _util.as_returns_frame(pd.DataFrame({"a": [np.nan]}))

    def test_align_pair_index_join(self):
        idx_a = pd.RangeIndex(10)
        idx_b = pd.RangeIndex(5, 15)
        a = pd.Series(np.arange(10.0), index=idx_a)
        b = pd.Series(np.arange(10.0) * 2, index=idx_b)
        va, vb, idx = _util.align_pair(a, b)
        assert len(va) == len(vb) == 5
        assert idx is not None and len(idx) == 5
        assert np.allclose(va, [5.0, 6.0, 7.0, 8.0, 9.0])
        assert np.allclose(vb, (va - 5.0) * 2.0)

    def test_align_pair_positional(self):
        va, vb, idx = _util.align_pair(np.zeros(5), np.ones(5))
        assert idx is None and len(va) == 5
        with pytest.raises(ValueError):
            _util.align_pair(np.zeros(5), np.zeros(6))
        with pytest.raises(ValueError):
            _util.align_pair(pd.Series([np.nan], index=[1]),
                             pd.Series([1.0], index=[2]))   # 无共同索引

    def test_as_weights_alignment(self):
        cols = pd.Index(["A", "B", "C"])
        w = pd.Series([0.3, 0.3, 0.4], index=["C", "A", "B"])
        arr, idx = _util.as_weights(w, cols)
        assert np.allclose(arr, [0.3, 0.4, 0.3])          # 按 A,B,C 重排
        assert idx is cols
        arr2, idx2 = _util.as_weights(w)                  # 无 columns -> 原序
        assert np.allclose(arr2, [0.3, 0.3, 0.4])
        assert idx2 is w.index

    def test_as_weights_validation(self):
        cols = pd.Index(["A", "B", "C"])
        with pytest.raises(ValueError):
            _util.as_weights(pd.Series([0.5, 0.5], index=["A", "B"]), cols)
        with pytest.raises(ValueError):
            _util.as_weights(np.array([0.5, 0.5]), cols)
        with pytest.raises(ValueError):
            _util.as_weights(np.array([0.5, np.nan]))
        with pytest.raises(ValueError):
            _util.as_weights(np.array([]))

    def test_wrap_1d(self):
        idx = pd.Index(["A", "B"])
        s = _util.wrap_1d(np.array([1.0, 2.0]), idx, name="v")
        assert isinstance(s, pd.Series) and s.name == "v"
        a = _util.wrap_1d(np.array([1.0, 2.0]))
        assert isinstance(a, np.ndarray)

    def test_as_mapping(self):
        assert _util.as_mapping(None) == {}
        assert _util.as_mapping({"A": 1, "B": np.nan}) == {"A": 1.0}
        assert _util.as_mapping(pd.Series([1.0, 2.0], index=["A", "B"])) == \
            {"A": 1.0, "B": 2.0}
        with pytest.raises(TypeError):
            _util.as_mapping([1.0, 2.0])

    def test_tail_mask(self):
        v = np.arange(100.0)
        m = _util.tail_mask(v, 0.9)
        assert int(m.sum()) == 10
        assert v[m].max() == 9.0
        # 全并列时保底含最差观测
        flat = np.zeros(10)
        assert _util.tail_mask(flat, 0.95).any()

    def test_describe_seq(self):
        d = _util.describe_seq([1.0, 2.0, 3.0, 4.0])
        assert d["count"] == 4.0 and d["min"] == 1.0 and d["max"] == 4.0
        assert d["median"] == 2.5 and d["mean"] == 2.5
        empty = _util.describe_seq([])
        assert empty["count"] == 0.0 and math.isnan(empty["median"])

    def test_check_helpers(self):
        assert _util.check_confidence(0.95) == 0.95
        with pytest.raises(ValueError):
            _util.check_confidence(1.5)
        assert _util.check_positive_periods(252) == 252.0
        with pytest.raises(ValueError):
            _util.check_positive_periods(0)

    def test_quantile_sorted_matches_numpy(self):
        v = np.array([3.0, 1.0, 2.0])
        assert _util.quantile_sorted(v, 0.5) == float(np.quantile(v, 0.5))
