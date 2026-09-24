"""tail 模块测试：Hill 尾指数还原、POT/GPD 单调性与外推、无 scipy 时的纯 numpy 回退。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import kairos_risk as kr
from kairos_risk import _util
from kairos_risk import tail as _tail


def _exact_gpd_sample(n: int = 100000, xi: float = 0.2, scale: float = 0.02,
                      seed: int = 0) -> np.ndarray:
    """逆变换精确抽样的 GPD(ξ, β) 样本（超出量口径，全为正）。"""
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    return scale * ((1.0 - u) ** (-xi) - 1.0) / xi


# --------------------------------------------------------------------------- #
# Hill 估计
# --------------------------------------------------------------------------- #
class TestHill:
    def test_recovers_pareto_xi(self, pareto_rets):
        h = kr.hill_estimator(pareto_rets, k=400)     # 真值 ξ=0.3
        assert abs(h.xi - 0.3) < 0.1
        assert h.k == 400
        assert h.n_obs == len(pareto_rets)
        assert abs(h.se - abs(h.xi) / math.sqrt(400)) < 1e-15
        assert h.tail_index == pytest.approx(1.0 / h.xi)
        assert h.threshold > 0.02                      # 基础损失 0.02 之上的分位

    def test_default_k_from_tail_quantile(self, pareto_rets):
        h = kr.hill_estimator(pareto_rets, tail_quantile=0.9)
        assert h.k == int(round(len(pareto_rets) * 0.1))

    def test_normal_tail_thinner_than_pareto(self, norm_rets, pareto_rets):
        hn = kr.hill_estimator(norm_rets, k=300)       # 正态：ξ≈0（指数型尾）
        hp = kr.hill_estimator(pareto_rets, k=300)
        assert hn.xi < 0.2 < hp.xi

    def test_gain_side(self):
        gains = -kr.make_pareto_losses(n=10000, seed=3, xi=0.3, scale=0.01,
                                       threshold=0.02)
        h = kr.hill_estimator(gains, k=300, side="gain")
        assert abs(h.xi - 0.3) < 0.12

    def test_hill_path(self, pareto_rets):
        hp = kr.hill_path(pareto_rets, k_min=100, k_max=800, n_points=10)
        assert list(hp.columns) == ["threshold", "xi", "se"]
        assert len(hp) >= 5
        assert (hp.index.to_numpy() >= 100).all()
        assert (hp.index.to_numpy() <= 800).all()

    def test_tail_index_infinite_for_short_tail(self):
        h = kr.HillResult(xi=-0.1, se=0.01, k=100, threshold=1.0,
                          n_exceed=100, n_obs=1000)
        assert h.tail_index == float("inf")

    def test_validation(self, pareto_rets):
        with pytest.raises(ValueError):
            kr.hill_estimator(pareto_rets.iloc[:5])            # 样本 < 8
        with pytest.raises(ValueError):
            kr.hill_estimator(pareto_rets, k=1)                # k < 2
        with pytest.raises(ValueError):
            kr.hill_estimator(pareto_rets, k=len(pareto_rets))  # k > n−2
        with pytest.raises(ValueError):
            kr.hill_estimator(pareto_rets, side="both")
        with pytest.raises(ValueError):
            # 收益全为正 -> 亏损尾全为负 -> 阈值非正
            kr.hill_estimator(-pareto_rets, k=100, side="loss")


# --------------------------------------------------------------------------- #
# GPD 拟合（PWM / 矩 / MLE）
# --------------------------------------------------------------------------- #
class TestGpdFit:
    def test_pwm_recovers_exact_sample(self):
        y = _exact_gpd_sample(xi=0.2, scale=0.02)
        xi, beta = kr.gpd_pwm(y)
        assert abs(xi - 0.2) < 0.01
        assert abs(beta - 0.02) / 0.02 < 0.05

    def test_moments_recovers_exact_sample(self):
        y = _exact_gpd_sample(xi=0.2, scale=0.02)
        xi, beta = kr.gpd_moments(y)
        assert abs(xi - 0.2) < 0.02
        assert abs(beta - 0.02) / 0.02 < 0.05

    def test_pwm_and_moments_validation(self):
        with pytest.raises(ValueError):
            kr.gpd_pwm(np.array([0.1]))                 # 样本不足
        with pytest.raises(ValueError):
            kr.gpd_pwm(np.array([0.1, -0.2, 0.3]))      # 非全正
        with pytest.raises(ValueError):
            kr.gpd_moments(np.array([0.1, -0.2]))

    def test_mle_vs_pwm_close(self, pareto_rets):
        pytest.importorskip("scipy")     # 本用例显式走 scipy MLE 路径
        f_mle = kr.fit_gpd(pareto_rets, tail_quantile=0.95, method="mle")
        f_pwm = kr.fit_gpd(pareto_rets, tail_quantile=0.95, method="pwm")
        assert f_mle.method == "scipy_mle" and f_pwm.method == "pwm"
        assert abs(f_mle.xi - f_pwm.xi) < 0.05
        assert abs(f_mle.scale - f_pwm.scale) / f_mle.scale < 0.10

    def test_fit_metadata(self, pareto_rets):
        fit = kr.fit_gpd(pareto_rets, tail_quantile=0.95)
        assert abs(fit.xi - 0.3) < 0.12
        assert fit.n_exceed == pytest.approx(int(0.05 * len(pareto_rets)), abs=10)
        assert abs(fit.exceedance_prob - fit.n_exceed / fit.n_obs) < 1e-12
        assert fit.threshold > 0.02
        assert fit.side == "loss"

    def test_exceedances(self, pareto_rets):
        y, u = kr.exceedances(pareto_rets, tail_quantile=0.95)
        assert (y > 0).all()
        assert u == pytest.approx(float(np.quantile(-pareto_rets.to_numpy(), 0.95)))
        assert len(y) >= 5
        with pytest.raises(ValueError):
            kr.exceedances(pareto_rets, threshold=10.0)   # 无超出量

    def test_fit_from_raw_exceedances(self, pareto_rets):
        y, u = kr.exceedances(pareto_rets, tail_quantile=0.95)
        fit = kr.fit_gpd(exceedances_=y, threshold=u, method="pwm")
        assert fit.exceedance_prob == 1.0
        with pytest.raises(ValueError):
            kr.fit_gpd(exceedances_=y)                    # 缺 threshold
        with pytest.raises(ValueError):
            kr.fit_gpd()                                  # 什么都没给
        with pytest.raises(ValueError):
            kr.fit_gpd(pareto_rets, method="bayes")

    def test_survival_at_pot_var_equals_tail_prob(self, pareto_rets):
        fit = kr.fit_gpd(pareto_rets, tail_quantile=0.95)
        for c in (0.97, 0.99):
            q = kr.pot_var(fit, c)
            assert fit.survival(q) == pytest.approx(1.0 - c, rel=1e-8, abs=1e-12)
        assert math.isnan(float(fit.survival(fit.threshold - 1.0)))  # 阈值外无定义

    def test_mean_excess_formula(self, pareto_rets):
        fit = kr.fit_gpd(pareto_rets, tail_quantile=0.95)
        assert fit.mean_excess() == pytest.approx(fit.scale / (1.0 - fit.xi))
        lvl = fit.threshold + fit.scale
        assert fit.mean_excess(lvl) == pytest.approx(
            (fit.scale + fit.xi * fit.scale) / (1.0 - fit.xi))
        inf_fit = kr.GpdFit(xi=1.2, scale=0.01, threshold=0.02, n_exceed=100,
                            n_obs=1000, exceedance_prob=0.1, method="pwm")
        assert inf_fit.mean_excess() == float("inf")


# --------------------------------------------------------------------------- #
# POT VaR / ES：单调性与外推
# --------------------------------------------------------------------------- #
class TestPot:
    def test_monotone_in_confidence(self, pareto_rets):
        fit = kr.fit_gpd(pareto_rets, tail_quantile=0.95)
        q97, q99, q999 = (kr.pot_var(fit, c) for c in (0.97, 0.99, 0.999))
        assert q97 < q99 < q999
        assert kr.pot_es(fit, 0.99) >= q99
        assert kr.pot_es(fit, 0.999) >= q999

    def test_monotone_in_threshold(self, pareto_rets):
        # 更低阈值 -> 更多超出量、更大 ζ；同置信度 VaR 仍应稳定在同一量级
        fit_hi = kr.fit_gpd(pareto_rets, tail_quantile=0.98)
        fit_lo = kr.fit_gpd(pareto_rets, tail_quantile=0.90)
        assert fit_hi.threshold > fit_lo.threshold
        q_hi = kr.pot_var(fit_hi, 0.995)
        q_lo = kr.pot_var(fit_lo, 0.995)
        assert abs(q_hi - q_lo) / q_hi < 0.35

    def test_extrapolation_sane_range(self, pareto_rets):
        fit = kr.fit_gpd(pareto_rets, tail_quantile=0.95)
        q999 = kr.pot_var(fit, 0.999)
        # 真值：P(L>x)=0.001 -> x = 0.02 + 0.01·((0.001)^(-0.3)−1)/0.3 ≈ 0.251
        assert 0.12 < q999 < 0.45

    def test_out_of_range_confidence_raises(self, pareto_rets):
        fit = kr.fit_gpd(pareto_rets, tail_quantile=0.95)
        with pytest.raises(ValueError):
            kr.pot_var(fit, 0.90)      # 尾概率 0.10 > ζ=0.05，落在拟合区间外

    def test_xi_ge_one_es_infinite(self):
        fit = kr.GpdFit(xi=1.1, scale=0.01, threshold=0.02, n_exceed=100,
                        n_obs=1000, exceedance_prob=0.1, method="pwm")
        assert kr.pot_es(fit, 0.99) == float("inf")

    def test_tail_ratio(self, norm_rets, pareto_rets):
        tr_n = kr.tail_ratio(norm_rets, 0.95)
        assert abs(tr_n - 1.2537) < 0.02          # 正态理论值 ≈ φ(z)/α/z
        tr_p = kr.tail_ratio(pareto_rets, 0.95)
        assert tr_p > tr_n                        # 肥尾 -> 越过 VaR 后恶化更多

    def test_tail_summary(self, pareto_rets):
        ts = kr.tail_summary(pareto_rets)
        needed = {"empirical_var", "empirical_es", "empirical_tail_ratio",
                  "hill_xi", "hill_se", "hill_k", "gpd_xi", "gpd_scale",
                  "gpd_threshold", "gpd_exceedance_prob", "pot_var", "pot_es"}
        assert needed <= set(ts.index)
        assert ts.attrs["method"] in ("scipy_mle", "pwm", "moments")
        assert float(ts.loc["pot_es", "value"]) >= float(ts.loc["pot_var", "value"])
        assert float(ts.loc["empirical_es", "value"]) >= float(ts.loc["empirical_var", "value"])


# --------------------------------------------------------------------------- #
# scipy 缺失时的回退路径
# --------------------------------------------------------------------------- #
class TestNoScipyFallback:
    def test_auto_falls_back_to_pwm(self, pareto_rets, no_scipy):
        fit = kr.fit_gpd(pareto_rets, tail_quantile=0.95)
        assert fit.method == "pwm"
        assert kr.pot_var(fit, 0.99) > 0
        assert kr.pot_es(fit, 0.99) > kr.pot_var(fit, 0.99)

    def test_mle_raises_import_error(self, pareto_rets, no_scipy):
        with pytest.raises(ImportError):
            kr.gpd_mle(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
        with pytest.raises(ImportError):
            kr.fit_gpd(pareto_rets, method="mle")

    def test_fallback_close_to_scipy_path(self, pareto_rets, monkeypatch):
        pytest.importorskip("scipy")     # 先取 scipy MLE 参考值，再切到回退路径对比
        fit_sp = kr.fit_gpd(pareto_rets, tail_quantile=0.95, method="mle")
        q_sp, e_sp = kr.pot_var(fit_sp, 0.99), kr.pot_es(fit_sp, 0.99)
        monkeypatch.setattr(_util, "_try_import_scipy", lambda: None)
        monkeypatch.setattr(_tail, "_try_import_scipy", lambda: None)
        fit_fb = kr.fit_gpd(pareto_rets, tail_quantile=0.95)
        assert fit_fb.method == "pwm"
        q_fb, e_fb = kr.pot_var(fit_fb, 0.99), kr.pot_es(fit_fb, 0.99)
        assert abs(q_fb - q_sp) / q_sp < 0.05
        assert abs(e_fb - e_sp) / e_sp < 0.05

    def test_distribution_fallbacks(self, t_rets, no_scipy):
        # 学生 t 参数 VaR / PSR / Clopper-Pearson 全走纯 numpy 回退
        v = kr.parametric_var(t_rets, 0.95, distribution="student_t", df=5.0)
        assert v > 0
        z = kr.parametric_var(mean=0.0, std=0.01, confidence=0.95)
        assert abs(z - 1.6448536269514722 * 0.01) < 1e-9   # 二分回退精度
        assert 0.0 <= kr.psr_from_returns(t_rets) <= 1.0
        ci = kr.win_rate_ci(t_rets, method="clopper_pearson")
        assert 0.0 <= ci.lower <= ci.point <= ci.upper <= 1.0

    def test_hill_and_summary_need_no_scipy(self, pareto_rets, no_scipy):
        h = kr.hill_estimator(pareto_rets, k=400)
        assert abs(h.xi - 0.3) < 0.1
        ts = kr.tail_summary(pareto_rets, method="moments")
        assert ts.attrs["method"] == "moments"
