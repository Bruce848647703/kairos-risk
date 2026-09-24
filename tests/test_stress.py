"""stress 模块测试：情景损益方向与量级、历史重放取到最差窗口、反向压力测试收敛。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import kairos_risk as kr


# --------------------------------------------------------------------------- #
# 情景构造与冲击合成
# --------------------------------------------------------------------------- #
class TestScenarios:
    def test_market_shock_pnl(self, w4, cov4):
        betas = {c: b for c, b in zip(w4.index, [1.2, 0.9, 0.6, 0.3])}
        sc = kr.Scenario.market(betas, shock=-0.20)
        res = kr.apply_scenario(w4, sc, value=1e6, cov=cov4)
        expected_pct = float(sum(w4[c] * betas[c] for c in w4.index)) * -0.20
        assert res.pnl_pct < 0                       # 多头 + 市场暴跌 -> 亏损
        assert abs(res.pnl_pct - expected_pct) < 1e-15
        assert abs(res.pnl - 1e6 * expected_pct) < 1e-6
        assert abs(float(res.contribution.sum()) - res.pnl_pct) < 1e-15
        assert abs(res.value_after - 1e6 * (1.0 + expected_pct)) < 1e-6
        assert res.var_stressed > res.var_baseline > 0

    def test_market_rally_is_profit(self, w4):
        sc = kr.Scenario.market({c: 1.0 for c in w4.index}, shock=+0.10, name="rally")
        res = kr.apply_scenario(w4, sc)
        assert res.pnl_pct > 0

    def test_idiosyncratic_only_hits_target(self, w4):
        target = w4.index[0]
        sc = kr.Scenario.idiosyncratic({target: -0.30}, name="blowup")
        res = kr.apply_scenario(w4, sc)
        assert abs(res.pnl_pct - float(w4[target]) * -0.30) < 1e-15
        assert float(res.contribution.drop(target).abs().sum()) == 0.0

    def test_vol_spike_zero_pnl_doubles_vol(self, w4, cov4):
        sc = kr.Scenario.vol_spike(2.0)
        res = kr.apply_scenario(w4, sc, value=1e6, cov=cov4)
        assert res.pnl_pct == 0.0
        assert abs(res.vol_stressed - 2.0 * res.vol_baseline) < 1e-15
        assert abs(res.var_stressed - 2.0 * res.var_baseline) < 1e-12

    def test_factor_shock_linear(self, panel4, w4):
        expo = pd.DataFrame([[1.0, 0.2], [0.8, 0.5], [0.3, 0.9], [0.5, -0.4]],
                            index=panel4.columns, columns=["F1", "F2"])
        sc = kr.Scenario(name="f", factor_shocks={"F1": -0.05, "F2": 0.02},
                         exposures=expo)
        res = kr.apply_scenario(w4, sc)
        expected = float(w4.to_numpy() @ (expo.to_numpy() @ np.array([-0.05, 0.02])))
        assert abs(res.pnl_pct - expected) < 1e-15

    def test_composition_is_additive(self, w4):
        target = w4.index[1]
        sc = kr.Scenario(name="combo", market_shock=-0.1,
                         betas={c: 1.0 for c in w4.index},
                         asset_shocks={target: -0.2})
        res = kr.apply_scenario(w4, sc)
        expected = -0.1 * float(w4.sum()) - 0.2 * float(w4[target])
        assert abs(res.pnl_pct - expected) < 1e-15

    def test_from_history_replays_row(self, panel4, w4):
        d0 = panel4.index[100]
        sc = kr.Scenario.from_history(panel4, d0)
        res = kr.apply_scenario(w4, sc)
        assert abs(res.pnl_pct - float(panel4.loc[d0] @ w4)) < 1e-15
        with pytest.raises(KeyError):
            kr.Scenario.from_history(panel4, "1900-01-01")

    def test_scale_semantics(self):
        sc = kr.Scenario(name="s", market_shock=-0.1, betas={"A": 1.0},
                         asset_shocks={"A": -0.05}, vol_multiplier=2.0)
        s2 = sc.scale(2.0)
        assert abs(s2.market_shock + 0.2) < 1e-15
        assert abs(s2.asset_shocks["A"] + 0.1) < 1e-15
        assert abs(s2.vol_multiplier - 3.0) < 1e-15   # 1 + 2·(2−1)
        s0 = sc.scale(0.0)
        assert s0.market_shock == 0.0 and s0.vol_multiplier == 1.0
        s_neg = sc.scale(-1.0)
        assert abs(s_neg.vol_multiplier - 2.0) < 1e-15  # 反向时波动放大依然为正
        assert s_neg.market_shock > 0

    def test_scenario_table(self, w4, cov4, panel4):
        scs = [kr.Scenario.market({c: 1.0 for c in w4.index}, -0.1),
               kr.Scenario.vol_spike(1.5),
               kr.Scenario.from_history(panel4, panel4.index[5])]
        tab = kr.scenario_table(w4, scs, value=1e6, cov=cov4)
        assert list(tab.index) == [s.name for s in scs]
        assert {"pnl_pct", "pnl", "value_after", "var_stressed"} <= set(tab.columns)
        with pytest.raises(ValueError):
            kr.scenario_table(w4, [])

    def test_validation(self, w4):
        with pytest.raises(ValueError):   # market_shock 缺 betas
            kr.apply_scenario(w4, kr.Scenario(name="bad", market_shock=-0.1))
        with pytest.raises(ValueError):   # 未知资产
            kr.apply_scenario(w4, kr.Scenario(name="bad", asset_shocks={"NOPE": -0.1}))
        with pytest.raises(ValueError):   # factor_shocks 缺 exposures
            kr.apply_scenario(w4, kr.Scenario(name="bad", factor_shocks={"F": -0.1}))
        with pytest.raises(ValueError):
            kr.Scenario.vol_spike(0.0)
        with pytest.raises(ValueError):
            kr.apply_scenario(w4, kr.Scenario(name="flat"), value=-1.0)


# --------------------------------------------------------------------------- #
# 历史重放
# --------------------------------------------------------------------------- #
class TestReplay:
    def test_crash_days_are_selected(self):
        crash = kr.make_crash_panel(n_assets=4, n_periods=500, seed=17)
        w = pd.Series([0.25] * 4, index=crash.columns)
        rep = kr.historical_replay(crash, w, n_worst=3, window=1, value=1e6)
        crash_dates = {crash.index[p] for p in (120, 260, 400)}
        assert set(rep.hit_dates) == crash_dates   # 三个合成暴跌日恰为最差窗口
        assert rep.worst_pnl < 0
        assert rep.frame.shape[0] == 3
        assert abs(rep.worst_pnl - float(rep.frame["pnl"].min())) < 1e-9
        assert abs(rep.total_pnl - float(rep.frame["pnl"].sum())) < 1e-6
        assert abs(rep.mean_pnl - float(rep.frame["pnl"].mean())) < 1e-6

    def test_window1_contributions_sum_to_portfolio(self):
        crash = kr.make_crash_panel(n_assets=4, n_periods=200, seed=17,
                                    crash_positions=(50,))
        w = pd.Series([0.4, 0.3, 0.2, 0.1], index=crash.columns)
        rep = kr.historical_replay(crash, w, n_worst=5, window=1)
        contrib_cols = [c for c in rep.frame.columns if c.startswith("contrib_")]
        rowsum = rep.frame[contrib_cols].sum(axis=1)
        assert np.allclose(rowsum.to_numpy(),
                           rep.frame["portfolio_return"].to_numpy(), atol=1e-15)

    def test_multi_window_compounding(self):
        panel = kr.make_multi_asset_panel(n_assets=3, n_periods=100, seed=5)
        w = pd.Series([1.0 / 3.0] * 3, index=panel.columns)
        rep = kr.historical_replay(panel, w, n_worst=4, window=5)
        assert rep.n_windows == 96          # 100 − 5 + 1
        assert rep.window == 5
        multi = (1.0 + panel).rolling(5).apply(np.prod, raw=True).dropna() - 1.0
        port = multi.to_numpy() @ w.to_numpy()
        # 最差窗口必须是全局最差（n_worst 覆盖了它）
        assert abs(float(rep.frame["portfolio_return"].min()) - port.min()) < 1e-15

    def test_non_overlapping_spacing(self):
        crash = kr.make_crash_panel(n_assets=4, n_periods=500, seed=17)
        w = pd.Series([0.25] * 4, index=crash.columns)
        rep = kr.historical_replay(crash, w, n_worst=5, window=5, non_overlapping=True)
        pos = [list(crash.index).index(d) for d in rep.hit_dates]
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                assert abs(pos[i] - pos[j]) >= 5

    def test_worst_windows_helper(self):
        crash = kr.make_crash_panel(n_assets=4, n_periods=200, seed=17,
                                    crash_positions=(50,))
        w = pd.Series([0.25] * 4, index=crash.columns)
        fr = kr.worst_windows(crash, w, n_worst=2)
        assert isinstance(fr, pd.DataFrame) and len(fr) == 2

    def test_validation(self, panel4, w4):
        with pytest.raises(ValueError):
            kr.historical_replay(panel4, w4, n_worst=0)
        with pytest.raises(ValueError):
            kr.historical_replay(panel4, w4, window=0)
        with pytest.raises(ValueError):
            kr.historical_replay(panel4, w4, window=10 ** 4)  # 窗口超过样本
        with pytest.raises(ValueError):
            kr.historical_replay(panel4, w4, value=0.0)


# --------------------------------------------------------------------------- #
# 反向压力测试
# --------------------------------------------------------------------------- #
class TestReverseStress:
    def test_pnl_metric_analytic(self, w4, cov4):
        betas = {c: b for c, b in zip(w4.index, [1.2, 0.9, 0.6, 0.3])}
        sc = kr.Scenario.market(betas, shock=-0.20)
        base_loss_pct = -float(sum(w4[c] * betas[c] for c in w4.index)) * -0.20  # 0.18
        target = 5e4
        rs = kr.reverse_stress_test(w4, sc, target_loss=target, value=1e6,
                                    cov=cov4, metric="pnl")
        # 线性损益下 λ 有解析解：λ = target / (value · base_loss_pct)
        assert abs(rs.multiplier - target / (1e6 * base_loss_pct)) < 1e-6
        assert abs(rs.achieved_loss - target) < 1e-3
        assert rs.converged and rs.metric == "pnl" and rs.iterations >= 1
        assert len(rs.shocks) == len(w4)

    def test_direction_flip_for_rally(self, w4):
        rally = kr.Scenario.market({c: 1.0 for c in w4.index}, shock=+0.10, name="rally")
        rs = kr.reverse_stress_test(w4, rally, target_loss=0.05, value=1.0)
        assert rs.multiplier < 0                        # 需反转情景方向才亏损
        assert abs(rs.achieved_loss - 0.05) < 1e-8
        assert abs(rs.multiplier + 0.5) < 1e-6          # λ = −0.05/0.10

    def test_var_metric_monotone_in_target(self, w4, cov4):
        sc = kr.Scenario.market({c: 1.0 for c in w4.index}, shock=-0.20,
                                vol_multiplier=1.5)
        r1 = kr.reverse_stress_test(w4, sc, target_loss=3e4, value=1e6,
                                    cov=cov4, metric="var")
        r2 = kr.reverse_stress_test(w4, sc, target_loss=9e4, value=1e6,
                                    cov=cov4, metric="var")
        assert r1.converged and r2.converged
        assert r2.multiplier > r1.multiplier            # 目标越大所需冲击越强
        assert abs(r1.achieved_loss - 3e4) / 3e4 < 1e-6
        assert abs(r2.achieved_loss - 9e4) / 9e4 < 1e-6

    def test_unreachable_target_raises(self, w4):
        flat = kr.Scenario(name="flat")                 # 全零冲击永远不亏
        with pytest.raises(ValueError):
            kr.reverse_stress_test(w4, flat, target_loss=0.1, value=1.0,
                                   max_multiplier=8)

    def test_validation(self, w4):
        sc = kr.Scenario.market({c: 1.0 for c in w4.index}, -0.1)
        with pytest.raises(ValueError):
            kr.reverse_stress_test(w4, sc, target_loss=0.0)
        with pytest.raises(ValueError):
            kr.reverse_stress_test(w4, sc, target_loss=0.01, metric="bad")
        with pytest.raises(ValueError):
            kr.reverse_stress_test(w4, sc, target_loss=0.01, metric="var")  # 缺 cov
