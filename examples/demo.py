"""Kairos Risk 演示：合成多资产收益 -> VaR/ES/成分风险 -> 因子风险分解 -> 回撤分析
-> 压力测试 -> EVT 尾部 -> PSR/DSR 防过拟合。

运行： python examples/demo.py
所有数据均为离线合成（固定 seed），可复现；未安装 scipy 时自动走纯 numpy / math 回退路径。
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import kairos_risk as kr

try:  # scipy 仅用于增强（GPD 极大似然、特殊函数）；缺失时全部功能自动回退
    import scipy  # noqa: F401

    HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    HAVE_SCIPY = False


def main() -> None:
    ann = math.sqrt(252.0)
    if not HAVE_SCIPY:
        print("[提示] 未检测到 scipy：分布分位数与 GPD 拟合走纯 numpy 回退路径（结果同量级）。")

    print("=" * 70)
    print("① 合成 5 资产 · 2 因子收益面板（n=750，固定 seed=7，学生 t 肥尾）")
    fp = kr.make_factor_panel(n_assets=5, n_factors=2, n_periods=750, seed=7)
    rets = fp.assets
    w = pd.Series([0.30, 0.25, 0.20, 0.15, 0.10], index=rets.columns)
    port = kr.portfolio_returns(rets, w)
    rep = kr.risk_report(rets)
    print(rep.loc[["annualized_return", "volatility", "excess_kurtosis",
                   "historical_var", "es_historical"]].round(4).to_string())
    print(f"组合每期波动={kr.volatility(port):.5f}  年化={kr.volatility(port) * ann:.2%}")

    print("=" * 70)
    print("② 组合 VaR / ES 与欧拉分解（成分之和 = 组合总风险）")
    cov = kr.covariance_matrix(rets)
    rd = kr.component_risk(w, cov)
    print(f"σp={rd.volatility:.5f}  Σ成分风险={float(rd.component.sum()):.5f}  "
          f"分散化比率={rd.diversification_ratio:.3f}  风险有效N={rd.effective_n:.2f}")
    for c in (0.95, 0.99):
        cv = kr.component_var(w, rets, c)
        ce = kr.component_es(w, rets, c)
        print(f"  c={c:.0%}: VaR={cv.value:.4f} (Σ成分={float(cv.component.sum()):.4f})   "
              f"ES={ce.value:.4f} (Σ成分={float(ce.component.sum()):.4f})")
    ce95 = kr.component_es(w, rets, 0.95)
    print("  成分 ES 占比@95%:", {str(k): round(float(v), 3) for k, v in ce95.pct.items()})

    print("=" * 70)
    print("③ 因子风险分解（2 因子 + 特质；最小二乘还原已知暴露）")
    fm = kr.fit_factor_model(rets, fp.factors)
    brk = kr.factor_risk_decomposition(w, fm)
    print("估计暴露（与生成过程真值的最大偏差 "
          f"{float(np.abs(fm.exposures - fp.exposures).max().max()):.4f}）:")
    print(fm.exposures.round(3).to_string())
    print(f"因子方差占比={brk.factor_share:.1%}  特质方差占比={1.0 - brk.factor_share:.1%}")
    print(f"可加性校验: 因子+特质={brk.factor_variance + brk.idio_variance:.4e} "
          f"== 总方差={brk.total_variance:.4e}")

    print("=" * 70)
    print("④ 回撤分析（组合收益序列）")
    s = kr.drawdown_report(port, periods_per_year=252).summary()
    print(f"最大回撤={s['max_drawdown']:.2%}  当前回撤={s['current_drawdown']:.2%}  "
          f"最长水下期={int(s['longest_drawdown_periods'])}  "
          f"水下时间占比={s['time_underwater_ratio']:.1%}  Calmar={s['calmar_ratio']:.2f}")
    top3 = kr.top_drawdowns(port, n=3)
    print("Top3 深度回撤区间:")
    print(kr.episodes_frame(top3)[["trough", "depth", "down_periods",
                                   "recovery_periods", "recovered"]].to_string())

    print("=" * 70)
    print("⑤ 压力测试：假设情景（市值 1 亿） + 历史重放 + 反向压力")
    betas = {c: b for c, b in zip(rets.columns, [1.2, 1.0, 0.8, 0.6, 0.4])}
    scs = [
        kr.Scenario.market(betas, shock=-0.20, vol_multiplier=1.8),
        kr.Scenario.idiosyncratic({str(rets.columns[0]): -0.35}, name="single_blowup"),
        kr.Scenario.vol_spike(2.0),
    ]
    print(kr.scenario_table(w, scs, value=1e8, cov=cov).round(4).to_string())
    crash = kr.make_crash_panel(n_assets=5, n_periods=750, seed=17)
    replay = kr.historical_replay(crash, pd.Series(0.2, index=crash.columns),
                                  n_worst=3, window=1, value=1e8)
    print("历史重放最差 3 窗口:", [str(pd.Timestamp(d).date()) for d in replay.hit_dates],
          f" 最差损益={replay.worst_pnl:,.0f} 元")
    rs = kr.reverse_stress_test(w, scs[0], target_loss=2e7, value=1e8,
                                cov=cov, metric="var")
    print(f"反向压力（冲击后 95% VaR 口径）：亏 2000 万需冲击倍数 "
          f"λ={rs.multiplier:.3f}（收敛={rs.converged}）")

    print("=" * 70)
    print("⑥ EVT 尾部：Hill 尾指数 + POT/GPD 外推（学生 t(5) 亏损尾，理论 ξ=1/ν=0.2）")
    fat = kr.make_student_t_returns(n=6000, seed=5, df=5.0, scale=0.01)
    hill = kr.hill_estimator(fat, k=150)
    fit = kr.fit_gpd(fat, tail_quantile=0.95)
    print(f"Hill ξ={hill.xi:.3f}±{hill.se:.3f} (k={hill.k})   "
          f"GPD[{fit.method}]: ξ={fit.xi:.3f}, β={fit.scale:.5f}, "
          f"u={fit.threshold:.4f}, ζ={fit.exceedance_prob:.3f}")
    for c in (0.99, 0.999):
        print(f"  POT VaR@{c:.1%}={kr.pot_var(fit, c):.4f}   "
              f"POT ES@{c:.1%}={kr.pot_es(fit, c):.4f}   "
              f"历史 VaR@{c:.1%}={kr.historical_var(fat, c):.4f}")

    print("=" * 70)
    print("⑦ 回测防过拟合：PSR / 缩水夏普 DSR")
    strat = port + 0.0004                       # 模拟叠加正漂移后的策略收益
    sr = kr.per_period_sharpe(strat)
    print(f"每期夏普={sr:.4f}  PSR={kr.psr_from_returns(strat):.4f}  "
          f"最小跟踪长度(95%)={kr.minimum_track_length(sr, 0.95):.0f} 期")
    for n in (1, 100, 1000):
        d = kr.dsr_from_returns(strat, n_trials=n)
        print(f"  N={n:>4d} 次试验: DSR={d.dsr:.4f}  门槛SR0={d.sr0:.4f}  "
              f"haircut={d.haircut:.4f}   （试验越多，门槛越高、DSR 越低）")
    budget = kr.trials_for_target_dsr(sr, len(strat), target_dsr=0.95)
    print(f"维持 DSR≥95% 的试验预算: N ≤ {budget}")
    print()
    print("演示完成：以上全部数值离线可复现（固定 seed），未联网、未依赖外部数据。")


if __name__ == "__main__":
    main()
