"""Kairos Risk —— 自研轻量的 Python 风险分析库。

覆盖风险分析的完整链路：

- measures:        基础风险度量（历史/参数/修正 VaR、ES-CVaR、下行风险、
                   跟踪误差、信息比率、Beta/Jensen-Alpha/R²、高阶矩、相关矩阵）；
- decomposition:   组合风险分解（边际/成分风险贡献、成分 VaR 与成分 ES 的欧拉分配、
                   分散化比率、风险集中度）；
- factor:          因子风险模型（lstsq 估计暴露/因子协方差/特质风险，
                   组合的因子风险贡献与特质风险分解）；
- drawdown:        回撤分析（水下曲线、最大回撤、回撤区间的下跌期/恢复期、Calmar）；
- stress:          情景与压力测试（假设情景损益、历史重放、反向压力测试）；
- tail:            极值理论（Hill 尾部指数、POT/GPD 尾部 VaR-ES、尾部比率）；
- backtest_risk:   回测防过拟合指标（PSR、缩水夏普 DSR、最小跟踪长度、
                   损失概率、胜率置信区间）；
- sim:             合成收益/情景生成器（肥尾、因子结构，固定 seed 可复现）。

设计原则：必需依赖仅 numpy/pandas（scipy 为**可选增强**，缺失时全部功能自动回退到
自研的纯 numpy / math 实现）；输出「同型进出」（DataFrame/Series 进 -> Series 出，
ndarray 进 -> ndarray 出）；损失类度量一律返回正数。
"""
from .backtest_risk import (
    DeflatedSharpe,
    WinRateCI,
    backtest_risk_report,
    deflated_sharpe_ratio,
    dsr_from_returns,
    expected_max_sharpe,
    loss_probability,
    minimum_track_length,
    per_period_sharpe,
    probabilistic_sharpe_ratio,
    probability_of_min_loss,
    psr_from_returns,
    sharpe_std_error,
    trials_for_target_dsr,
    win_rate,
    win_rate_ci,
)
from .decomposition import (
    RiskDecomposition,
    TailDecomposition,
    component_es,
    component_risk,
    component_var,
    diversification_ratio,
    marginal_risk,
    portfolio_returns,
    risk_effective_n,
    worst_tail_sample,
)
from .drawdown import (
    DrawdownEpisode,
    DrawdownReport,
    calmar_ratio,
    current_drawdown,
    drawdown_episodes,
    drawdown_report,
    drawdown_series,
    duration_summary,
    episodes_frame,
    longest_drawdown,
    max_drawdown,
    running_peak,
    time_underwater_ratio,
    top_drawdowns,
    underwater,
    wealth_curve,
)
from .factor import (
    FactorModel,
    FactorRiskBreakdown,
    exposure_summary,
    factor_risk_decomposition,
    factor_var_contributions,
    factor_variance,
    fit_factor_model,
    idiosyncratic_variance,
    market_model,
    portfolio_exposure,
)
from .measures import (
    RegressionStats,
    annualized_return,
    beta,
    cornish_fisher_z,
    correlation_matrix,
    covariance_matrix,
    cvar,
    downside_deviation,
    expected_shortfall,
    historical_var,
    information_ratio,
    jensen_alpha,
    kurtosis,
    modified_var,
    parametric_var,
    r_squared,
    regression_stats,
    regression_table,
    risk_report,
    sharpe_ratio,
    skewness,
    sortino_ratio,
    tracking_error,
    volatility,
)
from .sim import (
    FactorPanel,
    make_crash_panel,
    make_drawdown_path,
    make_factor_panel,
    make_mixture_returns,
    make_multi_asset_panel,
    make_pareto_losses,
    make_student_t_returns,
)
from .stress import (
    ReplayResult,
    ReverseStressResult,
    Scenario,
    StressResult,
    apply_scenario,
    historical_replay,
    reverse_stress_test,
    scenario_table,
    worst_windows,
)
from . import sim  # noqa: E402  允许 kr.sim.* 与 kr.make_* 两种访问方式
from .tail import (
    GpdFit,
    HillResult,
    exceedances,
    fit_gpd,
    gpd_mle,
    gpd_moments,
    gpd_pwm,
    hill_estimator,
    hill_path,
    pot_es,
    pot_var,
    tail_ratio,
    tail_summary,
)

__version__ = "0.1.0"

__all__ = [
    # measures
    "volatility", "skewness", "kurtosis", "correlation_matrix", "covariance_matrix",
    "annualized_return", "historical_var", "parametric_var", "modified_var",
    "cornish_fisher_z", "expected_shortfall", "cvar", "downside_deviation",
    "sortino_ratio", "sharpe_ratio", "tracking_error", "information_ratio",
    "beta", "jensen_alpha", "r_squared", "regression_stats", "regression_table",
    "RegressionStats", "risk_report",
    # decomposition
    "component_risk", "marginal_risk", "diversification_ratio", "risk_effective_n",
    "component_es", "component_var", "portfolio_returns", "worst_tail_sample",
    "RiskDecomposition", "TailDecomposition",
    # factor
    "fit_factor_model", "market_model", "factor_risk_decomposition", "factor_variance",
    "idiosyncratic_variance", "portfolio_exposure", "factor_var_contributions",
    "exposure_summary", "FactorModel", "FactorRiskBreakdown",
    # drawdown
    "wealth_curve", "running_peak", "drawdown_series", "underwater", "max_drawdown",
    "current_drawdown", "longest_drawdown", "time_underwater_ratio", "drawdown_episodes",
    "top_drawdowns", "episodes_frame", "duration_summary", "calmar_ratio",
    "drawdown_report", "DrawdownEpisode", "DrawdownReport",
    # stress
    "Scenario", "apply_scenario", "scenario_table", "historical_replay", "worst_windows",
    "reverse_stress_test", "StressResult", "ReplayResult", "ReverseStressResult",
    # tail
    "hill_estimator", "hill_path", "exceedances", "fit_gpd", "gpd_pwm", "gpd_moments",
    "gpd_mle", "pot_var", "pot_es", "tail_ratio", "tail_summary", "HillResult", "GpdFit",
    # backtest_risk
    "per_period_sharpe", "sharpe_std_error", "probabilistic_sharpe_ratio",
    "psr_from_returns", "minimum_track_length", "expected_max_sharpe",
    "deflated_sharpe_ratio", "dsr_from_returns", "trials_for_target_dsr",
    "loss_probability", "probability_of_min_loss", "win_rate", "win_rate_ci",
    "backtest_risk_report", "DeflatedSharpe", "WinRateCI",
    # sim
    "sim", "make_factor_panel", "make_multi_asset_panel", "make_student_t_returns",
    "make_mixture_returns", "make_pareto_losses", "make_crash_panel",
    "make_drawdown_path", "FactorPanel",
    # meta
    "__version__",
]
