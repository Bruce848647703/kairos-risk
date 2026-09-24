# Kairos Risk

> Kairos 量化系列的风险分析模块 —— 一个**自研、轻量**的 Python 风险度量、分解与回测防过拟合库。

`kairos_risk` 覆盖「度量 → 分解 → 归因 → 压测 → 尾部 → 显著性」的完整风险链路：
VaR/ES（历史/参数/修正）、欧拉成分风险分解、因子风险模型、回撤区间分析、
情景/历史/反向压力测试、EVT 极值尾部（Hill + POT/GPD）、PSR/缩水夏普 DSR。
核心代码全部原创，必需依赖仅 `numpy` 与 `pandas`；`scipy` 为**可选增强**，
缺失时全部功能自动回退到自研的纯 numpy / `math.erf` / 不完全贝塔实现。

## 特性
- **基础度量 `measures`**：波动/偏度/超额峰度、历史 VaR、参数 VaR（正态与学生 t，
  自研分位数无需 scipy）、Cornish-Fisher 修正 VaR、ES/CVaR（历史/正态/学生 t 闭式）、
  下行标准差、Sortino/Sharpe、跟踪误差、信息比率、Beta/Jensen-Alpha/R²、一键 `risk_report`。
- **风险分解 `decomposition`**：边际/成分风险贡献（MRC/CRC，欧拉精确可加 Σ CRC=σp）、
  成分 VaR 与成分 ES 的尾部欧拉分配、分散化比率、风险赫芬达尔与有效资产数。
- **因子模型 `factor`**：`numpy.linalg.lstsq` 估计暴露/α/因子协方差/特质风险（不用
  sklearn/statsmodels），组合的「逐因子 + 逐资产特质」双口径精确可加分解。
- **回撤分析 `drawdown`**：水下曲线、最大/当前回撤、回撤区间（episode）切分——
  下跌期/恢复期/是否收复、Top-N 回撤、持续期分布、Calmar 与一键体检报告。
- **压力测试 `stress`**：假设情景（市场 beta 传导 / 因子冲击 / 个股暴跌 / 波动放大，可组合）、
  历史重放（最差 N 窗口，可非重叠）、**反向压力测试**（二分搜索达成目标损失的冲击倍数）。
- **EVT 尾部 `tail`**：Hill 尾指数（含选 k 路径图）、POT/GPD 拟合（scipy MLE /
  自研 PWM / 矩估计三路）、McNeil–Frey 尾部 VaR-ES 外推、尾部比率、一页纸摘要。
- **防过拟合 `backtest_risk`**：PSR（含偏度/峰度修正）、缩水夏普 DSR（试验次数越多
  门槛越高）、最小跟踪长度、E[max SR]、损失概率、胜率置信区间（Wilson/Wald/Clopper-Pearson 精确）。
- **合成器 `sim`**：学生 t / 混合正态 / 帕累托损失 / 多资产面板 / 同步暴跌面板 /
  已知暴露的因子面板（附**解析真值协方差**，便于白盒校验），全部固定 seed 可复现。
- **同型进出**：DataFrame/Series 输入返回带资产索引的 Series，纯 ndarray 输入返回
  ndarray；权重与协方差成对传入时按资产名对齐，杜绝顺序错位。
- **可测试**：230 个测试全部离线、确定性，`pytest -q` 全绿；含「无 scipy」回退路径的
  monkeypatch 专项测试。

## 安装
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .              # 或 pip install numpy pandas
pip install -e ".[dev]"       # 需要跑测试时
pip install -e ".[scipy]"     # 可选：GPD 极大似然与更快的特殊函数
```

## 快速开始
### ① 风险度量
```python
import kairos_risk as kr

rets = ...  # 收益面板 DataFrame(index=日期, columns=资产) 或单资产 Series
print(kr.historical_var(rets, 0.95))                 # 历史 VaR（正数损失）
print(kr.expected_shortfall(rets, 0.99, method="student_t"))
print(kr.risk_report(rets).round(4))                 # 一键体检表
```

### ② 成分风险与因子分解
```python
w = ...                                              # 组合权重 Series
cov = kr.covariance_matrix(rets)
rd = kr.component_risk(w, cov)
print(rd.component.sum(), rd.volatility)             # 欧拉可加：两者相等
print(rd.diversification_ratio, rd.effective_n)      # 分散化比率 / 风险有效N

fm = kr.fit_factor_model(rets, factor_rets)          # lstsq 因子模型
brk = kr.factor_risk_decomposition(w, fm)
print(brk.factor_share)                              # 因子风险占比
```

### ③ 回撤与压力测试
```python
rep = kr.drawdown_report(port_rets)
print(rep.summary())                                 # MDD/当前回撤/Calmar/水下占比

sc = kr.Scenario.market(betas, shock=-0.20, vol_multiplier=1.8)
res = kr.apply_scenario(w, sc, value=1e8, cov=cov)
print(res.pnl, res.var_stressed)
rs = kr.reverse_stress_test(w, sc, target_loss=2e7, value=1e8, cov=cov, metric="var")
print(rs.multiplier)                                 # 打穿目标所需冲击倍数
```

### ④ EVT 尾部与防过拟合
```python
hill = kr.hill_estimator(rets_1d, k=150)             # 尾指数 ξ（>0 肥尾）
fit = kr.fit_gpd(rets_1d, tail_quantile=0.95)        # scipy MLE，缺失自动 PWM
print(kr.pot_var(fit, 0.999), kr.pot_es(fit, 0.999)) # 尾部外推

print(kr.psr_from_returns(strat_rets))               # 概率夏普
dsr = kr.dsr_from_returns(strat_rets, n_trials=50)   # 缩水夏普（50 次试验）
print(dsr.dsr, dsr.haircut)
```

完整可运行示例见 [`examples/demo.py`](examples/demo.py)（合成多资产收益 →
VaR/ES/成分风险 → 因子分解 → 回撤 → 压测 → EVT → PSR/DSR，一键跑通）。

## API 概览
| 模块 | 关键对象 | 说明 |
|---|---|---|
| `measures` | `historical_var` `parametric_var` `modified_var` `expected_shortfall`/`cvar` `volatility` `downside_deviation` `sortino_ratio` `sharpe_ratio` `tracking_error` `information_ratio` `beta` `jensen_alpha` `r_squared` `regression_stats` `risk_report` | 基础风险度量 |
| `decomposition` | `component_risk` `marginal_risk` `diversification_ratio` `risk_effective_n` `component_var` `component_es` `portfolio_returns` `worst_tail_sample` `RiskDecomposition` `TailDecomposition` | 欧拉风险分解 |
| `factor` | `fit_factor_model` `market_model` `factor_risk_decomposition` `portfolio_exposure` `factor_var_contributions` `exposure_summary` `FactorModel` `FactorRiskBreakdown` | 因子风险模型 |
| `drawdown` | `wealth_curve` `running_peak` `drawdown_series` `underwater` `max_drawdown` `current_drawdown` `longest_drawdown` `time_underwater_ratio` `drawdown_episodes` `top_drawdowns` `duration_summary` `calmar_ratio` `drawdown_report` | 回撤分析 |
| `stress` | `Scenario`（`market`/`idiosyncratic`/`vol_spike`/`from_history`） `apply_scenario` `scenario_table` `historical_replay` `worst_windows` `reverse_stress_test` | 情景与压力测试 |
| `tail` | `hill_estimator` `hill_path` `exceedances` `fit_gpd` `gpd_pwm` `gpd_moments` `gpd_mle` `pot_var` `pot_es` `tail_ratio` `tail_summary` `HillResult` `GpdFit` | EVT 极值尾部 |
| `backtest_risk` | `per_period_sharpe` `sharpe_std_error` `probabilistic_sharpe_ratio` `psr_from_returns` `minimum_track_length` `expected_max_sharpe` `deflated_sharpe_ratio` `dsr_from_returns` `trials_for_target_dsr` `loss_probability` `probability_of_min_loss` `win_rate` `win_rate_ci` `backtest_risk_report` | 回测防过拟合 |
| `sim` | `make_student_t_returns` `make_mixture_returns` `make_pareto_losses` `make_multi_asset_panel` `make_crash_panel` `make_factor_panel` `make_drawdown_path` `FactorPanel` | 合成数据生成器 |

## 设计要点
- **数据约定**：「收益面板」指 DataFrame(index=期次, columns=资产)；单资产接受
  Series/一维 ndarray。损失类度量（VaR/ES/回撤/压力损失）一律返回**正数**；
  情景损益 `pnl` 为带符号金额。
- **欧拉可加性**：σp 与 ES 都是权重的一次齐次函数，成分贡献之和**精确等于**
  组合总风险（测试中以恒等式断言）；成分 VaR 采用「尾部均值 + VaR/ES 比例缩放」
  的工程可加近似（严格的 VaR 边际需要条件密度，样本上不可直接估计）。
- **scipy 单一入口**：全库只经 `_util._try_import_scipy` 取用 scipy；缺失（或测试
  monkeypatch）时正态/学生 t 的 CDF-PPF 走 `math.erf` 二分与自研 Lentz 连分式
  不完全贝塔，GPD 拟合走自研 PWM（概率加权矩），Clopper-Pearson 区间走二分反演
  ——两条路径结果互相印证（测试对照到 1e-9/1e-6 量级）。
- **防未来函数与确定性**：历史重放只用「当期已实现收益」；所有随机数据由
  `numpy.random.default_rng(seed)` 生成，同 seed 必得同结果；回撤区间扫描为
  单遍 O(n) 状态机。
- **PSR/DSR 口径**：夏普一律用**每期**（非年化）口径进入公式，方差修正含偏度与
  非超额峰度项（正态 γ₄=3）；DSR 门槛 SR₀ = max(E[max SR(N,V)], SR*)，
  因此 DSR 随试验次数 N 单调不增——这正是它防过拟合的机制。
- **EVT 口径**：Hill 与 POT 默认研究**亏损尾**（−r 的右尾）；POT VaR/ES 为
  McNeil–Frey 外推，置信度落入拟合区间之外（尾概率 ≥ ζ）时显式报错而非静默外推。

## 测试
```bash
make test          # 或 python -m pytest -q
```

## 项目结构
```
kairos_risk/    核心包（measures / decomposition / factor / drawdown /
                stress / tail / backtest_risk / sim / _util）
examples/       可运行示例
tests/          pytest 测试
```

## 许可
MIT © 2026 Bruce848647703，见 [LICENSE](LICENSE)。

## 参考与致谢
本项目为**独立原创实现**，未复制任何第三方代码。设计思路受业界通用风险范式
（VaR/ES 风险度量、欧拉成分风险分解、多因子风险模型、回撤与 Calmar、情景/历史/
反向压力测试、Hill 与 POT/GPD 极值理论、PSR/DSR 回测防过拟合）启发，
在此向开源量化社区致谢。算法与接口均为本仓库自研。
