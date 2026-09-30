# Kairos Risk

[![CI](https://github.com/Bruce848647703/kairos-risk/actions/workflows/ci.yml/badge.svg)](https://github.com/Bruce848647703/kairos-risk/actions/workflows/ci.yml)

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
- **真实数据接入 `realdata`**：本地 CSV 日线目录 → 对齐收盘价/收益面板；处理前复权负价、
  停牌缺口（前填、无未来函数）、重复/乱序日期，并用**交易所涨跌停硬约束**反解「等差（减法）
  前复权」的复权偏移**下界**做口径修复（`D̂ ≤ D_真`，只把伪收益拉回真实口径、不会把风险修小），
  残余越界按 clip/zero/drop 处理，全过程留痕、离线不联网。
- **同型进出**：DataFrame/Series 输入返回带资产索引的 Series，纯 ndarray 输入返回
  ndarray；权重与协方差成对传入时按资产名对齐，杜绝顺序错位。
- **可测试**：267 个测试全部离线、确定性，`pytest -q` 全绿；含「无 scipy」回退路径的
  monkeypatch 专项测试，以及真实数据接入（tmp_path 造 CSV）的白盒测试。

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

## 真实数据风险报告
[`examples/real_risk_report.py`](examples/real_risk_report.py) 在**真实 A 股行情**上跑完整条
风险链路，产出可审计的报告与指标文件（离线、只读、约 2 秒跑完）：

```bash
python examples/real_risk_report.py \
    --data-dir ../kairos-data/data/ashare        # 省略则按「兄弟仓库」相对定位
# 产物：research/real_risk/{REPORT.md, risk_metrics.json, drawdown.csv}
```

流程：`realdata.load_close_panel` 读 38 只标的的前复权日线 → `repair_close_panel` 修复复权口径
→ `simple_returns` + `sanitize_returns` 得到清洁收益面板 → 逆波动权重组合 → 度量 / 成分分解 /
因子 / 回撤 / 压测 / EVT / PSR-DSR → 25 项一致性校验（含欧拉可加性、ES≥VaR、DSR 单调性）。

真实数字（样本 2018-11-29 ~ 2026-09-23，1898 个交易日 × 35 只标的，市值 1 亿元，
`--scheme inverse_vol`；完整表格见 [`research/real_risk/REPORT.md`](research/real_risk/REPORT.md)）：

| 维度 | 关键结果 |
|---|---|
| 收益/波动 | 年化收益 12.11%，年化波动 20.47%，夏普 0.59，超额峰度 4.21（肥尾） |
| VaR（单日） | 95%：历史 1.83% / 正态 2.07% / 学生 t 1.98% / CF 修正 1.96% |
| VaR（单日） | 99%：历史 3.46% / 正态 2.95% / 学生 t 3.28% / **CF 修正 4.23%** |
| ES/CVaR | 95% 历史 2.86%（286 万元）；99% 历史 4.79%（479 万元） |
| 成分风险 | σp=0.012892，Σ CRC 误差 1.7e-18（欧拉精确可加）；DR=1.744（零相关上界 √35=5.92）；风险有效 N=33.85 |
| 因子风险 | 组合 beta 0.948，**系统性方差占比 94.50%**、特质 5.50%，个股 R² 中位 0.320 |
| 回撤 | **MDD 34.22%**（2021-02-10 峰 → 2022-10-31 谷，至样本末未收复），最长水下 1361 个交易日，水下时间占比 94.89%，Calmar 0.35 |
| 压力测试 | 市场 −20% + 波动×1.8 → 亏 1896.8 万元；历史最差单日重放 −784.7 万元（2020-02-03，= 99% VaR 的 2.3 倍）；最差 21 日窗口 −1663.1 万元 |
| 反向压力 | 亏掉 10% 市值：pnl 口径需市场跌 10.5%（λ=0.527），冲击后 VaR 口径只需 λ=0.381 |
| EVT 尾部 | **Hill ξ=0.4505±0.0327**（k=190，α≈2.2）；GPD(MLE) ξ=0.1983、β=0.00830、ζ=0.0501；POT VaR 99.9%=6.74%（674 万元，= 正态法 1.7 倍）、POT ES 99.9%=8.99% |
| 防过拟合 | 每期夏普 0.0417（年化 0.66），**PSR=0.9649**；DSR：N=1 → 0.9649，N=35 → 0.3739，**N=100 → 0.2372** |

可调参数：`--scheme {inverse_vol,equal}`、`--start/--end`、`--value`、`--confidence/--confidence99/
--confidence999`、`--market-shock/--vol-multiplier/--idio-shock`、`--reverse-target-pct`、
`--replay-window`、`--hill-k/--hill-tail-quantile`、`--tail-quantile`、`--n-trials`、
`--tol/--max-compression/--max-bad-days/--on-bad`、`--out`、`--no-write`。

### 数据声明（真实数据口径与已知偏差）
- 数据为同系列 [`kairos-data`](../kairos-data) 仓库已提交的 **38 只 A 股前复权日线 CSV**
  （`date,open,high,low,close,volume`；来源：腾讯/新浪公开行情接口）。本仓库**只读、不联网、
  不改写、不再分发**原始数据；数据版权归原作者/数据源所有，仅用于研究、学习与演示，
  不用于任何商业用途，**不保证准确、完整或及时**。
- 该数据集的前复权为**等差（减法）口径** `P_adj = P_raw − D_t`（`D_t` = t 日之后的累计分红），
  直接 `pct_change` 会把收益放大 `k = P_raw/P_adj` 倍：修复前有 155 个交易日的收益**超出交易所
  涨跌停幅度**（物理上不可能），高分红标的的年化波动甚至被放大到 8000%。
  `realdata.repair_close_panel` 用涨跌停硬约束反解出偏移**下界** `D̂ ≤ D_真` 并令
  `P_repaired = P_adj + D̂`，因此**修复只会把伪收益拉回真实口径，不会把风险修小**（保守）。
- 复权价被压到近零、舍入误差已主导收益的 3 只标的（`sh600809`/`sh601088`/`sh601899`，
  压缩倍数 k̂ = 817/132/4.3 > 闸门 3.0）被整只剔除；残余 71 个越界观测按涨跌停边界 clip。
  因 `D̂` 是下界，高分红标的**早期（2019-2021）的波动/VaR 仍可能偏大（偏保守）**。
- 组合为**样本内静态权重**（不含调仓与交易成本），市场因子用同篮子等权组合代理；
  因此所有数字是「若一直持有该静态组合」的风险口径，**不构成任何投资建议**。
- 全部口径、剔除清单与 25 项校验结果都写入 `REPORT.md` 第 0/8 节与 `risk_metrics.json`，
  可离线复现：`python examples/real_risk_report.py --data-dir <kairos-data>/data/ashare`。


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
| `realdata` | `load_close_panel` `simple_returns` `log_returns` `repair_close_panel` `estimate_dividend_offset` `sanitize_returns` `daily_price_limit` `panel_summary` `default_data_dir` `has_local_data` `symbols_of` | 真实 CSV 行情接入与复权口径修复 |

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
                stress / tail / backtest_risk / sim / realdata / _util）
examples/       可运行示例（demo.py 合成数据；real_risk_report.py 真实数据）
tests/          pytest 测试（全部离线、确定性）
research/       真实数据研究产物（real_risk/REPORT.md + risk_metrics.json + drawdown.csv）
```

## 许可
MIT © 2026 Bruce848647703，见 [LICENSE](LICENSE)。

## 参考与致谢
本项目为**独立原创实现**，未复制任何第三方代码。设计思路受业界通用风险范式
（VaR/ES 风险度量、欧拉成分风险分解、多因子风险模型、回撤与 Calmar、情景/历史/
反向压力测试、Hill 与 POT/GPD 极值理论、PSR/DSR 回测防过拟合）启发，
在此向开源量化社区致谢。算法与接口均为本仓库自研。
