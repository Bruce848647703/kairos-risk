"""测试共用夹具：固定 seed 的合成数据与「无 scipy」回退环境（离线、可复现）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import kairos_risk as kr
from kairos_risk import _util
from kairos_risk import tail as _tail


@pytest.fixture()
def panel4() -> pd.DataFrame:
    """4 资产 600 期合成收益面板（等相关 ρ=0.35 + 漂移，默认波动 1.0%~1.8%）。"""
    return kr.make_multi_asset_panel(n_assets=4, n_periods=600, seed=3)


@pytest.fixture()
def cov4(panel4) -> pd.DataFrame:
    """panel4 的样本协方差。"""
    return kr.covariance_matrix(panel4)


@pytest.fixture()
def w4(panel4) -> pd.Series:
    """固定的 4 资产组合权重（和为 1，递减）。"""
    return pd.Series([0.4, 0.3, 0.2, 0.1], index=panel4.columns)


@pytest.fixture()
def t_rets() -> pd.Series:
    """学生 t（ν=5）肥尾收益 4000 期，scale=0.01。"""
    return kr.make_student_t_returns(n=4000, seed=5, df=5.0, scale=0.01)


@pytest.fixture()
def norm_rets() -> pd.Series:
    """大样本正态收益 N(0, 0.01²)，用于分布数值校验（50000 期）。"""
    return pd.Series(np.random.default_rng(1).standard_normal(50000) * 0.01,
                     name="normal")


@pytest.fixture()
def pareto_rets() -> pd.Series:
    """GPD/帕累托型极端损失收益（真值 ξ=0.3、scale=0.01、基础损失 0.02）。"""
    return kr.make_pareto_losses(n=20000, seed=11, xi=0.3, scale=0.01, threshold=0.02)


@pytest.fixture()
def factor_env():
    """已知暴露 B 的因子面板（无肥尾、4000 期）：返回 (FactorPanel, 真值 B)。"""
    b = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8], [-0.5, 0.5]])
    fp = kr.make_factor_panel(n_assets=4, n_factors=2, n_periods=4000, seed=7,
                              tail=False, exposures=b, idio_std=0.004,
                              factor_std=0.010, alpha=0.0002)
    return fp, b


@pytest.fixture()
def no_scipy(monkeypatch):
    """模拟「无 scipy」环境：把库内两处 scipy 单一入口都替换为返回 None。

    全库只有 ``_util`` 与 ``tail`` 两个模块直接持有 ``_try_import_scipy`` 绑定，
    同时 monkeypatch 两处即可让所有分布函数 / GPD 拟合走纯 numpy 回退路径。
    """
    monkeypatch.setattr(_util, "_try_import_scipy", lambda: None)
    monkeypatch.setattr(_tail, "_try_import_scipy", lambda: None)
