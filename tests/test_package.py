"""包级测试：版本号、__all__ 导出完整性、子模块访问与依赖铁律。"""
from __future__ import annotations

import importlib
import pathlib

import kairos_risk as kr

SUBMODULES = ("measures", "decomposition", "factor", "drawdown",
              "stress", "tail", "backtest_risk", "sim")


def test_version():
    assert kr.__version__ == "0.1.0"


def test_all_exports_resolvable():
    assert len(kr.__all__) > 50
    for name in kr.__all__:
        assert hasattr(kr, name), f"__all__ 中的 {name} 无法从包命名空间解析"


def test_submodules_importable():
    for mod in SUBMODULES:
        m = importlib.import_module(f"kairos_risk.{mod}")
        assert m is not None
        # 每个子模块自己的 __all__（若有）都应可从子模块解析
        for name in getattr(m, "__all__", ()):
            assert hasattr(m, name), f"{mod}.{name} 缺失"


def test_dual_access_to_sim():
    assert kr.sim.make_drawdown_path is kr.make_drawdown_path


def test_no_forbidden_hard_dependencies():
    """铁律：源码禁止依赖 sklearn / statsmodels；scipy 只允许出现在可选入口。"""
    root = pathlib.Path(kr.__file__).parent
    for py in sorted(root.glob("*.py")):
        text = py.read_text(encoding="utf-8")
        assert "sklearn" not in text.replace("不依赖 scipy / sklearn", "")
        assert "statsmodels" not in text
        if py.name not in ("_util.py", "tail.py"):
            # 除单一入口与 GPD-MLE 外，任何模块不得直接 import scipy
            assert "import scipy" not in text and "from scipy" not in text


def test_public_api_smoke():
    """顶层主 API 至少可调用一次（端到端最小链路）。"""
    import pandas as pd

    panel = kr.make_multi_asset_panel(n_assets=3, n_periods=200, seed=1)
    w = pd.Series(1.0 / 3.0, index=panel.columns)
    assert kr.component_risk(w, kr.covariance_matrix(panel)).volatility > 0
    assert kr.max_drawdown(kr.portfolio_returns(panel, w)) >= 0.0
