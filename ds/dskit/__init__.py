"""dskit: the shared data-science helpers behind every notebook in this portfolio.

One import gives each project the same audit checks, EDA charts, evaluation (with bootstrap confidence
intervals), explainability and model-selection utilities, so the notebooks read as one body of work and
the method is tested once (tests/test_dskit.py) instead of re-implemented per project.

    from dskit import set_style, audit, eda, evaluate, explain, select, report
"""
from . import audit, eda, evaluate, explain, report, select  # noqa: F401
from .style import CYCLE, PALETTE, fmt_money, fmt_pct, headline, savefig, set_style  # noqa: F401

__version__ = "1.0.0"
