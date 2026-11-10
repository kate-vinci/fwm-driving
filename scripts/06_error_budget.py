#!/usr/bin/env python
"""Complement: composition-error budget (true quantities substituted into the coupler).

Thin entry point: runs fwm.error_budget as a script from the repository root, so that the relative paths
(data/, models/, results/) resolve as in the paper. Arguments are passed through:
    python scripts/06_error_budget.py --help
"""
import os, runpy, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
runpy.run_module("fwm.error_budget", run_name="__main__", alter_sys=True)
