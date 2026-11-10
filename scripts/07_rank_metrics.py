#!/usr/bin/env python
"""Complement: ranking agreement with the oracle from the per-decision logs.

Thin entry point: runs fwm.rank_metrics as a script from the repository root, so that the relative paths
(data/, models/, results/) resolve as in the paper. Arguments are passed through:
    python scripts/07_rank_metrics.py --help
"""
import os, runpy, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
runpy.run_module("fwm.rank_metrics", run_name="__main__", alter_sys=True)
