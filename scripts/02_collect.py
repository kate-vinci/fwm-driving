#!/usr/bin/env python
"""Collect transitions for one condition (called by 01_prepare; standalone use for inspection).

Thin entry point: runs fwm.collect as a script from the repository root, so that the relative paths
(data/, models/, results/) resolve as in the paper. Arguments are passed through:
    python scripts/02_collect.py --help
"""
import os, runpy, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
runpy.run_module("fwm.collect", run_name="__main__", alter_sys=True)
