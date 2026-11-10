#!/usr/bin/env python
"""Train or fine-tune one learned model (called by 01_prepare; standalone use for inspection).

Thin entry point: runs fwm.train as a script from the repository root, so that the relative paths
(data/, models/, results/) resolve as in the paper. Arguments are passed through:
    python scripts/03_train.py --help
"""
import os, runpy, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
runpy.run_module("fwm.train", run_name="__main__", alter_sys=True)
