#!/usr/bin/env python
"""Stage 1: collect the data of every condition and train every learned model for one seed.

Thin entry point: runs fwm.prepare as a script from the repository root, so that the relative paths
(data/, models/, results/) resolve as in the paper. Arguments are passed through:
    python scripts/01_prepare.py --help
"""
import os, runpy, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
runpy.run_module("fwm.prepare", run_name="__main__", alter_sys=True)
