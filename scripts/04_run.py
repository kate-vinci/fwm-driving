#!/usr/bin/env python
"""Stage 2: evaluate one system on one test (T1-T6), one seed, one variant; writes results/<test>/<system>_<seed><tag>.jsonl.

Thin entry point: runs fwm.run as a script from the repository root, so that the relative paths
(data/, models/, results/) resolve as in the paper. Arguments are passed through:
    python scripts/04_run.py --help
"""
import os, runpy, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
runpy.run_module("fwm.run", run_name="__main__", alter_sys=True)
