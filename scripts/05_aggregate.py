#!/usr/bin/env python
"""Stage 3: aggregate every result file into tables (md, tex), figures and summary.json.

Thin entry point: runs fwm.aggregate as a script from the repository root, so that the relative paths
(data/, models/, results/) resolve as in the paper. Arguments are passed through:
    python scripts/05_aggregate.py --help
"""
import os, runpy, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
runpy.run_module("fwm.aggregate", run_name="__main__", alter_sys=True)
