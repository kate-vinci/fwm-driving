#!/usr/bin/env python
"""Final step: regenerate every number of the paper (LaTeX macros) from the result files.

Thin entry point: runs fwm.make_numbers as a script from the repository root, so that the relative paths
(data/, models/, results/) resolve as in the paper. Arguments are passed through:
    python scripts/08_make_numbers.py --help
"""
import os, runpy, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
runpy.run_module("fwm.make_numbers", run_name="__main__", alter_sys=True)
