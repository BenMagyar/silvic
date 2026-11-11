#!/usr/bin/env -S uv run --script
# /// script
# requires-python = "==3.12.*"
# dependencies = ["lightgbm==4.7.0", "numpy"]
# ///
"""Generates cases_v4.json and v4_<case>.txt: gen_models.py's regression_nan_cat, lambdarank and single_leaf cases
trained and predicted by LightGBM 4.x, whose files silvic parses (they are not written back byte for byte: silvic
writes `version=v3`). Same rows and JSON format as gen_models.py.

Usage, with the repo's LightGBM 4.7 environment:
  .venv/bin/python crates/silvic-core/tests/fixtures/lightgbm/gen_models_v4.py
or with uv:
  ./gen_models_v4.py
"""

import os
import sys

import lightgbm as lgb

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_models  # noqa: E402

CASES = ["regression_nan_cat", "lambdarank", "single_leaf"]


def main():
    if not lgb.__version__.startswith("4."):
        sys.exit(f"needs lightgbm 4.x, found {lgb.__version__}")
    gen_models.write_fixtures(CASES, "v4_", "cases_v4.json", "gen_models_v4.py")


if __name__ == "__main__":
    main()
