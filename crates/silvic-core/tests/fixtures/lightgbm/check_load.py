"""Checks that LightGBM and lleaves read a silvic-written model as silvic does.

Input (written by `tests/model_lightgbm.rs`): a JSON file
  {"cases": [{"name", "model_file", "rows": [[value]], "raw": [value], "predicted": [value],
              "lleaves_skip": [row index]}]}
with floats as strings in Rust's shortest round-trip form ("NaN", "inf", "-inf"), "raw" and "predicted" being
`Model::predict` with and without `raw_score`. For each case the interpreter's LightGBM loads the model file and
must predict both exactly (same bits); with --lleaves, lleaves compiles it and must predict "predicted" within
1e-12 on every row but "lleaves_skip": rows holding a nonzero value of magnitude at most kZeroThreshold, which
LightGBM's dense-row reader turns into 0 and lleaves compares as is. Exits non-zero and names every mismatch.

Usage:
  .venv-lgb3/bin/python check_load.py --lleaves cases.json   # LightGBM 3.3.5 and lleaves 1.1.0
  .venv/bin/python check_load.py cases.json                  # LightGBM 4.x
"""

import argparse
import json
import os
import sys

import lightgbm as lgb
import numpy as np

LLEAVES_TOLERANCE = 1e-12


def floats(values):
    return np.array([float(v) for v in values], dtype=np.float64)


def compare_bits(name, what, got, expected):
    mismatch = np.flatnonzero(got.view(np.uint64) != expected.view(np.uint64))
    for i in mismatch[:5]:
        print(f"{name}: {what} row {i}: lightgbm {got[i]!r}, silvic {expected[i]!r}")
    return len(mismatch) == 0


def check_lleaves(name, model_file, rows, expected, skip):
    import lleaves

    model = lleaves.Model(model_file=model_file)
    model.compile()
    keep = np.setdiff1d(np.arange(len(rows)), np.array(skip, dtype=np.int64))
    got = model.predict(rows[keep], n_jobs=1)
    error = np.abs(got - expected[keep])
    bad = np.flatnonzero(~(error <= LLEAVES_TOLERANCE))
    for i in bad[:5]:
        print(f"{name}: lleaves row {keep[i]}: lleaves {got[i]!r}, silvic {expected[keep[i]]!r}")
    return len(bad) == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("cases")
    parser.add_argument("--lleaves", action="store_true", help="also compile and check with lleaves")
    args = parser.parse_args()
    with open(args.cases) as f:
        cases = json.load(f)["cases"]
    base = os.path.dirname(os.path.abspath(args.cases))
    ok = True
    for case in cases:
        name = case["name"]
        model_file = os.path.join(base, case["model_file"])
        rows = np.array([floats(row) for row in case["rows"]], dtype=np.float64)
        raw, predicted = floats(case["raw"]), floats(case["predicted"])
        booster = lgb.Booster(model_file=model_file)
        ok &= compare_bits(name, "raw", booster.predict(rows, raw_score=True), raw)
        ok &= compare_bits(name, "predicted", booster.predict(rows), predicted)
        if args.lleaves:
            ok &= check_lleaves(name, model_file, rows, predicted, case["lleaves_skip"])
    tools = f"lightgbm {lgb.__version__}" + (" and lleaves" if args.lleaves else "")
    print(f"{len(cases)} models checked with {tools}: {'ok' if ok else 'MISMATCH'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
