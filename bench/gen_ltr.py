"""Synthetic learning-to-rank data.

136 dense features (MSLR-WEB30K's width): 90 integer counts with many zeros, 41
floats, 5 low-cardinality categoricals (<= 20 values). Query sizes are log-normal around 100 with a long tail;
labels are graded relevance 0..4 drawn from a hidden nonlinear score, with 30% of queries holding no relevant item.
Rows are sorted by query, as LightGBM requires.

    python bench/gen_ltr.py --rows 2000000 --out bench/data/ltr_2m

writes `<out>.train.npz` and `<out>.valid.npz` (valid is 10% of the queries) with arrays `X` (float32, C order),
`y` (float32), `group` (uint32 query sizes) and `categorical` (feature indices).
"""

import argparse
from pathlib import Path

import numpy as np

NUM_COUNTS = 90
NUM_FLOATS = 41
NUM_CATEGORICAL = 5
CATEGORY_LEVELS = (4, 7, 12, 16, 20)


def query_sizes(rng, rows):
    sizes = []
    total = 0
    while total < rows:
        size = int(min(5000, max(2, rng.lognormal(mean=4.4, sigma=0.8))))
        size = min(size, rows - total) if rows - total >= 2 else 2
        sizes.append(size)
        total += size
    return np.array(sizes, dtype=np.uint32)


def features(rng, rows):
    counts = rng.negative_binomial(n=0.6, p=0.15, size=(rows, NUM_COUNTS)).astype(np.float32)
    floats = rng.normal(size=(rows, NUM_FLOATS)).astype(np.float32)
    floats[rng.random(size=floats.shape) < 0.02] = np.nan
    categorical = np.stack([rng.integers(0, levels, size=rows) for levels in CATEGORY_LEVELS], axis=1)
    return np.concatenate([counts, floats, categorical.astype(np.float32)], axis=1)


def labels(rng, X, sizes):
    counts = np.log1p(X[:, :NUM_COUNTS])
    floats = np.nan_to_num(X[:, NUM_COUNTS : NUM_COUNTS + NUM_FLOATS])
    cats = X[:, NUM_COUNTS + NUM_FLOATS :].astype(np.int64)
    weights = rng.normal(size=NUM_COUNTS) * (rng.random(NUM_COUNTS) < 0.3)
    score = counts @ weights * 0.3 + np.tanh(floats[:, 0] * floats[:, 1]) + 0.5 * np.sin(floats[:, 2] * 2)
    score += 0.8 * (cats[:, 0] == 2) - 0.6 * (cats[:, 3] % 3 == 0) + 0.3 * (counts[:, 0] > 1.5) * floats[:, 3]
    score += rng.normal(scale=0.7, size=len(score))
    y = np.zeros(len(score), dtype=np.float32)
    start = 0
    for size in sizes:
        end = start + int(size)
        if rng.random() >= 0.3:
            q = score[start:end]
            ranks = q.argsort().argsort() / max(1, end - start - 1)
            y[start:end] = np.digitize(ranks, [0.6, 0.8, 0.92, 0.98]).astype(np.float32)
        start = end
    return y


def generate(rows, seed=7):
    """Train and validation splits (validation is the last 10% of the queries) as dicts of numpy arrays."""
    rng = np.random.default_rng(seed)
    sizes = query_sizes(rng, rows)
    total = int(sizes.sum())
    X = features(rng, total)
    y = labels(rng, X, sizes)
    categorical = np.arange(NUM_COUNTS + NUM_FLOATS, NUM_COUNTS + NUM_FLOATS + NUM_CATEGORICAL)
    num_valid_queries = max(1, len(sizes) // 10)
    split_row = int(sizes[:-num_valid_queries].sum())
    train = {"X": X[:split_row], "y": y[:split_row], "group": sizes[:-num_valid_queries], "categorical": categorical}
    valid = {"X": X[split_row:], "y": y[split_row:], "group": sizes[-num_valid_queries:], "categorical": categorical}
    return train, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    train, valid = generate(args.rows, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f"{args.out}.train.npz", **train)
    np.savez(f"{args.out}.valid.npz", **valid)
    rows = len(train["y"]) + len(valid["y"])
    queries = len(train["group"]) + len(valid["group"])
    print(f"{rows} rows, {queries} queries, {train['X'].shape[1]} features -> {args.out}.{{train,valid}}.npz")


if __name__ == "__main__":
    main()
