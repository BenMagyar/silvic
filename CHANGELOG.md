# Changelog

## Unreleased

- `silvic-core`: LightGBM 3.3.5 parameters by name and alias (`Params::from_json`). Unknown keys and LightGBM
  features outside silvic's scope (other objectives, DART, GOSS, early stopping, monotone constraints, linear trees)
  are errors; keys that only tune LightGBM's own execution (`force_row_wise`, `deterministic`, ...) are accepted and
  have no effect.
- `silvic-core`: binning ported from LightGBM 3.3.5 `BinMapper::FindBin` and `Random::Sample`: for the same sample,
  every threshold LightGBM splits on is a silvic bin boundary, bit for bit, and `feature_infos` matches. Bin mappers
  come from a sample allgathered over every rank, so all ranks bin identically; a rank whose input fails validation
  fails every rank with the same error.
- `silvic-core`: objectives `regression` (L2), `binary` and `lambdarank` (with `label_gain`, truncation,
  `lambdarank_norm` and LightGBM's sigmoid table) and metrics `l2`, `rmse`, `binary_logloss`, `auc` and `ndcg@k`,
  matching LightGBM 4.7 to 1e-12. Lambdarank has no query-size limit: LightGBM's 10000-position discount table is
  extended by its formula.
- `silvic-core`: leaf-wise trees on quantized gradients by default (`use_quantized_grad`, `num_grad_quant_bins` 4,
  stochastic rounding) with leaves renewed from exact fixed-point gradient sums (`quant_train_renew_leaf` defaults to
  true, unlike LightGBM). Histograms are integer sums keyed by global row id, so training is bit-reproducible and N
  ranks train the model of one. Float histograms remain available with `use_quantized_grad=false`.
- `silvic-core`: a leaf whose rows all have zero hessian is never split and an empty child is rejected, so
  lambdarank data with many queries without a relevant item trains in quantized mode, where LightGBM 4.7 aborts with
  `Check failed: (best_split_info.left_count) > (0)`.
- `silvic-core`: model text byte-identical to LightGBM 3.3.5's `model_to_string()`; LightGBM 3.3.5 and 4.x models
  parse and write back byte for byte, and predictions equal LightGBM's bit for bit, NaN, zero and unseen categories
  included. Models load in LightGBM 3.3.5 and 4.7, give `pred_contrib` SHAP values that sum to the prediction, and
  compile in lleaves 1.1.0.
- `silvic-core`: `CpuBackend`, the reference backend, parallel over rayon: per-leaf feature masks, prefetched row
  gathers, deterministic merges.
- `silvic-gpu`: Metal and CUDA backends from one kernel source compiled at run time. Quantization, quantized
  histograms, partitions, leaf sums and score updates match `CpuBackend` bit for bit on Metal, and quantized L2
  models are byte-identical across the two; binary and lambdarank gradients agree within 1e-5 (Metal computes them in
  f32). One device sync per split: the partition and the smaller child's histogram run in one command buffer.
- `silvic-gpu`: the CUDA device layer (cudarc with the driver and NVRTC loaded at run time, `--fmad=false`) builds
  without a CUDA toolkit and never panics on a missing or mismatched driver, and its kernel source passes clang's
  CUDA front end; it has not run on NVIDIA hardware.
- `silvic-net`: `TcpCollective`, ring allreduce and allgather over TCP with a token handshake, deterministic fold
  order, framed operations that detect diverging ranks, and failures that propagate around the ring instead of
  hanging.
- `silvic-python`: the `silvic` package mirroring LightGBM's Python API (`Dataset`, `train`, `Booster`,
  `log_evaluation`, `record_evaluation`, `SilvicRegressor`, `SilvicClassifier`, `SilvicRanker`), pandas categoricals
  stored as LightGBM stores them, `Booster.to_lightgbm()`, and `device_type` `auto`/`cpu`/`gpu`/`metal`/`cuda`.
- `silvic-python`: `silvic.ray.train` trains one model across Ray actors in a placement group: rows are hash-split
  by query group in one streaming pass, each actor's rows are put in a data-only order, and the actors reduce over a
  `TcpCollective` ring; the model equals single-process training on the same ordered rows, and an actor failure kills
  the rest and raises the root error.
