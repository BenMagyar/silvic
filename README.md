# silvic

silvic trains and serves gradient-boosted trees in Rust with a Python API and
LightGBM-compatible model files. It supports L2 regression, binary
classification and learning to rank (`lambdarank`). Training uses its own
quantization and can produce different trees from LightGBM.

[Documentation](https://benmagyar.github.io/silvic/)

## Installation and use

```sh
pip install silvic
```

The Python API provides `silvic.Dataset` and `silvic.train`, with
`SilvicRegressor`, `SilvicClassifier` and `SilvicRanker` estimators.
The `silvic[ray]` extra provides distributed training with Ray workers and
checkpoint recovery.

## Backends

Training supports CPU, Apple Metal, NVIDIA CUDA and AWS Neuron
(Trainium, Trainium2 and Inferentia2) through explicit device selection.
Serving supports CPU, CUDA and Neuron. CPU and CUDA default to float64 prediction
semantics for supported LightGBM model files; CUDA also supports float32, and
Neuron serving uses explicitly selected float32. Metal supports training only.

## License

silvic is licensed under the [MIT License](LICENSE).
