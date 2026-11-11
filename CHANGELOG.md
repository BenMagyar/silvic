# Changelog

## Initial release

silvic trains gradient-boosted trees for L2 regression, binary classification and lambdarank on CPU,
Apple Metal, NVIDIA CUDA and AWS Neuron. It reads and writes LightGBM-compatible model files.
Training uses its own quantization and can produce different trees from LightGBM.

The Python API provides `Dataset`, `train`, `Booster` and scikit-learn estimators. Ray distributed
training combines workers into one model and resumes from checkpoints after worker or spot-node loss.

Serving supports LightGBM and silvic models. CPU and CUDA serving default to float64, matching
LightGBM prediction semantics for supported models. CUDA offers optional float32 serving, and Neuron
serving uses explicit float32. The Triton Inference Server backend serves these model files with
configuration auto-completion and dynamic batching.
