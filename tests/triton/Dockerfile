# A small tritonserver for the end-to-end test and bench: the tritonserver 2.51.0 (NGC 24.10) binary that the
# nvidia-pytriton 0.7.0 wheel bundles, which runs without a GPU, plus the test's clients (tritonclient, LightGBM,
# pytest). 0.9 GB, against 21 GB (a 7.5 GB download) for nvcr.io/nvidia/tritonserver:24.10-py3 on arm64. The wheel is
# manylinux_2_35, so the base is Debian bookworm (glibc 2.36).
#
#   docker build -t silvic-triton-e2e tests/triton
#
# tests/triton/run.sh builds the backend, this image, a model repository, and runs the test against it.
FROM python:3.12-slim-bookworm

# libgomp1: LightGBM's OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "nvidia-pytriton==0.7.0" "tritonclient[http,grpc]==2.74.0" "lightgbm==4.6.0" \
        "pytest==8.4.2" \
    && ln -s "$(python -c 'import pytriton, os; print(os.path.dirname(pytriton.__file__))')/tritonserver" \
        /opt/tritonserver \
    && mkdir -p /opt/tritonserver/backends/silvic

ENV PATH=/opt/tritonserver/bin:$PATH
