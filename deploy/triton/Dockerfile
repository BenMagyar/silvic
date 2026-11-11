# tritonserver with the silvic backend: libtriton_silvic.so, built from this repository, in
# /opt/tritonserver/backends/silvic/ of NVIDIA's tritonserver image. Build from the repository root:
#
#   docker build -f deploy/triton/Dockerfile -t tritonserver-silvic .
#   docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v "$PWD/models":/models tritonserver-silvic \
#       tritonserver --model-repository=/models
#
# Both stages follow the target platform (linux/amd64 or linux/arm64). TRITON_VERSION is the NGC release; 24.10 is
# tritonserver 2.51.0, the server the end-to-end test ran against (tests/triton). The backend's glibc floor is
# 2.34 (bookworm builds it; the 24.10 image is Ubuntu 22.04, glibc 2.35).
ARG TRITON_VERSION=24.10

FROM rust:1.94-slim-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock rust-toolchain.toml rustfmt.toml ./
COPY crates crates
# rust-toolchain.toml asks for stable; build with the image's toolchain, the workspace's rust-version.
ENV RUSTUP_TOOLCHAIN=1.94.1
RUN cargo build --release --locked -p silvic-triton

FROM nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-py3
COPY --from=build /src/target/release/libtriton_silvic.so /opt/tritonserver/backends/silvic/libtriton_silvic.so
