#!/usr/bin/env bash
# Runs the end-to-end test (tests/triton/test_backend.py), or bench/triton.py, against a real tritonserver in Docker:
#
# 1. builds libtriton_silvic.so for Linux in rust:1.94-slim-bookworm, with the target directory in the Docker volume
#    silvic-triton-target and the cargo registry in silvic-triton-cargo (not in the repository);
# 2. builds the silvic-triton-e2e image (tests/triton/Dockerfile: pytriton's tritonserver 2.51.0 and the clients);
# 3. writes the model repository with make_repository.py ($PYTHON, default .venv/bin/python, which needs silvic);
# 4. starts tritonserver with the backend in explicit model-control mode, every model loaded, on the Docker network
#    silvic-triton-net, and runs pytest (or the bench) in a second container on that network.
#
#   tests/triton/run.sh [pytest args]
#   tests/triton/run.sh bench [bench/triton.py args]
#
# SILVIC_TRITON_MODELS (default ~/.cache/silvic-triton/models) is the repository; Docker must be able to mount it.
set -euo pipefail
cd "$(dirname "$0")/../.."

models=${SILVIC_TRITON_MODELS:-$HOME/.cache/silvic-triton/models}
python=${PYTHON:-.venv/bin/python}
server=silvic-triton-server
network=silvic-triton-net

docker volume create silvic-triton-target >/dev/null
docker volume create silvic-triton-cargo >/dev/null
docker run --rm -v "$PWD":/src:ro -v silvic-triton-target:/target -v silvic-triton-cargo:/usr/local/cargo/registry \
    -e CARGO_TARGET_DIR=/target -e RUSTUP_TOOLCHAIN=1.94.1 -w /src rust:1.94-slim-bookworm \
    cargo build --release --locked -p silvic-triton
docker build -q -t silvic-triton-e2e tests/triton >/dev/null
"$python" tests/triton/make_repository.py --out "$models"

docker network inspect "$network" >/dev/null 2>&1 || docker network create "$network" >/dev/null
docker rm -f "$server" >/dev/null 2>&1 || true
trap 'docker rm -f "$server" >/dev/null' EXIT
docker run -d --name "$server" --network "$network" -v silvic-triton-target:/target:ro -v "$models":/models:ro \
    silvic-triton-e2e sh -c 'cp /target/release/libtriton_silvic.so /opt/tritonserver/backends/silvic/ &&
        exec tritonserver --model-repository=/models --model-control-mode=explicit --load-model="*" \
            --strict-readiness=true' >/dev/null
docker exec "$server" python -c '
import time, urllib.request
for _ in range(120):
    try:
        urllib.request.urlopen("http://localhost:8000/v2/health/ready")
        break
    except OSError:
        time.sleep(0.5)
else:
    raise SystemExit("tritonserver did not become ready")
' || { docker logs "$server"; exit 1; }

client=(docker run --rm --network "$network" -v "$PWD":/src:ro -v "$models":/models:ro -e PYTHONDONTWRITEBYTECODE=1
    -e TRITON_URL="$server:8000" -e TRITON_REPOSITORY=/models -w /src silvic-triton-e2e)
if [[ ${1:-} == bench ]]; then
    shift
    "${client[@]}" python bench/triton.py "$@"
else
    "${client[@]}" python -m pytest -p no:cacheprovider -v tests/triton "$@"
fi
