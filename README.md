# AsyncVLA Runtime Setup Guide

This repository contains an unofficial split-inference runtime for [AsyncVLA](https://asyncvla.github.io/):

- **Server side (GPU workstation):** base VLA inference and projected token generation
- **Client side (Raspberry Pi 5):** edge adapter inference (Hailo) and control loop

## 1. Prerequisites

- Python `3.10`
- [`uv`](https://docs.astral.sh/uv/)
- Linux environment
- For server mode: NVIDIA GPU + CUDA-compatible PyTorch runtime
- For client mode: Raspberry Pi 5 (`aarch64`) and Hailo runtime (AI HAT)

## 2. Clone and Enter Repository

```bash
# Download model weight
git clone https://huggingface.co/NHirose/AsyncVLA_release
cd AsyncVLA_release
git-lfs pull
cd ..

# Download code
git clone https://github.com/mimi-neoki/AsyncVLA.git
git clone https://github.com/mimi-neoki/AsyncVLA-Runtime.git
cd AsyncVLA-Runtime
```

## 3. Environment Setup with `uv`

This repository is split into two `uv` projects:

- Root project (`./pyproject.toml`): server runtime (`server`, `test` extras)
- Client project (`./client/pyproject.toml`): Pi edge runtime (`lerobot`, `hailort`, etc.)

### Option A: Single environment (quick switching)

```bash
uv sync --extra server --extra test
# ... work on server
uv sync --project client
# ... work on client
```

### Option B: Separate environments (recommended)

```bash
UV_PROJECT_ENVIRONMENT=.venv.server uv sync --extra server --extra test
UV_PROJECT_ENVIRONMENT=.venv.client uv sync --project client
```

Then use the corresponding Python interpreter for each side:

- Server: `.venv.server/bin/python`
- Client: `.venv.client/bin/python`

## 4. Raspberry Pi 5 + HailoRT Setup

The client project references a local HailoRT wheel path in `client/pyproject.toml`:

```toml
hailort = { path = "../hailort-5.2.0-cp310-cp310-linux_aarch64.whl" }
```

Place the wheel in the repository root before syncing the client project.
If you already created `.venv.client` without it, you can also install manually:

```bash
.venv.client/bin/pip install hailort-5.2.0-cp310-cp310-linux_aarch64.whl
```

`hailort-5.2.0-cp310-cp310-linux_aarch64.whl` can be obtained from the **Hailo Developer Zone**.

You need to compile model into .hef for inference on Hailo chip 

## 5. Verify Installation

### Server environment

```bash
UV_PROJECT_ENVIRONMENT=.venv.server uv sync --extra server --extra test
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv.server/bin/pytest -q
```

### Client environment

```bash
UV_PROJECT_ENVIRONMENT=.venv.client uv sync --project client
.venv.client/bin/python -c "import cv2; print(cv2.__version__)"
```

After installing HailoRT wheel:

```bash
.venv.client/bin/python -c "import hailo_platform; print('hailo_platform OK')"
```

## 6. Run Commands

### Start base VLA server (workstation)

```bash
PYTHONPATH=. .venv.server/bin/python scripts/run_base_vla_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --hf-dir <path to AsyncVLA Huggingface model repository> \
  --asyncvla-repo-dir <path to AsyncVLA repository> \
  --device cuda \
  --dtype float16 \

```
On MacBook GPU, use `--device mps` instead of `--device cuda`.
If you face a CUDA OOM, you can use ```--quantization 8bit```

### Start Pi edge client (Raspberry Pi 5)

```bash
PYTHONPATH=. .venv.client/bin/python scripts/run_pi_edge_client.py \
  --policy-url http://<server-ip>:8000/infer \
  --hef models/edge_adapter_v520.hef
```

Optional language/task overrides can be sent from client:

```bash
--instruction "go to the target" --task-mode language_and_pose --task-id 8
```
