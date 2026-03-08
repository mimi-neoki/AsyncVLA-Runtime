# AsyncVLA Runtime Setup Guide

This repository contains a split-inference runtime for AsyncVLA:

- **Server side (GPU workstation):** base VLA inference and projected token generation
- **Client side (Raspberry Pi 5):** edge adapter inference (Hailo) and control loop

## 1. Prerequisites

- Python `3.10`
- [`uv`](https://docs.astral.sh/uv/)
- Linux environment
- For server mode: NVIDIA GPU + CUDA-compatible PyTorch runtime
- For client mode: Raspberry Pi 5 (`aarch64`) and Hailo runtime

## 2. Clone and Enter Repository

```bash
git clone https://github.com/mimi-neoki/AsyncVLA-Runtime.git
cd AsyncVLA-Runtime
```

## 3. Environment Setup with `uv` Extras

This project uses extras to separate dependencies:

- `server`: base VLA server runtime
- `client`: Pi edge runtime
- `test`: test tooling

### Option A: Single environment (quick switching)

```bash
uv sync --extra server --extra test
# ... work on server
uv sync --extra client
# ... work on client
```

### Option B: Separate environments (recommended)

```bash
UV_PROJECT_ENVIRONMENT=.venv.server uv sync --extra server --extra test
UV_PROJECT_ENVIRONMENT=.venv.client uv sync --extra client
```

Then use the corresponding Python interpreter for each side:

- Server: `.venv.server/bin/python`
- Client: `.venv.client/bin/python`

## 4. Raspberry Pi 5 + HailoRT Setup

The client extra intentionally does **not** pin a local Hailo wheel file in `pyproject.toml`.
Install HailoRT wheel manually on Raspberry Pi 5:

```bash
.venv.client/bin/pip install hailort-5.2.0-cp310-cp310-linux_aarch64.whl
```

`hailort-5.2.0-cp310-cp310-linux_aarch64.whl` can be obtained from the **Hailo Developer Zone**.

## 5. Verify Installation

### Server environment

```bash
UV_PROJECT_ENVIRONMENT=.venv.server uv sync --extra server --extra test
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv.server/bin/pytest -q
```

### Client environment

```bash
UV_PROJECT_ENVIRONMENT=.venv.client uv sync --extra client
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
  --hf-dir ~/gitrepo/AsyncVLA_release \
  --asyncvla-repo-dir ~/gitrepo/AsyncVLA \
  --device cuda \
  --dtype float16 \
  --quantization 8bit
```

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

