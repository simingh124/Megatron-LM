# Fused GDN Dependency Install Notes

This document explains how Codex should install the two libraries required by the fused ARMT Gated DeltaNet path:

- `flash-linear-attention`
- `causal-conv1d`

The goal is not to encode one machine's policy, but to provide a repeatable playbook that works across environments with minimal trial-and-error.

## Scope

Use this guide when the repo needs to run the fused GDN path:

- `recurrent_memory_backend="gated_deltanet"`
- `recurrent_gdn_use_fla_kernel=True`
- `recurrent_gdn_use_causal_conv1d=True`

Do not copy host-specific proxy rules, venv paths, or other global agent constraints into this document. Always adapt `python`, `pip`, package index, and proxy setup to the target environment.

## Versions That Worked Once

These versions were successfully verified together in one environment:

- `flash-linear-attention==0.4.2`
- `causal-conv1d==1.6.1`

Treat this as a known-good starting point, not a universal guarantee.

## Install Strategy

Recommended order:

1. Confirm the target environment's `python` and `pip`.
2. Check `torch` version, CUDA version, and GPU compute capability.
3. Install `flash-linear-attention`.
4. Install `causal-conv1d`.
5. Verify imports.
6. Run repo-level fused-path tests.

## 1. Preflight Checks

Always check the active environment first:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch.cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    major, minor = torch.cuda.get_device_capability(0)
    print("sm", f"{major}{minor}")
    print("gpu0", torch.cuda.get_device_name(0))
PY
```

Why this matters:

- `flash-linear-attention` and `causal-conv1d` are only useful for the fused CUDA path.
- `causal-conv1d` source builds are sensitive to the exact `torch` and CUDA pairing.
- The active GPU architecture determines which CUDA `sm` target is actually needed.

## 2. Install `flash-linear-attention`

Try the straightforward install first:

```bash
python -m pip install flash-linear-attention==0.4.2
```

Notes:

- This package usually installs more easily than `causal-conv1d`.
- It pulls in `fla-core`.
- If the environment requires a private package mirror or proxy, adapt the `pip install` command accordingly.

## 3. Install `causal-conv1d`

### Recommended first attempt

Try the normal install first:

```bash
python -m pip install causal-conv1d==1.6.1
```

If that succeeds, stop here and move to verification.

### Common failure: build isolation pulls the wrong `torch`

One real failure mode is:

- the current environment already has one `torch` build
- `pip` build isolation creates a temporary build env
- that build env installs a different `torch`
- CUDA/torch versions then mismatch during wheel build

Symptoms usually look like:

- `RuntimeError: The detected CUDA version ... mismatches the version that was used to compile PyTorch ...`

When that happens, retry without build isolation:

```bash
python -m pip install --no-build-isolation causal-conv1d==1.6.1
```

Important:

- Always use the intended environment's `python -m pip`, not a bare `pip`, to avoid accidentally building against the wrong Python or `torch`.

### Common failure: source build is extremely slow

Another real failure mode is that upstream `causal-conv1d` source builds may compile for many CUDA architectures, for example multiple `sm_*` targets unrelated to the current machine.

This makes the build much slower than necessary.

Important detail:

- Upstream `setup.py` may hardcode `-gencode` flags based on CUDA version.
- In that case, simply exporting `TORCH_CUDA_ARCH_LIST` is not enough, because the package build script may ignore it and still emit all its default targets.

### Recommended fallback: local source install patched to one active arch

If the normal source build is too slow or keeps rebuilding many architectures, use this workflow.

#### Step A: download source only

```bash
mkdir -p /tmp/causal-conv1d-build
python -m pip download --no-deps --no-binary :all: \
  "causal-conv1d==1.6.1" \
  -d /tmp/causal-conv1d-build
```

#### Step B: unpack it

```bash
cd /tmp/causal-conv1d-build
tar -xzf causal_conv1d-1.6.1.tar.gz
cd causal_conv1d-1.6.1
```

#### Step C: detect the active GPU arch

```bash
python - <<'PY'
import torch
major, minor = torch.cuda.get_device_capability(0)
print(f"{major}{minor}")
PY
```

Assume this prints `90` on an H800-like machine.

#### Step D: patch `setup.py`

Patch the CUDA `gencode` section so that it compiles only the active architecture instead of the full upstream list.

Conceptually, replace the upstream block that appends many targets such as:

- `sm_62`
- `sm_70`
- `sm_72`
- `sm_75`
- `sm_80`
- `sm_87`
- `sm_90`
- `sm_100`
- `sm_120`

with a minimal block like:

```python
if bare_metal_version < Version("11.8"):
    raise RuntimeError("causal-conv1d requires CUDA 11.8+ for an sm_90 build")
cc_flag.append("-gencode")
cc_flag.append("arch=compute_90,code=sm_90")
```

Replace `90` with the actual target arch detected on the current machine.

This patch must be applied only in the temporary downloaded source tree, not in this repo.

#### Step E: install from the patched local source

```bash
CUDA_HOME=/usr/local/cuda \
MAX_JOBS=4 \
CAUSAL_CONV1D_FORCE_BUILD=TRUE \
python -m pip install --no-build-isolation .
```

Adjust:

- `CUDA_HOME` if the environment uses a different CUDA toolkit path
- `MAX_JOBS` if the machine can tolerate more parallel compilation

## 4. Verification

After installation:

```bash
python - <<'PY'
import torch
import fla
import causal_conv1d

print("torch", torch.__version__)
print("torch.cuda", torch.version.cuda)
print("fla", getattr(fla, "__version__", "unknown"))
print("causal_conv1d", getattr(causal_conv1d, "__version__", "unknown"))
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
PY
```

## 5. Repo-Level Validation

If the packages import successfully, verify the actual fused path in this repo.

### Fused GDN path

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest \
  "examples/armt/tests/test_armt_train.py::TestARMTTraining::test_armt_single_step[gated_deltanet-True-True]" \
  -q
```

### TP=2 path

If at least 2 GPUs are available:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_port=29610 \
  -m pytest \
  examples/armt/tests/test_armt_train.py::TestARMTTraining::test_armt_tp \
  -q
```

## Pitfalls Summary

### Pitfall 1: using the wrong `pip`

Problem:

- `pip` may point to a different environment than the `python` used to run training.

Fix:

- Prefer `python -m pip ...` from the target environment.

### Pitfall 2: `causal-conv1d` build isolation installs a different `torch`

Problem:

- Temporary build env pulls another `torch`, causing CUDA mismatch at compile time.

Fix:

- Retry with `--no-build-isolation`.

### Pitfall 3: upstream `causal-conv1d` compiles too many GPU targets

Problem:

- Build time becomes unnecessarily long.

Fix:

- Patch the temporary source tree so it emits only the active `sm_*` target.

### Pitfall 4: upstream tries to fetch a prebuilt wheel that does not exist

Problem:

- The package may guess a wheel URL first and get a 404.

Fix:

- Force a local source build and install from the patched temporary source tree.

### Pitfall 5: import success is not enough

Problem:

- The libraries may install, but the repo's fused path can still fail at runtime.

Fix:

- Always run the repo-level fused GDN test after installation.

## Suggested Codex Behavior In New Environments

When Codex is asked to enable fused GDN support in a fresh environment, the preferred order is:

1. Inspect `torch`, CUDA, and GPU arch.
2. Install `flash-linear-attention`.
3. Try `causal-conv1d` normally.
4. If it fails with a CUDA/torch mismatch, retry with `--no-build-isolation`.
5. If it is building too many architectures, patch the temporary source tree to only the active `sm`.
6. Verify imports.
7. Run the fused GDN and TP tests in this repo.

That sequence proved more reliable than repeatedly retrying the same `pip install` command unchanged.
