# Triton FlashAttention crash repro (gfx1151 / ROCm)

This repo reproduces a **HIP illegal memory access** when using **Triton FlashAttention** on RDNA (gfx1151, e.g. Strix Halo) in the `Qwen/Qwen-Image` diffusers pipeline.  
The **same workload** using **PyTorch SDPA** (i.e., FlashAttention disabled) **does not crash**.

We provide:
- a **minimal Docker image** with TheRock ROCm wheels, Triton, and the ROCm FlashAttention fork
- a **simple runtime switch** (env var) to enable/disable the FA shim
- a small driver `qwen_crash_reproduction.py` (no LoRA, just denoising) that hits the same attention paths as Qwen Image Studio

## Build the container

```bash
docker build -t triton-fa-repro .
# or with podman:
# podman build -t triton-fa-repro .
````

## Run the container

You need GPU device access inside the container:

```bash
docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --security-opt seccomp=unconfined \
  triton-fa-repro
```

(Use the equivalent flags with `podman` if you prefer.)

If you already have downlaoded the model weights on your host and want to reuse the HuggingFacve cache:

```bash
podman run --rm -it \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --security-opt seccomp=unconfined \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface:Z \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  triton-fa-repro
```

## Reproduce

> `AMD_SERIALIZE_KERNEL=1` and/or `TRITON_DISABLE_AUTOTUNING=1` can make the failure surface sooner.

### 1) Crashy path (Triton FlashAttention via shim)

```bash
export QWEN_FA_SHIM=1           # enable FA shim
export QWEN_FA_DEBUG=0          # optionally set to 1 to see more logs
export AMD_SERIALIZE_KERNEL=1   # optional: surface HIP errors earlier
python qwen_crash_reproduction.py --prompt "dog" --steps 4 --size 16:9 --iters 10
# Expect: HIP illegal memory access during denoising on gfx1151 (intermittent but frequent)
```

### 2) Stable baseline (PyTorch SDPA)

```bash
unset QWEN_FA_SHIM              # disable FA shim -> SDPA
python qwen_crash_reproduction.py --prompt "dog" --steps 4 --size 16:9 --iters 10
# Expect: completes without HIP crashes
```

## Why this matters

* We first observed this in **Qwen Image Studio** while enabling Triton FlashAttention for speedups.
* The same failure pattern shows up here with the **exact attention call shape** and dtype (`bf16`, head dim 128), but **disappears** when using **standard PyTorch SDPA**.
* The repro avoids unrelated factors (e.g., LoRA merging) to isolate the kernel path.

## Notes

* Model: `Qwen/Qwen-Image` is public on Hugging Face (downloaded on first run).
* Environment variables you may toggle:

  * `AMD_SERIALIZE_KERNEL=1`
  * `TRITON_DISABLE_AUTOTUNING=1`
  * `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` (already set in image)
* Hardware: tested on gfx1151 (Strix Halo). Other RDNA parts may hit the same path.
