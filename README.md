# Triton FlashAttention crash repro (gfx1151 / ROCm)

## ✅ Update: FIXED in Linux mainline

The issue reproduced here — a **HIP illegal memory access** crash on RDNA (gfx1151, e.g. Strix Halo) — was **not a Triton FlashAttention bug**, but rather a **kernel-side issue in the AMD GPU driver (amdgpu/MES)**.  
FlashAttention was simply a reliable way to **trigger** the problem because of its compute kernels.

**Root cause:**  
The kernel bug was in how the **MES (Micro Engine Scheduler)** handled long compute workloads.  
The following patch fixes it by enabling a safety bit (`lr_compute_wa`) to prevent MES hangs on long-running jobs:

> **Commit:** [drm/amdgpu: Enable MES lr_compute_wa by default](https://github.com/torvalds/linux/commit/1fb710793ce2619223adffaf981b1ff13cd48f17)  
> *"The MES set resources packet has an optional bit 'lr_compute_wa'  
> which can be used for preventing MES hangs on long compute jobs.  
> Set this bit by default."*

This change is merged in **Linus’ tree** and will be included in **Linux 6.18-rc1**.  
With this fix applied, the crash **no longer reproduces** — verified on my gfx1151 system using the same Triton FlashAttention workload.

### 🔧 How to get the fix on Fedora

You can easily install mainline kernels built directly from Linus’ tree:

```bash
sudo dnf -y copr enable @kernel-vanilla/stable
sudo dnf upgrade 'kernel*'
````

Then reboot into the updated kernel.
If you’re using Secure Boot, remember to **disable it** before booting unsigned kernels.

Alternatively, experienced users can **cherry-pick the individual commit** into their kernel tree — that’s how this fix was originally verified.

---

## Original description

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
