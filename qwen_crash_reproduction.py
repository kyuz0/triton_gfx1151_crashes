#!/usr/bin/env python3
# Minimal Qwen-Image FlashAttention repro (no LoRA).

import os, sys, time, secrets
from datetime import datetime

import torch
import torch.nn.functional as F

# --- FlashAttention shim (simple on/off switch) ---
if os.getenv("QWEN_FA_SHIM", "0").lower() in {"1", "true", "yes"}:
    try:
        from flash_attn.flash_attn_interface import flash_attn_func as _fa
    except Exception:
        _fa = None

    import torch
    import torch.nn.functional as F

    _orig = F.scaled_dot_product_attention
    _dbg  = os.getenv("QWEN_FA_DEBUG", "0").lower() in {"1","true","yes","on"}
    _sync = os.getenv("QWEN_FA_SYNC",  "0").lower() in {"1","true","yes","on"}

    def _sdpa_or_fa(*args, **kw):
        # Extract common args from both positional/keyword call patterns
        q = kw.get("query", args[0] if args else None)
        k = kw.get("key",   args[1] if len(args) > 1 else None)
        v = kw.get("value", args[2] if len(args) > 2 else None)
        attn_mask = kw.get("attn_mask", kw.get("attention_mask"))
        dropout_p = kw.get("dropout_p", 0.0)
        is_causal = kw.get("is_causal", False)
        scale     = kw.get("scale", kw.get("softmax_scale"))

        # Minimal preconditions so we don't call FA in clearly-unsupported cases
        can_try_fa = (
            _fa is not None
            and q is not None and k is not None and v is not None
            and attn_mask is None
            and dropout_p == 0.0
            and not is_causal
            and q.is_cuda and k.is_cuda and v.is_cuda
            and q.dtype in (torch.float16, torch.bfloat16)
            and q.shape[1] == k.shape[1] == v.shape[1]
        )

        if can_try_fa:
            try:
                if scale is None:
                    scale = (q.shape[-1]) ** -0.5
                out = _fa(
                    q.transpose(1, 2).contiguous(),
                    k.transpose(1, 2).contiguous(),
                    v.transpose(1, 2).contiguous(),
                    dropout_p=0.0,
                    softmax_scale=scale,
                    causal=False,
                ).transpose(1, 2)
                if _sync:
                    torch.cuda.synchronize()
                if _dbg:
                    print("ATTN: FA")
                return out
            except Exception as e:
                if _dbg:
                    print(f"ATTN: FA->SDPA due to {type(e).__name__}: {e}")

        if _dbg:
            print("ATTN: SDPA")
        return _orig(*args, **kw)

    F.scaled_dot_product_attention = _sdpa_or_fa
    print("ATTN: FA shim ON")
else:
    print("ATTN: FA shim OFF")
# --------------------------------------------------


def _rt_no_sigmas(
    scheduler,
    num_inference_steps=None,
    device=None,
    timesteps=None,
    sigmas=None,
    **kwargs,
):
    scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    ts = scheduler.timesteps
    return ts, len(ts)

def get_device_and_dtype():
    if torch.cuda.is_available():
        print("Using CUDA/ROCm")
        return "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        print("Using MPS")
        return "mps", torch.bfloat16
    else:
        print("Using CPU")
        return "cpu", torch.float32

def main():
    import argparse
    from diffusers import DiffusionPipeline
    from diffusers.pipelines.qwenimage import pipeline_qwenimage as _qimg

    ap = argparse.ArgumentParser("Qwen-Image FA repro (no LoRA)")
    ap.add_argument("--prompt", default="dog", type=str)
    ap.add_argument("--steps", default=50, type=int)
    ap.add_argument("--iters", default=5, type=int)
    ap.add_argument("--num-images", default=1, type=int)
    ap.add_argument("--size", default="16:9",
                    choices=["1:1","16:9","9:16","4:3","3:4","3:2","2:3"])
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    device, torch_dtype = get_device_and_dtype()
    torch.set_default_device(device)

    print(f"CLI: Loading base pipeline: Qwen/Qwen-Image (dtype={torch_dtype})")
    pipe = DiffusionPipeline.from_pretrained(
        "Qwen/Qwen-Image",
        torch_dtype=torch_dtype,
        use_safetensors=True,
        device_map=device,
    )

    # Patch FlowMatch timesteps (match your CLI)
    _qimg.retrieve_timesteps = _rt_no_sigmas

    # VAE bf16 + tiling (match your CLI)
    try:
        pipe.vae.to(device=device, dtype=torch_dtype)
        if hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        print("VAE: native tiling ENABLED (bf16)")
    except Exception as e:
        print(f"VAE: native tiling not available ({e})")

    pipe.set_progress_bar_config(disable=False, leave=True, miniters=1)

    negative_prompt = " "
    aspect_ratios = {
        "1:1": (1328, 1328),
        "16:9": (1664, 928),
        "9:16": (928, 1664),
        "4:3": (1472, 1140),
        "3:4": (1140, 1472),
        "3:2": (1584, 1056),
        "2:3": (1056, 1584),
    }
    width, height = aspect_ratios[args.size]
    num_images = max(1, int(args.num_images))
    steps = int(args.steps)
    cfg_scale = 4.0

    print(f"CLI: Generation config: steps={steps}, cfg={cfg_scale}, size={args.size}, images={num_images}")

    for i in range(1, args.iters + 1):
        seed = (args.seed if args.seed is not None else secrets.randbits(63)) + (i - 1)
        gen_device = "cpu" if device == "mps" else device
        generator = torch.Generator(device=gen_device).manual_seed(seed)

        print(f"\n[iter {i}/{args.iters}] seed={seed}")
        t0 = time.perf_counter()
        try:
            out = pipe(
                prompt=args.prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                true_cfg_scale=cfg_scale,
                generator=generator,
            )
            _ = out.images[0]
        except Exception as ex:
            dt = time.perf_counter() - t0
            print(f"[FAIL] iter={i} after {dt:.2f}s: {type(ex).__name__}: {ex}")
            print("NOTE: If this is a HIP/Triton crash, consider AMD_SERIALIZE_KERNEL=1 and/or TRITON_DISABLE_AUTOTUNING=1")
            sys.exit(2)

        dt = time.perf_counter() - t0
        print(f"[ok] iter={i} done in {dt:.2f}s")

    print("\n[done] all iterations completed without exceptions.")

if __name__ == "__main__":
    main()
