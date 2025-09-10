# Minimal ROCm + PyTorch (TheRock) + FlashAttention (Triton AMD) + Qwen repro
FROM registry.fedoraproject.org/fedora:rawhide

# --- Base system deps (keep toolchain for Triton JIT + ps/kill) ---
RUN dnf -y install --setopt=install_weak_deps=False --nodocs \
      python3.13 python3.13-devel \
      gcc gcc-c++ binutils make \
      git rsync curl ca-certificates \
      bash coreutils procps-ng which findutils \
      libatomic libdrm-devel \
  && dnf clean all && rm -rf /var/cache/dnf/*

# --- Python venv ---
RUN /usr/bin/python3.13 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
ENV PIP_NO_CACHE_DIR=1
RUN printf 'source /opt/venv/bin/activate\n' > /etc/profile.d/venv.sh && chmod 0644 /etc/profile.d/venv.sh
RUN python -m pip install --upgrade pip setuptools wheel

# --- ROCm + PyTorch (TheRock) ---
ARG ROCM_INDEX=https://rocm.nightlies.amd.com/v2/gfx1151
RUN python -m pip install --index-url ${ROCM_INDEX} 'rocm[libraries,devel]' && \
    python -m pip install --index-url ${ROCM_INDEX} --pre torch pytorch-triton-rocm numpy

# --- Python libs needed by the repro (CPU wheels are fine) ---
RUN python -m pip install \
      diffusers transformers accelerate huggingface_hub safetensors \
      einops packaging psutil Pillow regex tqdm requests

# --- Hugging Face cache path (host will bind-mount here) ---
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers

# --- Enable Triton AMD backend in flash-attn at runtime ---
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE

# --- Build & install FlashAttention (ROCm fork) ---
WORKDIR /opt
RUN git clone https://github.com/ROCm/flash-attention.git && \
    cd flash-attention && \
    git checkout main_perf && \
    python -m pip install packaging && \
    python setup.py install && \
    cd /opt && rm -rf /opt/flash-attention

# --- Persist ROCm/Triton env for interactive shells (auto-set toolchain paths) ---
RUN cat >/etc/profile.d/01-rocm-env-for-triton.sh <<'EOS' && chmod 0644 /etc/profile.d/01-rocm-env-for-triton.sh
#!/usr/bin/env bash
# Detect and export ROCm toolchain paths from the _rocm_sdk_core package
eval "$(
python3 - <<'PY'
import pathlib, _rocm_sdk_core as r
base = pathlib.Path(r.__file__).parent / "lib" / "llvm" / "bin"
lib  = pathlib.Path(r.__file__).parent / "lib"
print(f'export TRITON_HIP_LLD_PATH="{base / "ld.lld"}"')
print(f'export TRITON_HIP_CLANG_PATH="{base / "clang++"}"')
print(f'export LD_LIBRARY_PATH="{lib}:$LD_LIBRARY_PATH"')
PY
)"
# Enable Triton AMD backend for flash-attn
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
EOS

# --- Bring in the minimal repro repo (contains qwen_crash_reproduction.py) ---
RUN git clone --depth=1 https://github.com/kyuz0/triton_gfx1151_crashes /opt/triton_gfx1151_crashes && \
    rm -rf /opt/triton_gfx1151_crashes/.git
WORKDIR /opt/triton_gfx1151_crashes

# Drop into a login shell so /etc/profile.d/* and venv are loaded
CMD ["/bin/bash", "-l"]
