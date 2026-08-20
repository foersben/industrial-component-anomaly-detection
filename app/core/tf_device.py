"""TensorFlow device configuration with automatic CUDA and VRAM detection.

Why Runtime Detection in Addition to Install-Time Feature Selection?
======================================================================
Pixi's ``gpu`` and ``cpu`` install-time features determine *which build* of TensorFlow
is installed (``tensorflow[and-cuda]`` vs ``tensorflow-cpu``). However, several runtime
situations make a second, dynamic check necessary:

1. **Mixed environments**: A developer may have the ``gpu`` feature installed on a
   machine that temporarily lacks a working CUDA driver (e.g., after an OS update).
   Without runtime detection, TF would crash with a cryptic CUDA initialisation error.

2. **Insufficient VRAM**: The GPU feature is installed but the card only has 1 GB VRAM,
   which is not enough to hold the CAE model + batch data. Running on CPU is safer.

3. **Shared GPU systems**: An HPC cluster may have CUDA available but all GPU memory is
   currently occupied. The module can detect free VRAM and fall back to CPU.

4. **Optimal CPU threading**: Even when a GPU is available, the pure-CPU path needs
   explicit configuration to exploit AVX2 SIMD and oneDNN.

CUDA and VRAM Detection Strategy
----------------------------------
The module attempts detection in two ways, in priority order:

1. **nvidia-smi subprocess** (reliable, no Python deps, works with any driver):
   Queries free VRAM in MiB for the first available GPU.

2. **pynvml** (Python bindings to NVML, already in ``pyproject.toml`` deps):
   More structured, avoids subprocess overhead. Used as fallback to nvidia-smi.

3. **Graceful fallback**: If neither works, silently falls back to CPU mode.

CPU AVX2 / SIMD Configuration
--------------------------------
When running on CPU, ``tensorflow-cpu`` is already compiled with AVX2 support.
AVX2 (Advanced Vector Extensions 2) is a 256-bit SIMD instruction set present on
all Intel Haswell (2013+) and AMD Zen (2017+) processors.

Benefits for convolution and matrix multiplication:
- **float32**: Processes 8 values per AVX2 register simultaneously (256 / 32 = 8).
- **float16**: Processes 16 values simultaneously.
- Result: 4–8× throughput improvement over scalar code for conv2d and dense layers.

To fully activate AVX2 in TensorFlow-CPU, we set:
- ``TF_ENABLE_ONEDNN_OPTS=1``: Enables Intel oneDNN (formerly MKL-DNN), which provides
  AVX2/AVX-512 optimised kernels for all convolution, pooling, and matmul operations.
- ``OMP_NUM_THREADS``: Sets the number of OpenMP threads to the full CPU core count.
- ``TF_NUM_INTRAOP_THREADS``: Number of threads used within a single TF op (e.g., a
  single convolution layer). Should equal the number of physical cores.
- ``TF_NUM_INTEROP_THREADS``: Threads running different TF ops in parallel. Set to 1
  to avoid contention when running sequential pipeline operations.

These must be set **before** importing TensorFlow, which is why environment variables
are used (they are read at library load time, not at runtime).

GPU Configuration
------------------
When running on GPU, we enable **memory growth** (``set_memory_growth``):
- By default, TF allocates ALL available VRAM at startup, starving other processes.
- Memory growth causes TF to allocate only as much VRAM as actually needed, growing
  incrementally. This allows multiple models (e.g., PyTorch PatchCore + TF CAE) to
  coexist on the same GPU.

Module Contents
---------------
- ``HardwareProfile``: Dataclass summarising detected hardware.
- ``detect_hardware``: Detect CUDA availability and free VRAM.
- ``configure_tensorflow``: Configure TF for GPU or CPU-AVX2 based on hardware.
- ``get_device_summary``: Human-readable summary of the selected configuration.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Minimum free VRAM (in MiB) required to use the GPU path.
# Below this threshold we fall back to CPU even if CUDA is technically available.
# 2048 MiB = 2 GB: comfortably fits the CAE model + one batch at img_size=256.
DEFAULT_MIN_VRAM_MIB: int = 2048


@dataclass
class HardwareProfile:
    """Detected hardware capabilities and the resulting TF device strategy.

    Attributes:
        cuda_available: True if at least one CUDA-capable GPU was detected.
        gpu_name: GPU model name string (empty string if no GPU found).
        total_vram_mib: Total VRAM of the primary GPU in MiB (0 if no GPU).
        free_vram_mib: Free VRAM of the primary GPU in MiB (0 if no GPU).
        cpu_cores: Number of logical CPU cores available.
        has_avx2: True if the CPU reports AVX2 support in /proc/cpuinfo.
        selected_device: ``"gpu"`` or ``"cpu"`` chosen by ``configure_tensorflow``.
        reason: Human-readable explanation of why the device was selected.
    """

    cuda_available: bool = False
    gpu_name: str = ""
    total_vram_mib: int = 0
    free_vram_mib: int = 0
    cpu_cores: int = 4
    has_avx2: bool = False
    selected_device: str = "cpu"
    reason: str = "Not yet configured."
    tf_config_applied: list[str] = field(default_factory=list)


def _query_nvidia_smi() -> tuple[bool, str, int, int]:
    """Query GPU information via the nvidia-smi command-line tool.

    Returns:
        Tuple of (cuda_available, gpu_name, total_vram_mib, free_vram_mib).
        All values are 0/empty/False if nvidia-smi is not found or fails.
    """
    query = "gpu_name,memory.total,memory.free"
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return False, "", 0, 0

        first_line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in first_line.split(",")]
        if len(parts) < 3:
            return False, "", 0, 0

        name = parts[0]
        total_mib = int(parts[1])
        free_mib = int(parts[2])
        return True, name, total_mib, free_mib

    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        return False, "", 0, 0


def _query_pynvml() -> tuple[bool, str, int, int]:
    """Query GPU information via the pynvml Python bindings (NVML C library).

    pynvml is more structured than nvidia-smi and avoids subprocess overhead.
    It reads directly from the NVML shared library, which is installed alongside
    the CUDA driver.

    Returns:
        Tuple of (cuda_available, gpu_name, total_vram_mib, free_vram_mib).
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name_raw = pynvml.nvmlDeviceGetName(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        # nvmlDeviceGetName may return bytes or str depending on pynvml version
        name = name_raw.decode("utf-8") if isinstance(name_raw, bytes) else str(name_raw)
        total_mib = int(mem_info.total) // (1024 * 1024)
        free_mib = int(mem_info.free) // (1024 * 1024)

        return True, name, total_mib, free_mib

    except Exception as exc:
        logger.debug("pynvml query failed: %s", exc)
        return False, "", 0, 0


def _check_avx2_support() -> bool:
    """Check if the current CPU supports AVX2 (256-bit SIMD) via /proc/cpuinfo.

    Returns:
        True if the CPU flags include AVX2, False otherwise or on non-Linux systems.
    """
    try:
        with open("/proc/cpuinfo") as f:
            return "avx2" in f.read()
    except OSError:
        return False


def detect_hardware() -> HardwareProfile:
    """Detect available hardware: CUDA GPU presence, VRAM, CPU cores, AVX2 support.

    Detection order for CUDA/VRAM:
    1. Try nvidia-smi (works with any driver, no Python deps).
    2. Fall back to pynvml (Python NVML bindings, already installed).
    3. If both fail, assume CPU-only.

    Returns:
        Populated ``HardwareProfile`` dataclass (``selected_device`` is not yet
        set — call ``configure_tensorflow()`` to finalise it).
    """
    profile = HardwareProfile()
    profile.cpu_cores = os.cpu_count() or 4
    profile.has_avx2 = _check_avx2_support()

    # Try nvidia-smi first
    cuda_ok, gpu_name, total_mib, free_mib = _query_nvidia_smi()

    if not cuda_ok:
        # Fall back to pynvml
        cuda_ok, gpu_name, total_mib, free_mib = _query_pynvml()

    profile.cuda_available = cuda_ok
    profile.gpu_name = gpu_name
    profile.total_vram_mib = total_mib
    profile.free_vram_mib = free_mib

    logger.info(
        "Hardware detected — CUDA: %s | GPU: %s | VRAM: %d MiB free / %d MiB total | CPU cores: %d | AVX2: %s",
        cuda_ok,
        gpu_name or "N/A",
        free_mib,
        total_mib,
        profile.cpu_cores,
        profile.has_avx2,
    )

    return profile


def _apply_gpu_config(profile: HardwareProfile) -> None:
    """Apply TensorFlow GPU configuration (memory growth, logging).

    Memory growth prevents TF from claiming all VRAM at startup, allowing the
    PyTorch-based models to coexist on the same GPU.

    Args:
        profile: Hardware profile to record applied settings into.
    """
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            logger.warning("TF finds no GPU devices even though CUDA was detected. Falling back to CPU.")
            _apply_cpu_avx2_config(profile)
            return

        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        profile.selected_device = "gpu"
        profile.tf_config_applied.append("memory_growth=True on all GPUs")
        logger.info("TF GPU configured: %d device(s), memory growth enabled.", len(gpus))

    except Exception as exc:
        logger.warning("GPU TF configuration failed (%s). Falling back to CPU-AVX2.", exc)
        _apply_cpu_avx2_config(profile)


def _apply_cpu_avx2_config(profile: HardwareProfile) -> None:
    """Apply TensorFlow CPU configuration optimised for AVX2 SIMD.

    Environment variables are set BEFORE TF imports them at library load time.
    TF threading is also configured programmatically if TF is already loaded.

    Key settings:
    - ``TF_ENABLE_ONEDNN_OPTS=1``: Activates Intel oneDNN (MKL-DNN), which provides
      hand-optimised AVX2 256-bit and AVX-512 512-bit convolution/matmul kernels.
    - ``OMP_NUM_THREADS``: OpenMP thread count for oneDNN parallel loops.
    - ``TF_NUM_INTRAOP_THREADS``: Parallelism within one TF op (should be all cores).
    - ``TF_NUM_INTEROP_THREADS=1``: Serial scheduling of TF ops to avoid cache thrash.
    - ``TF_CPP_MIN_LOG_LEVEL=2``: Suppress oneDNN info banners from C++ layer.

    Args:
        profile: Hardware profile to record applied settings into.
    """
    cores = str(profile.cpu_cores)

    # Must be set before importing TF for oneDNN to pick them up
    env_settings: dict[str, str] = {
        "TF_ENABLE_ONEDNN_OPTS": "1",  # Intel oneDNN = AVX2 kernels
        "OMP_NUM_THREADS": cores,  # OpenMP threads for oneDNN
        "TF_NUM_INTRAOP_THREADS": cores,  # Threads inside one TF op
        "TF_NUM_INTEROP_THREADS": "1",  # Serialise op scheduling
        "TF_CPP_MIN_LOG_LEVEL": "2",  # Suppress C++ info logs
    }

    for key, value in env_settings.items():
        # setdefault: only set if not already overridden by the user's environment
        if key not in os.environ:
            os.environ[key] = value
            profile.tf_config_applied.append(f"{key}={value}")
        else:
            logger.debug("Env var %s already set to '%s', not overriding.", key, os.environ[key])

    # Additionally configure TF threading programmatically (works even if TF is loaded)
    try:
        import tensorflow as tf

        # Hide any GPUs even if the gpu-feature TF is installed (insurance)
        tf.config.set_visible_devices([], "GPU")

        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(profile.cpu_cores)

        profile.tf_config_applied.append(f"threading: intraop={profile.cpu_cores}, interop=1")

    except Exception as exc:
        logger.debug("Programmatic TF threading config failed (may be set via env): %s", exc)

    profile.selected_device = "cpu"
    avx2_note = "with AVX2 (256-bit SIMD)" if profile.has_avx2 else "without AVX2 (consider upgrading CPU)"
    logger.info(
        "TF CPU-AVX2 configured: %d cores %s | oneDNN: enabled.",
        profile.cpu_cores,
        avx2_note,
    )


def configure_tensorflow(min_vram_mib: int = DEFAULT_MIN_VRAM_MIB) -> HardwareProfile:
    """Detect hardware and configure TensorFlow for optimal performance.

    This is the single entry point that should be called once at application startup
    (or at the beginning of any function that imports TensorFlow). It is safe to call
    multiple times — subsequent calls return the cached profile without reconfiguring.

    Decision logic:
        1. Detect CUDA availability and free VRAM.
        2. If CUDA is available AND free VRAM >= ``min_vram_mib`` → GPU path.
        3. Otherwise → CPU path with AVX2 + oneDNN configuration.

    Args:
        min_vram_mib: Minimum free VRAM in MiB required to choose the GPU path.
            Default: 2048 MiB (2 GB). Increase for larger models or batch sizes.

    Returns:
        ``HardwareProfile`` with ``selected_device`` set to ``"gpu"`` or ``"cpu"``.
    """
    # Fast path: return cached result if already configured
    if configure_tensorflow._cached_profile is not None:  # type: ignore[attr-defined]
        return configure_tensorflow._cached_profile  # type: ignore[attr-defined]

    profile = detect_hardware()

    if profile.cuda_available and profile.free_vram_mib >= min_vram_mib:
        profile.reason = (
            f"GPU selected: {profile.gpu_name} with {profile.free_vram_mib} MiB free (>= threshold {min_vram_mib} MiB)."
        )
        _apply_gpu_config(profile)
    else:
        if profile.cuda_available:
            profile.reason = (
                f"CPU selected: GPU '{profile.gpu_name}' has only {profile.free_vram_mib} MiB free "
                f"(< threshold {min_vram_mib} MiB). Using AVX2 CPU path."
            )
        else:
            profile.reason = "CPU selected: No CUDA-capable GPU detected. Using AVX2 256-bit SIMD + oneDNN CPU path."
        _apply_cpu_avx2_config(profile)

    logger.info("Device decision: %s", profile.reason)

    configure_tensorflow._cached_profile = profile  # type: ignore[attr-defined]
    return profile


# Initialise the cache slot on the function object (simple module-level singleton)
configure_tensorflow._cached_profile = None  # type: ignore[attr-defined]


def get_device_summary(profile: HardwareProfile | None = None) -> str:
    """Return a human-readable summary of the TF device configuration.

    Useful for displaying in Streamlit UIs or log headers.

    Args:
        profile: An already-computed ``HardwareProfile``. If None, runs detection
            and configuration first.

    Returns:
        Multi-line string summarising hardware and device selection.
    """
    if profile is None:
        profile = configure_tensorflow()

    lines = [
        "=" * 60,
        "TensorFlow Device Configuration",
        "=" * 60,
        f"  CUDA available  : {profile.cuda_available}",
    ]

    if profile.cuda_available:
        lines += [
            f"  GPU             : {profile.gpu_name}",
            f"  VRAM free/total : {profile.free_vram_mib} / {profile.total_vram_mib} MiB",
        ]

    lines += [
        f"  CPU cores       : {profile.cpu_cores}",
        f"  AVX2 (256-bit)  : {profile.has_avx2}",
        f"  Selected device : {profile.selected_device.upper()}",
        f"  Reason          : {profile.reason}",
        "  Applied settings:",
    ]

    for setting in profile.tf_config_applied:
        lines.append(f"    • {setting}")

    lines.append("=" * 60)
    return "\n".join(lines)
