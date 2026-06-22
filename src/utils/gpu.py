"""GPU memory management utilities.

The tuning loop creates and discards many models (one per Optuna trial × inner
fold). PyTorch's caching allocator holds freed GPU memory unless explicitly
released, so without cleanup the memory of every fitted model accumulates and
eventually exhausts the GPU — even though only one model is alive at a time.
`free_gpu_memory()` forces collection of the discarded model objects and returns
their memory to the device. Call it after each model is done being used.
"""

import gc


def free_gpu_memory() -> None:
    """Force Python GC and release cached GPU memory back to the device.

    Safe to call when CUDA is unavailable (the torch import / empty_cache is
    guarded), so it is a no-op for CPU-only (e.g. GBDT) runs.
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
