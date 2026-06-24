"""GPU memory management utilities.

The tuning loop creates and discards many models (one per Optuna trial × inner
fold). PyTorch's caching allocator holds freed GPU memory unless explicitly
released, so without cleanup the memory of every fitted model accumulates and
eventually exhausts the GPU — even though only one model is alive at a time.
`free_gpu_memory()` forces collection of the discarded model objects and returns
their memory to the device. Call it after each model is done being used.
"""

import gc
import logging

logger = logging.getLogger("tabular_benchmark")


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


def is_oom_error(exc: Exception) -> bool:
    """True if the exception is a CUDA out-of-memory / allocator error.

    Covers torch.cuda.OutOfMemoryError, torch.AcceleratorError ("CUDA error:
    out of memory"), and the downstream "unknown error" a prior OOM can trigger.
    """
    name = type(exc).__name__
    if name in ("OutOfMemoryError", "AcceleratorError"):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error" in msg


def create_fit_with_retry(
    model_name, task_type, n_classes, seed, model_kwargs,
    X_train, y_train, X_val=None, y_val=None,
    min_batch_size=32,
):
    """Create and fit a model, halving batch_size on GPU OOM until it fits.

    A single oversized config on a high-token input can exceed GPU memory in
    one forward/backward pass. Rather than fail the whole experiment, retry the
    same model (identical hyperparameters) with progressively smaller batches —
    a standard "auto batch size" fallback that only triggers on OOM, so it never
    changes the behaviour of configs that already fit. Returns the fitted model.
    Re-raises the OOM if even the smallest batch does not fit, or any non-OOM error.
    """
    from src.models.factory import create_model

    base_bs = int(model_kwargs.get("batch_size", 256))
    sizes = [base_bs]
    bs = base_bs
    while bs > min_batch_size:
        bs //= 2
        sizes.append(max(bs, min_batch_size))

    last_exc = None
    for i, bs in enumerate(sizes):
        kwargs = dict(model_kwargs)
        if i > 0:
            kwargs["batch_size"] = bs
            logger.warning(f"{model_name}: GPU OOM — retrying fit with batch_size={bs}")
        try:
            model = create_model(model_name, task_type, n_classes, seed=seed, **kwargs)
            model.fit(X_train, y_train, X_val, y_val)
            return model
        except Exception as e:
            if is_oom_error(e) and i < len(sizes) - 1:
                last_exc = e
                model = locals().get("model", None)
                del model
                free_gpu_memory()
                continue
            raise
    raise last_exc
