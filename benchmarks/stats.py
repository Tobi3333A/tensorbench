import statistics
import numpy as np

def summarize_latencies(times):
    return {
        "mean_ms": statistics.mean(times),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "min_ms": min(times),
        "max_ms": max(times),
    }

def build_result(backend, device, input_shape, times):
    stats = summarize_latencies(times)

    return {
        "backend": backend,
        "device": device,
        "input_shape": list(input_shape),
        "iterations": len(times),
        **stats,
    }

def compare_outputs(reference, candidate):
    if isinstance(reference, (tuple, list)):
        reference = reference[0]

    if isinstance(candidate, (tuple, list)):
        candidate = candidate[0]

    if hasattr(reference, "detach"):
        reference = reference.detach().cpu().numpy()

    if hasattr(candidate, "detach"):
        candidate = candidate.detach().cpu().numpy()

    reference = np.asarray(reference)
    candidate = np.asarray(candidate)

    return {
        "max_absolute_error": float(np.max(np.abs(reference - candidate))),
        "mean_absolute_error": float(np.mean(np.abs(reference - candidate))),
    }
