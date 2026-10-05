import statistics


def summarize_latencies(times):
    ordered = sorted(times)

    return {
        "mean_ms": statistics.mean(times),
        "p50_ms": ordered[int(len(ordered) * 0.50)],
        "p95_ms": ordered[int(len(ordered) * 0.95)],
        "p99_ms": ordered[int(len(ordered) * 0.99)],
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
