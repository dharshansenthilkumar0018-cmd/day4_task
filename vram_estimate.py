"""
Day 4 - VRAM / memory estimator and experiments
Scenario: local open-model assistant on a laptop with a 16 GB memory budget.

The estimator is intentionally approximate. It estimates:
    weights + KV cache + runtime overhead
and returns a fit verdict.

Usage:
    python vram_estimate.py
"""

from dataclasses import dataclass


BYTES_PER_PARAM = {
    "FP16": 2.0,
    "Q8_0": 1.0,
    "Q5_K_M": 0.68,
    "Q4_K_M": 0.57,
    "Q3_K_M": 0.43,
}

AVAILABLE_GB = 16.0
OVERHEAD_GB = 0.20
KV_GB_PER_B_PER_K = 0.01


@dataclass
class Estimate:
    params_b: float
    precision: str
    context_k: int
    weights_gb: float
    kv_gb: float
    total_gb: float
    fits: bool


def estimate_memory(params_b: float, precision: str, context_k: int,
                    available_gb: float = AVAILABLE_GB) -> Estimate:
    """Return an approximate memory estimate and fit verdict."""
    if precision not in BYTES_PER_PARAM:
        raise ValueError(f"Unknown precision: {precision}")

    weights = params_b * BYTES_PER_PARAM[precision]
    kv_cache = params_b * context_k * KV_GB_PER_B_PER_K
    total = weights + kv_cache + OVERHEAD_GB
    fits = total <= available_gb * 0.70

    return Estimate(
        params_b=params_b,
        precision=precision,
        context_k=context_k,
        weights_gb=weights,
        kv_gb=kv_cache,
        total_gb=total,
        fits=fits,
    )


def print_estimate(label: str, e: Estimate) -> None:
    verdict = "fits comfortably" if e.fits else "does NOT fit"
    print(
        f"{label:<20} {e.precision:<8} {e.params_b:>4.1f}B ctx "
        f"{e.context_k:>3}K  weights {e.weights_gb:>5.2f} GB  "
        f"kv {e.kv_gb:>5.2f} GB  total {e.total_gb:>5.2f} GB -> {verdict}"
    )


def main():
    print(f"Available memory budget: {AVAILABLE_GB:.1f} GB")
    print("\nEstimate table:")

    configs = [
        ("Qwen small", 1.5, "Q4_K_M", 8),
        ("Granite / Qwen mid", 8.0, "Q4_K_M", 8),
        ("Mid at FP16", 8.0, "FP16", 8),
        ("Large local", 30.0, "Q4_K_M", 32),
        ("Server class", 70.0, "Q4_K_M", 32),
    ]

    for label, params, precision, ctx in configs:
        print_estimate(label, estimate_memory(params, precision, ctx))

    print("\nSame 8B model, different context lengths:")
    for ctx in (4, 8, 32, 128):
        print_estimate("8B agent", estimate_memory(8.0, "Q4_K_M", ctx))

    print("\nSame 8B model, different quantizations at 8K:")
    for q in ("Q3_K_M", "Q4_K_M", "Q5_K_M", "Q8_0", "FP16"):
        print_estimate("8B agent", estimate_memory(8.0, q, 8))


if __name__ == "__main__":
    main()
