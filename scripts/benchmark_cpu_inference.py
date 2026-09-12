"""Benchmark CPU inference latency and accuracy for SignalScope."""

import time
from PIL import Image
from app.inference.engine import SignalScopeInferenceEngine


def run_cpu_benchmark():
    print("Initializing SignalScopeInferenceEngine on CPU...")
    t_start = time.perf_counter()
    engine = SignalScopeInferenceEngine(device="cpu")
    t_load = (time.perf_counter() - t_start) * 1000

    img = Image.open("frontend/public/samples/authentic_real.jpg")

    # Warmup
    _ = engine.predict_calibrated_probability(img)

    # Detailed latency breakdown on CPU
    t0 = time.perf_counter()
    p = engine.predict_calibrated_probability(img)
    t_forward = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    res = engine.analyze(img)
    t_total = (time.perf_counter() - t0) * 1000

    print("==================================================")
    print(" CPU INFERENCE BENCHMARK RESULTS")
    print("==================================================")
    print(f"Device:                   {engine.device}")
    print(f"Cold Start / Load:        {t_load:.2f} ms")
    print(f"Single Forward Pass:      {t_forward:.2f} ms")
    print(f"Total Analyze (XAI+Stab): {t_total:.2f} ms")
    print(f"Verdict:                  {res.verdict}")
    print(f"Calibrated Probability:   {res.probability:.4f}")
    print(f"Authenticity Stability:   {res.evidence['robustness']['stability_score']:.4f}")
    print("==================================================")


if __name__ == "__main__":
    run_cpu_benchmark()
