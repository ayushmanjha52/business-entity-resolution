"""Logging, timing and memory helpers shared by all stages."""
import os
import sys
import threading
import time
from contextlib import contextmanager

import psutil

_T0 = time.perf_counter()
PEAK = {"rss_gb": 0.0}
TIMINGS: dict[str, float] = {}


def rss_gb() -> float:
    gb = psutil.Process(os.getpid()).memory_info().rss / 1e9
    PEAK["rss_gb"] = max(PEAK["rss_gb"], gb)
    return gb


def log(msg: str):
    print(f"[{time.perf_counter() - _T0:7.1f}s | {rss_gb():5.1f} GB] {msg}", flush=True)


@contextmanager
def stage(name: str):
    """Log a stage and record its wall time into TIMINGS (used by the benchmark report)."""
    log(f"▶ {name}")
    t = time.perf_counter()
    yield
    TIMINGS[name] = TIMINGS.get(name, 0.0) + time.perf_counter() - t
    log(f"✔ {name} ({TIMINGS[name]:.1f}s)")


def _sampler():   # true peak RSS: sample twice a second in a daemon thread
    while True:
        rss_gb()
        time.sleep(0.5)


threading.Thread(target=_sampler, daemon=True).start()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
