from __future__ import annotations


def lease_renew_interval(ttl_sec: float) -> float:
    ttl = max(1.0, float(ttl_sec))
    return max(1.0, min(ttl * 0.5, 30.0))
