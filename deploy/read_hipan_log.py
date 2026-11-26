"""Read a go2_hipan_low binary log (LogRow, packed) into numpy arrays and print a summary.

    python3 read_hipan_log.py logs/hipan_low_<time>.bin
"""
import sys

import numpy as np

STATES = ["WAIT", "RAMP", "HOLD", "POLICY", "DAMP", "DOWN"]
ROW = np.dtype([("t", "<f8"), ("state", "u1"), ("loop_dt", "<f4"), ("state_age", "<f4"), ("obs", "<f4", 62),
                ("action", "<f4", 12), ("target", "<f4", 12), ("q", "<f4", 12), ("dq", "<f4", 12), ("est", "<f4", 5),
                ("foot_raw", "<f4", 4), ("cmd", "<f4", 5), ("rpy", "<f4", 3), ("tau_est", "<f4", 12)])


def load(path):
    return np.fromfile(path, dtype=ROW)


if __name__ == "__main__":
    d = load(sys.argv[1])
    dt = d["loop_dt"][1:] * 1e3
    print("rows %d  %.1f s" % (len(d), d["t"][-1] - d["t"][0]))
    print("loop dt ms: p50 %.2f p99 %.2f max %.2f   state age ms: p99 %.2f max %.2f" % (
        np.median(dt), np.percentile(dt, 99), dt.max(), np.percentile(d["state_age"] * 1e3, 99),
        d["state_age"].max() * 1e3))
    for s in np.unique(d["state"]):
        m = d["state"] == s
        print("  %-6s %5d rows" % (STATES[s], m.sum()))
    print("foot_raw (FL FR RL RR): min", d["foot_raw"].min(0), "max", d["foot_raw"].max(0))
    print("cmd min", d["cmd"].min(0).round(2), "max", d["cmd"].max(0).round(2))
    m = d["state"] == 3
    if m.any():
        print("POLICY |target-q| mean %.3f max %.3f rad   est mean" % (
            np.abs(d["target"][m] - d["q"][m]).mean(), np.abs(d["target"][m] - d["q"][m]).max()),
            d["est"][m].mean(0).round(3))
