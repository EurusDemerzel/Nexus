#!/usr/bin/env python3
"""
fluctuating_stress.py
在 Linux 服务器上模拟真实端侧设备的资源波动（CPU / 内存 / RTT）。
用法:
  sudo python fluctuating_stress.py
Ctrl+C 自动清理所有子进程和网络延迟设置。
"""

import os
import random
import signal
import subprocess
import sys
import time
from datetime import datetime

# ── 配置 ──
CPU_LOW_CORES = 1
CPU_MED_CORES = 3
CPU_HIGH_CORES = 4
CPU_SWITCH_INTERVAL_MIN = 3
CPU_SWITCH_INTERVAL_MAX = 8
CPU_TIMEOUT = 10

MEM_ALLOC_MIN_MB = 100
MEM_ALLOC_MAX_MB = 2048
MEM_HOLD_MIN = 2
MEM_HOLD_MAX = 5
MEM_INTERVAL_MIN = 5
MEM_INTERVAL_MAX = 15

RTT_MIN_MS = 80
RTT_MAX_MS = 300
RTT_INTERVAL_MIN = 2
RTT_INTERVAL_MAX = 6

_CPU_PROC: subprocess.Popen | None = None
_MEM_BLOCK: bytearray | None = None
_shutdown = False


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ── CPU 波动 ──
def _set_cpu_load():
    global _CPU_PROC
    if _CPU_PROC is not None:
        try:
            _CPU_PROC.kill()
            _CPU_PROC.wait(timeout=3)
        except Exception:
            pass

    level = random.choice(["low", "medium", "high"])
    if level == "low":
        cores = CPU_LOW_CORES
    elif level == "medium":
        cores = CPU_MED_CORES
    else:
        cores = CPU_HIGH_CORES

    _CPU_PROC = subprocess.Popen(
        ["stress", "--cpu", str(cores), "--timeout", f"{CPU_TIMEOUT}s"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    log(f"CPU: {level} ({cores} core(s)), PID={_CPU_PROC.pid}")


# ── 内存波动 ──
def _memory_spike():
    global _MEM_BLOCK
    if _MEM_BLOCK is not None:
        _MEM_BLOCK = None  # let GC free it

    size_mb = random.randint(MEM_ALLOC_MIN_MB, MEM_ALLOC_MAX_MB)
    hold_s = random.randint(MEM_HOLD_MIN, MEM_HOLD_MAX)
    try:
        _MEM_BLOCK = bytearray(size_mb * 1024 * 1024)
        log(f"MEM: allocated {size_mb} MB, holding {hold_s}s")
        time.sleep(hold_s)
    except MemoryError:
        log(f"MEM: allocation of {size_mb} MB failed (OOM)")
    finally:
        _MEM_BLOCK = None


# ── RTT 波动 ──
def _set_rtt():
    delay_ms = random.randint(RTT_MIN_MS, RTT_MAX_MS)
    cmd = ["sudo", "tc", "qdisc", "replace", "dev", "lo", "root", "netem", "delay", f"{delay_ms}ms"]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=True)
        log(f"RTT: {delay_ms} ms on lo")
    except subprocess.CalledProcessError as e:
        log(f"RTT: FAILED (stderr={e.stderr.strip() if e.stderr else 'unknown'})")
    except FileNotFoundError:
        log("RTT: tc not found, skipping")


# ── 清理 ──
def _cleanup():
    global _CPU_PROC, _MEM_BLOCK
    log("Cleaning up ...")
    if _CPU_PROC is not None:
        _CPU_PROC.kill()
        _CPU_PROC = None
    _MEM_BLOCK = None
    try:
        subprocess.run(
            ["sudo", "tc", "qdisc", "del", "dev", "lo", "root"],
            capture_output=True, timeout=5,
        )
        log("RTT: removed lo delay")
    except Exception:
        pass
    log("Done.")


def _signal_handler(sig, frame):
    global _shutdown
    log(f"Received signal {sig}, shutting down ...")
    _shutdown = True


def main():
    if os.geteuid() != 0:
        log("WARNING: not running as root. RTT setting may fail. Use: sudo python fluctuating_stress.py")

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # 初始 RTT
    _set_rtt()

    last_cpu = time.time()
    last_mem = time.time()
    last_rtt = time.time()

    while not _shutdown:
        now = time.time()
        if now - last_cpu >= random.randint(CPU_SWITCH_INTERVAL_MIN, CPU_SWITCH_INTERVAL_MAX):
            _set_cpu_load()
            last_cpu = now
        # 内存 spike 在独立线程中（非阻塞），这里简单串行
        try:
            import threading
            _t = threading.Thread(target=_memory_spike, daemon=True)
            _t.start()
        except Exception:
            pass
        if now - last_rtt >= random.randint(RTT_INTERVAL_MIN, RTT_INTERVAL_MAX):
            _set_rtt()
            last_rtt = now
        time.sleep(1)

    _cleanup()


if __name__ == "__main__":
    main()
