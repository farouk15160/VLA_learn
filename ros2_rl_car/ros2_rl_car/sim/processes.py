"""Own and guard the Gazebo Classic process lifecycle."""

from __future__ import annotations

import os
import signal
import subprocess
import time


class GazeboAlreadyRunning(RuntimeError):
    pass


def gzserver_pids() -> tuple[int, ...]:
    result = subprocess.run(
        ["pgrep", "-u", str(os.getuid()), "-x", "gzserver"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"pgrep failed: {result.stderr.strip()}")
    return tuple(int(line) for line in result.stdout.splitlines() if line.isdigit())


def refuse_if_running() -> None:
    pids = gzserver_pids()
    if pids:
        raise GazeboAlreadyRunning(
            f"gzserver already running (PID(s) {pids}); run `rl-car stop` first"
        )


def stop_gzservers(timeout: float = 5.0) -> tuple[int, ...]:
    """SIGINT current-user Gazebo servers, with SIGTERM as a fallback."""
    stopped = gzserver_pids()
    pids = stopped
    for pid in stopped:
        # Gazebo Classic installs a SIGINT handler for orderly shutdown; in
        # practice some long-running servers ignore SIGTERM.
        os.kill(pid, signal.SIGINT)
    deadline = time.monotonic() + timeout
    while pids and time.monotonic() < deadline:
        alive = []
        for pid in pids:
            try:
                os.kill(pid, 0)
                alive.append(pid)
            except ProcessLookupError:
                pass
        pids = tuple(alive)
        if pids:
            time.sleep(0.1)
    # A process can exit between the last kill(0) probe and the deadline.
    # Re-resolve exact-name processes before declaring stop failure.
    pids = tuple(pid for pid in stopped if pid in set(gzserver_pids()))
    for pid in pids:
        os.kill(pid, signal.SIGTERM)
    if pids:
        time.sleep(0.5)
    pids = tuple(pid for pid in stopped if pid in set(gzserver_pids()))
    if pids:
        raise RuntimeError(f"gzserver PID(s) {pids} did not stop after {timeout}s")
    return stopped
