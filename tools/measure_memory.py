import os
import sys
import time
import shlex
import subprocess


def get_rss_kb(pid: int) -> int | None:
    """Return current RSS in kB for given PID, or None if unavailable."""
    status_path = f"/proc/{pid}/status"
    try:
        with open(status_path, "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    # Format: VmRSS:   <value> kB
                    if len(parts) >= 2 and parts[1].isdigit():
                        return int(parts[1])
    except FileNotFoundError:
        return None
    return None


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python measure_memory.py \"<command with args>\"")
        sys.exit(1)

    cmd_str = sys.argv[1]
    cmd = shlex.split(cmd_str)

    print(f"[measure_memory] Running: {cmd_str}")
    proc = subprocess.Popen(cmd)
    pid = proc.pid
    max_rss_kb = 0

    try:
        while True:
            ret = proc.poll()
            rss = get_rss_kb(pid)
            if rss is not None and rss > max_rss_kb:
                max_rss_kb = rss
            if ret is not None:
                break
            time.sleep(0.5)
    finally:
        # Ensure process is terminated if still alive
        if proc.poll() is None:
            proc.terminate()

    print(f"[measure_memory] Max RSS: {max_rss_kb} kB ({max_rss_kb / 1024:.2f} MB)")
    sys.exit(proc.returncode if proc.returncode is not None else 0)


if __name__ == "__main__":
    main()
