"""
Resilient CCT Test Runner
=========================
Auto-restarts the CCT test after GPU crashes (SIGILL, SIGSEGV, OOM).
The CCT test has built-in checkpointing, so each restart picks up where
it left off. This wrapper just automates the restart loop.

Usage:
    python -m test.run_cct_resilient
"""

import subprocess
import sys
import time
import os
import json

MAX_RESTARTS = 200   # safety limit
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
CHECKPOINT_FILE = os.path.join(RESULTS_DIR, "cct_checkpoint.json")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "cct_test_results.json")


def get_progress():
    """Read checkpoint to get current progress."""
    try:
        with open(CHECKPOINT_FILE, "r") as f:
            data = json.load(f)
        return len(data.get("completed_keys", [])), len(data.get("results", []))
    except (FileNotFoundError, json.JSONDecodeError):
        return 0, 0


def is_complete():
    """Check if the test completed (checkpoint removed, output exists)."""
    return not os.path.exists(CHECKPOINT_FILE) and os.path.exists(OUTPUT_FILE)


def main():
    print("=" * 60)
    print("  Resilient CCT Test Runner")
    print("  Auto-restarts on crash until completion")
    print("=" * 60)

    for attempt in range(1, MAX_RESTARTS + 1):
        keys_before, results_before = get_progress()
        print(f"\n  Attempt {attempt}/{MAX_RESTARTS} "
              f"(checkpoint: {keys_before} keys, {results_before} results)")

        # Run the CCT test as a subprocess
        result = subprocess.run(
            [sys.executable, "-m", "test.cct_test"],
            cwd=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
        )

        if result.returncode == 0:
            print("\n  CCT test completed successfully!")
            return 0

        keys_after, results_after = get_progress()
        progress = keys_after - keys_before

        print(f"\n  Process exited with code {result.returncode}")
        print(f"  Progress this run: +{progress} keys "
              f"(total: {keys_after} keys, {results_after} results)")

        if progress == 0:
            # No progress — might be stuck in a loop
            print("  WARNING: No progress made — waiting 10s before retry")
            time.sleep(10)

        if is_complete():
            print("\n  CCT test completed!")
            return 0

        # Brief pause to let GPU cool down
        print("  Restarting in 5s...")
        time.sleep(5)

    print(f"\n  Reached max restarts ({MAX_RESTARTS})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
