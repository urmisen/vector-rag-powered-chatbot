#!/usr/bin/env python3
"""
Warm up RAG resources (FAISS, embeddings, Vertex AI models) ahead of live traffic.

This script can be scheduled to run on deploy or periodically to keep warm pools
active and avoid cold-start penalties.
"""

import argparse
import sys
import time
from pathlib import Path


def _get_project_root() -> Path:
    """Return the project root directory (where pyproject.toml lives)."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: two levels up from scripts/ subpackages
    return current.parents[2]


PROJECT_ROOT = _get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.core import RAGManager  # noqa: E402
from app.infra.background_client import get_background_client_manager  # noqa: E402


def run_warmup_cycle(cycle_index: int) -> float:
    """Run a single warm-up cycle and return the elapsed seconds."""
    manager = RAGManager()
    manager.logger.info(f"[WarmPool] Warm-up cycle {cycle_index + 1} starting")
    start = time.perf_counter()
    manager._warmup_resources()
    elapsed = time.perf_counter() - start
    manager.logger.info(f"[WarmPool] Warm-up cycle {cycle_index + 1} finished in {elapsed:.2f}s")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Pre-warm RAG resources to minimize latency spikes.")
    parser.add_argument("--repeat", type=int, default=1, help="Number of warm-up cycles to execute (default: 1)")
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Seconds to wait between cycles when repeat > 1 (default: 300)"
    )
    parser.add_argument(
        "--bootstrap-client",
        action="store_true",
        help="Also start the background MCP client to pre-initialize the backend worker.",
    )
    parser.add_argument(
        "--client-email",
        type=str,
        default=None,
        help="Email context to use when pre-bootstrapping the background MCP client.",
    )
    parser.add_argument(
        "--client-timeout",
        type=int,
        default=180,
        help="Seconds to wait for the background MCP client to finish bootstrapping.",
    )
    args = parser.parse_args()

    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")
    if args.interval < 0:
        raise ValueError("--interval must be >= 0")

    total_elapsed = 0.0
    for cycle in range(args.repeat):
        total_elapsed += run_warmup_cycle(cycle)
        if cycle < args.repeat - 1 and args.interval:
            time.sleep(args.interval)

    avg_elapsed = total_elapsed / args.repeat
    print(f"[WarmPool] Completed {args.repeat} warm-up cycle(s); average latency {avg_elapsed:.2f}s")

    if args.bootstrap_client:
        manager = get_background_client_manager()
        manager.ensure_prewarmed()
        bridge = manager.get_or_create(args.client_email)
        print("[WarmPool] Bootstrapping background MCP client...")
        bridge.wait_until_ready(timeout=args.client_timeout)
        identity = bridge.get_authenticated_user() or bridge.get_gcp_username()
        print(f"[WarmPool] Background MCP client ready (identity: {identity})")
        manager.shutdown_all()


if __name__ == "__main__":
    main()
