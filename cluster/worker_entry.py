"""Minimal scheduler worker bootstrap; execution remains Orchestrator-owned."""

from __future__ import annotations

import argparse
import base64
import json
import os


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", choices=("slurm", "k8s", "ray"), required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args(argv)
    strategy = json.loads(
        base64.b64decode(os.environ["HARADIBOTS_STRATEGY_B64"]).decode()
    )
    print(
        json.dumps(
            {
                "job_id": args.job_id,
                "scheduler": args.scheduler,
                "status": "ready",
                "strategy": strategy,
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
