"""SLURM script generation and subprocess-isolated scheduler calls."""

from __future__ import annotations

import json
import base64
import os
import re
import subprocess
from pathlib import Path
from typing import Any


JOB_ID_PATTERN = re.compile(r"Submitted batch job (\d+)")


def generate_batch_script(strategy_config: dict[str, Any]) -> str:
    gpus = max(int(strategy_config.get("tp_degree", 1)), 1)
    cpus = max(int(strategy_config.get("cpu_threads", 4)), 1)
    payload = base64.b64encode(
        json.dumps(strategy_config, separators=(",", ":")).encode()
    ).decode()
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "#SBATCH --job-name=haradibots",
            f"#SBATCH --gres=gpu:{gpus}",
            f"#SBATCH --cpus-per-task={cpus}",
            "#SBATCH --output=haradibots-%j.log",
            f"export HARADIBOTS_STRATEGY_B64='{payload}'",
            'exec python -m cluster.worker_entry --scheduler slurm --job-id "$SLURM_JOB_ID"',
            "",
        ]
    )


def submit(script_path: str | Path) -> str:
    path = Path(script_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"SLURM script does not exist: {path}")
    result = subprocess.run(
        ["sbatch", str(path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed: {result.stderr.strip()}")
    match = JOB_ID_PATTERN.search(result.stdout)
    if not match:
        raise RuntimeError("sbatch output contains no job ID")
    return match.group(1)


def poll_status(job_id: str) -> str:
    if not job_id.isdigit():
        raise ValueError("SLURM job_id must be numeric")
    result = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%T"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
        env=os.environ.copy(),
    )
    if result.returncode != 0:
        raise RuntimeError(f"squeue failed: {result.stderr.strip()}")
    return result.stdout.strip() or "COMPLETED_OR_UNKNOWN"
