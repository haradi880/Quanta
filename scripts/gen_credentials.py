"""Generate one local development API key without printing the plaintext."""

from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CREDENTIALS_PATH = REPO_ROOT / "config" / "credentials.json"
ENV_PATH = REPO_ROOT / ".env"


def _read_store() -> dict[str, list[dict[str, object]]]:
    if not CREDENTIALS_PATH.exists():
        return {"credentials": []}
    store = json.loads(CREDENTIALS_PATH.read_text(encoding="utf-8"))
    if not isinstance(store, dict) or not isinstance(store.get("credentials"), list):
        raise ValueError("config/credentials.json must contain a credentials array")
    return store


def _write_private_file(path: Path, content: str) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


def main() -> int:
    if ENV_PATH.exists() and any(
        line.startswith("DEV_API_KEY=")
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines()
    ):
        raise RuntimeError(
            ".env already contains DEV_API_KEY; remove it explicitly before rotating"
        )

    api_key = f"hb_dev_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    credential_id = f"dev-{secrets.token_hex(8)}"

    store = _read_store()
    store["credentials"].append(
        {
            "id": credential_id,
            "subject": "local-development",
            "key_hash": key_hash,
            "revoked": False,
            "scopes": ["jobs:run", "jobs:read", "jobs:purge"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    _write_private_file(
        CREDENTIALS_PATH,
        json.dumps(store, indent=2, sort_keys=True) + "\n",
    )

    existing_env = ENV_PATH.read_text(encoding="utf-8") if ENV_PATH.exists() else ""
    separator = "" if not existing_env or existing_env.endswith("\n") else "\n"
    _write_private_file(
        ENV_PATH,
        f"{existing_env}{separator}DEV_API_KEY={api_key}\n",
    )

    print(
        "Development credential created in .env and hashed in "
        "config/credentials.json; plaintext was not printed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
