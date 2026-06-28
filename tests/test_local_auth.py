import json

from core.auth_middleware import ensure_local_api_key, validate_api_key


def test_local_key_is_created_reused_and_never_stored_in_credentials(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HARADIBOTS_CACHE_ROOT", str(tmp_path / "cache"))
    credentials = tmp_path / "credentials.json"

    first = ensure_local_api_key(credentials)
    second = ensure_local_api_key(credentials)
    identity = validate_api_key(first, credentials)
    stored = credentials.read_text(encoding="utf-8")

    assert first == second
    assert identity["subject"] == "trusted-local-interface"
    assert first not in stored
    assert len(json.loads(stored)["credentials"]) == 1
    assert (tmp_path / "cache" / "auth" / "local.key").exists()
