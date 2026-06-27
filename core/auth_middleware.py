"""Authentication boundary for every inbound HaradiBots job envelope."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jwt

from core.schemas import ErrorEnvelope, JobEnvelope, SCHEMA_VERSION


DEFAULT_CREDENTIALS_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "credentials.json"
)
JWT_AUDIENCE = "haradibots"
JWT_ALGORITHMS = ("HS256",)
request_identity: ContextVar[dict[str, Any] | None] = ContextVar(
    "request_identity",
    default=None,
)


class AuthError(Exception):
    """Raised when supplied credentials cannot be authenticated."""


def load_credentials(path: Path | str = DEFAULT_CREDENTIALS_PATH) -> list[dict[str, Any]]:
    """Load and validate the hashed API-key credential store."""

    credential_path = Path(path)
    try:
        raw_store = json.loads(credential_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuthError("credential store is unavailable or invalid") from exc

    if not isinstance(raw_store, dict):
        raise AuthError("credential store root must be an object")

    credentials = raw_store.get("credentials")
    if not isinstance(credentials, list):
        raise AuthError("credential store must contain a credentials array")

    for record in credentials:
        if not isinstance(record, dict):
            raise AuthError("credential record must be an object")
        key_hash = record.get("key_hash")
        if not isinstance(key_hash, str) or len(key_hash) != 64:
            raise AuthError("credential record contains an invalid SHA-256 hash")
        try:
            int(key_hash, 16)
        except ValueError as exc:
            raise AuthError("credential record contains an invalid SHA-256 hash") from exc
        if "api_key" in record or "plaintext" in record:
            raise AuthError("credential store must never contain plaintext API keys")

    return credentials


def validate_api_key(
    api_key: str,
    path: Path | str = DEFAULT_CREDENTIALS_PATH,
) -> dict[str, Any]:
    """Return the matching active identity, or raise ``AuthError``."""

    if not api_key:
        raise AuthError("API key is missing")

    supplied_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    for record in load_credentials(path):
        if not hmac.compare_digest(record["key_hash"], supplied_hash):
            continue
        if record.get("revoked", False):
            raise AuthError("API key has been revoked")
        return {
            "credential_id": record.get("id"),
            "subject": record.get("subject", record.get("id")),
            "scopes": record.get("scopes", []),
            "auth_type": "api_key",
        }

    raise AuthError("invalid API key")


def validate_jwt(token: str) -> dict[str, Any]:
    """Verify JWT signature, expiry, audience, and return caller identity."""

    if not token:
        raise AuthError("JWT is missing")

    secret = os.environ.get("HARADIBOTS_JWT_SECRET")
    if not secret:
        raise AuthError("JWT signing secret is not configured")
    if len(secret.encode("utf-8")) < 32:
        raise AuthError("JWT signing secret must be at least 32 bytes")

    try:
        claims = jwt.decode(
            token,
            secret,
            algorithms=list(JWT_ALGORITHMS),
            audience=JWT_AUDIENCE,
            options={"require": ["exp", "aud", "sub"]},
        )
    except jwt.PyJWTError as exc:
        raise AuthError("invalid JWT") from exc

    scopes = claims.get("scope", claims.get("scopes", []))
    if isinstance(scopes, str):
        scopes = scopes.split()
    if not isinstance(scopes, list) or not all(
        isinstance(scope, str) for scope in scopes
    ):
        raise AuthError("JWT scope claim is invalid")

    return {
        "subject": claims["sub"],
        "scopes": scopes,
        "auth_type": "jwt",
        "claims": claims,
    }


def _authentication_error(envelope: JobEnvelope, message: str) -> ErrorEnvelope:
    return ErrorEnvelope(
        schema_version=SCHEMA_VERSION,
        job_id=envelope.job_id,
        code=401,
        error="authentication_failed",
        message=message,
        timestamp_utc=datetime.now(timezone.utc),
    )


def authenticate(envelope: JobEnvelope) -> dict[str, Any] | ErrorEnvelope:
    """Authenticate a version 3.0 envelope before it enters the state machine."""

    request_identity.set(None)

    if envelope.schema_version != SCHEMA_VERSION:
        return _authentication_error(envelope, "unsupported schema version")
    if envelope.auth is None:
        return _authentication_error(envelope, "authentication block is required")

    try:
        if envelope.auth.api_key is not None:
            identity = validate_api_key(envelope.auth.api_key)
        elif envelope.auth.jwt_token is not None:
            identity = validate_jwt(envelope.auth.jwt_token)
        else:
            return _authentication_error(envelope, "credential is required")
    except AuthError as exc:
        return _authentication_error(envelope, str(exc))

    request_identity.set(identity)
    return identity
