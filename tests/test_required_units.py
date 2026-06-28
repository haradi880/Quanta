from datetime import datetime, timedelta, timezone
from uuid import uuid4

import jwt
import pytest
from pydantic import ValidationError

from core.auth_middleware import (
    AuthError,
    authenticate,
    ensure_local_api_key,
    load_credentials,
    validate_api_key,
    validate_jwt,
)
from core.orchestrator import JobState, StateMachineError, transition
from core.profiler import (
    calc_kv_cache,
    calc_partial_offload_layers,
    calc_weights_vram,
    get_thread_config,
    select_strategy,
)
from core.schemas import (
    AuthBlock,
    CallbackConfig,
    InterfaceType,
    JobEnvelope,
    JobMode,
    JobOperation,
    ModelSource,
    SystemPrompt,
    ValidationPolicy,
)


def _envelope(auth):
    return JobEnvelope(
        job_id=uuid4(),
        auth=auth,
        interface=InterfaceType.CLI,
        mode=JobMode.AUTO,
        operation=JobOperation.INSPECT,
        source_model=ModelSource(repo_id="owner/model"),
        validation_policy=ValidationPolicy(),
        system_prompt=SystemPrompt(preset_id="default"),
        callbacks=CallbackConfig(),
    )


def test_all_three_vram_formulas_and_validation():
    assert calc_weights_vram(1_000_000_000, 4) == 500_000_000
    assert calc_kv_cache(32, 1, 2048, 8, 128, 2) == 268_435_456
    assert calc_partial_offload_layers(6_000, 1_000, 10_000, 10) == 5
    assert calc_partial_offload_layers(100, 200, 10_000, 10) == 0
    assert calc_partial_offload_layers(100, 0, 0, 10) == 10
    with pytest.raises(ValueError):
        calc_weights_vram(-1, 4)
    with pytest.raises(ValueError):
        calc_kv_cache(1, 1, 1, -1, 1, 1)
    with pytest.raises(ValueError):
        calc_partial_offload_layers(1, 1, 1, 0)


@pytest.mark.parametrize(
    ("hardware", "model", "expected"),
    [
        ({"gpu_count": 0, "architecture": "x86_64"}, {"model_size_b": 7}, "CPU Only (x86)"),
        ({"gpu_count": 0, "architecture": "arm64"}, {"model_size_b": 7}, "CPU Only (ARM)"),
        (
            {"gpu_count": 0, "architecture": "arm64", "platform": "darwin"},
            {"model_size_b": 7},
            "Apple Silicon (M-series)",
        ),
        (
            {"gpu_count": 1, "gpus": [{"vram_free_bytes": 4 * 1024**3}]},
            {"model_size_b": 7, "num_layers": 32},
            "Low VRAM GPU",
        ),
        (
            {"gpu_count": 1, "gpus": [{"vram_free_bytes": 12 * 1024**3}]},
            {"model_size_b": 7},
            "Mid VRAM GPU",
        ),
        (
            {"gpu_count": 1, "gpus": [{"vram_free_bytes": 24 * 1024**3}]},
            {"model_size_b": 13},
            "High VRAM GPU",
        ),
        (
            {
                "gpu_count": 2,
                "gpus": [
                    {"vram_free_bytes": 24 * 1024**3},
                    {"vram_free_bytes": 24 * 1024**3},
                ],
            },
            {"model_size_b": 34},
            "Dual High VRAM",
        ),
        (
            {
                "gpu_count": 3,
                "gpus": [{"vram_free_bytes": 80 * 1024**3}] * 3,
            },
            {"model_size_b": 70},
            "Multi-GPU Cluster",
        ),
    ],
)
def test_every_decision_matrix_tier(hardware, model, expected):
    assert select_strategy(hardware, model)["hardware_tier"] == expected


def test_hardware_thread_pinning_rules():
    hybrid = get_thread_config(
        {
            "physical_cores": 12,
            "core_topology": "hybrid",
            "p_core_ids": [0, 2, 4, 6],
        }
    )
    assert hybrid["thread_count"] == 4
    assert hybrid["core_ids"] == [0, 2, 4, 6]
    uniform = get_thread_config(
        {"physical_cores": 8, "core_topology": "uniform", "p_core_ids": []}
    )
    assert uniform["thread_count"] == 7


def test_api_key_authentication_and_missing_auth(tmp_path):
    import hashlib
    import json

    key = "hb_test_secret"
    path = tmp_path / "credentials.json"
    path.write_text(
        json.dumps(
            {
                "credentials": [
                    {
                        "id": "test",
                        "key_hash": hashlib.sha256(key.encode()).hexdigest(),
                        "revoked": False,
                        "scopes": ["jobs:run"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert validate_api_key(key, path)["credential_id"] == "test"
    with pytest.raises(Exception, match="invalid API key"):
        validate_api_key("wrong", path)
    with pytest.raises(ValidationError):
        _envelope(None)
    invalid = _envelope(AuthBlock(api_key=key)).model_copy(update={"auth": None})
    error = authenticate(invalid)
    assert error.code == 401
    assert error.message == "authentication block is required"


def test_jwt_valid_expired_and_bad_secret(monkeypatch):
    secret = "x" * 32
    monkeypatch.setenv("HARADIBOTS_JWT_SECRET", secret)
    claims = {
        "sub": "operator",
        "aud": "haradibots",
        "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
        "scope": "jobs:run jobs:read",
    }
    token = jwt.encode(claims, secret, algorithm="HS256")
    assert validate_jwt(token)["scopes"] == ["jobs:run", "jobs:read"]
    claims["exp"] = datetime.now(timezone.utc) - timedelta(seconds=1)
    expired = jwt.encode(claims, secret, algorithm="HS256")
    with pytest.raises(Exception, match="invalid JWT"):
        validate_jwt(expired)
    monkeypatch.setenv("HARADIBOTS_JWT_SECRET", "short")
    with pytest.raises(Exception, match="at least 32 bytes"):
        validate_jwt(token)


def test_auth_store_rejects_malformed_revoked_and_plaintext_records(tmp_path):
    import hashlib
    import json

    path = tmp_path / "credentials.json"
    cases = [
        [],
        {},
        {"credentials": ["bad"]},
        {"credentials": [{"key_hash": "short"}]},
        {
            "credentials": [
                {
                    "key_hash": "0" * 64,
                    "api_key": "must-not-exist",
                }
            ]
        },
    ]
    for value in cases:
        path.write_text(json.dumps(value), encoding="utf-8")
        with pytest.raises(AuthError):
            load_credentials(path)

    key = "revoked"
    path.write_text(
        json.dumps(
            {
                "credentials": [
                    {
                        "id": "revoked",
                        "key_hash": hashlib.sha256(key.encode()).hexdigest(),
                        "revoked": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(AuthError, match="revoked"):
        validate_api_key(key, path)


def test_local_key_bootstrap_reuses_private_key(tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("HARADIBOTS_CACHE_ROOT", str(tmp_path / "cache"))
    store = tmp_path / "credentials.json"
    store.write_text('{"credentials": []}', encoding="utf-8")

    first = ensure_local_api_key(store)
    second = ensure_local_api_key(store)

    assert first == second
    assert validate_api_key(first, store)["subject"] == "trusted-local-interface"
    serialized = json.loads(store.read_text(encoding="utf-8"))
    assert len(serialized["credentials"]) == 1
    assert first not in store.read_text(encoding="utf-8")


def test_jwt_missing_invalid_scope_and_authenticate_success(monkeypatch):
    secret = "s" * 32
    monkeypatch.delenv("HARADIBOTS_JWT_SECRET", raising=False)
    with pytest.raises(AuthError, match="not configured"):
        validate_jwt("token")
    with pytest.raises(AuthError, match="missing"):
        validate_jwt("")

    monkeypatch.setenv("HARADIBOTS_JWT_SECRET", secret)
    claims = {
        "sub": "operator",
        "aud": "haradibots",
        "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
        "scope": {"invalid": True},
    }
    token = jwt.encode(claims, secret, algorithm="HS256")
    with pytest.raises(AuthError, match="scope"):
        validate_jwt(token)

    claims["scope"] = ["jobs:run"]
    token = jwt.encode(claims, secret, algorithm="HS256")
    envelope = _envelope(AuthBlock(jwt_token=token))
    assert authenticate(envelope)["subject"] == "operator"


def test_fsm_all_valid_paths_and_invalid_transition():
    state = JobState.IDLE
    for event in (
        "job_received",
        "profile_complete",
        "plan_complete",
        "execution_complete",
        "validation_complete",
        "teardown_complete",
    ):
        state = transition(state, event)
    assert state is JobState.IDLE

    assert transition(JobState.ERROR, "begin_teardown") is JobState.TEARDOWN
    assert transition(JobState.PLANNING, "cluster_required") is JobState.CLUSTER_DISPATCH
    with pytest.raises(StateMachineError):
        transition(JobState.IDLE, "execution_complete")
