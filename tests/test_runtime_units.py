import os
import sys

from core.runtime import NATIVE_ENV, configure_native_runtime


def test_native_runtime_missing_bundle_and_complete_discovery(tmp_path, monkeypatch):
    assert configure_native_runtime(tmp_path / "missing") == {}

    for name in NATIVE_ENV:
        path = tmp_path / name
        path.write_bytes(b"tool")
    for variable in NATIVE_ENV.values():
        monkeypatch.delenv(variable, raising=False)

    configured = configure_native_runtime(tmp_path)

    assert configured.keys() == set(NATIVE_ENV.values())
    assert all(os.path.isabs(value) for value in configured.values())


def test_native_runtime_preserves_explicit_environment(tmp_path, monkeypatch):
    binary = tmp_path / "llama-cli.exe"
    binary.write_bytes(b"tool")
    monkeypatch.setenv("HARADIBOTS_LLAMA_BIN", "explicit")

    assert configure_native_runtime(tmp_path) == {}
    assert os.environ["HARADIBOTS_LLAMA_BIN"] == "explicit"


def test_native_runtime_uses_frozen_vendor_root(tmp_path, monkeypatch):
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    binary = vendor / "redis-server.exe"
    binary.write_bytes(b"tool")
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.delenv("HARADIBOTS_REDIS_BIN", raising=False)

    configured = configure_native_runtime()

    assert configured["HARADIBOTS_REDIS_BIN"] == str(binary.resolve())
