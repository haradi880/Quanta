import os
import sys

from core.runtime import NATIVE_ENV, NATIVE_FILENAMES, configure_native_runtime


def test_native_runtime_missing_bundle_and_complete_discovery(tmp_path, monkeypatch):
    assert configure_native_runtime(tmp_path / "missing") == {}

    for name in NATIVE_ENV:
        path = tmp_path / NATIVE_FILENAMES[name]
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
    binary = vendor / "GarnetServer.exe"
    binary.write_bytes(b"tool")
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.delenv("HARADIBOTS_GARNET_BIN", raising=False)

    configured = configure_native_runtime()

    assert configured["HARADIBOTS_GARNET_BIN"] == str(binary.resolve())


def test_native_runtime_prefers_bundled_cuda_tools_on_nvidia_host(
    tmp_path,
    monkeypatch,
):
    import core.runtime as runtime

    cuda = tmp_path / "cuda"
    cuda.mkdir()
    for name in ("llama-completion", "llama-quantize", "llama-perplexity"):
        (tmp_path / NATIVE_FILENAMES[name]).write_bytes(b"cpu")
        (cuda / NATIVE_FILENAMES[name]).write_bytes(b"cuda")
        monkeypatch.delenv(NATIVE_ENV[name], raising=False)
    monkeypatch.setattr(runtime, "_nvidia_driver_available", lambda: True)

    configured = configure_native_runtime(tmp_path)

    assert configured["HARADIBOTS_LLAMA_BIN"] == str(
        (cuda / "llama-completion.exe").resolve()
    )
    assert configured["HARADIBOTS_LLAMA_QUANTIZE_BIN"] == str(
        (cuda / "llama-quantize.exe").resolve()
    )
