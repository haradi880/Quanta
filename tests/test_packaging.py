import hashlib
import json
from pathlib import Path

import pytest

from build.verify_bundle import REQUIRED_FILENAMES, REQUIRED_STEMS, verify_vendor
from core.runtime import NATIVE_ENV, configure_native_runtime


def write_manifest(root, assets):
    files = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
        and path != root / "vendor-manifest.json"
        and path != root / "README.md"
    }
    (root / "vendor-manifest.json").write_text(
        json.dumps({"schema_version": "1", "assets": assets, "files": files}),
        encoding="utf-8",
    )


def test_bundle_verifier_fails_closed_when_native_assets_are_missing(tmp_path):
    with pytest.raises(RuntimeError, match="offline bundle is incomplete"):
        verify_vendor(tmp_path)


def test_bundle_verifier_and_runtime_resolver_cover_all_native_tools(
    tmp_path,
    monkeypatch,
):
    assets = {}
    for stem in REQUIRED_STEMS:
        content = f"native-{stem}".encode()
        (tmp_path / REQUIRED_FILENAMES[stem]).write_bytes(content)
        assets[stem] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "source": "https://example.invalid/release",
            "license": "MIT",
        }
    converter = b"# converter"
    (tmp_path / "convert_hf_to_gguf.py").write_bytes(converter)
    assets["convert_hf_to_gguf.py"] = {
        "sha256": hashlib.sha256(converter).hexdigest(),
        "source": "https://example.invalid/source",
        "license": "MIT",
    }
    write_manifest(tmp_path, assets)
    verified = verify_vendor(tmp_path)
    for variable in NATIVE_ENV.values():
        monkeypatch.delenv(variable, raising=False)

    configured = configure_native_runtime(tmp_path)

    assert len(verified) == len(REQUIRED_STEMS) + 1
    assert set(configured) == set(NATIVE_ENV.values())
    assert all(Path(value).is_file() for value in configured.values())


def test_bundle_verifier_rejects_tampered_native_asset(tmp_path):
    assets = {}
    for stem in REQUIRED_STEMS:
        path = tmp_path / REQUIRED_FILENAMES[stem]
        path.write_bytes(stem.encode())
        assets[stem] = {
            "sha256": hashlib.sha256(stem.encode()).hexdigest(),
            "source": "https://example.invalid/release",
            "license": "MIT",
        }
    converter = tmp_path / "convert_hf_to_gguf.py"
    converter.write_bytes(b"converter")
    assets[converter.name] = {
        "sha256": hashlib.sha256(b"converter").hexdigest(),
        "source": "https://example.invalid/source",
        "license": "MIT",
    }
    write_manifest(tmp_path, assets)
    (tmp_path / REQUIRED_FILENAMES["llama-cli"]).write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="checksum mismatch for file: llama-cli.exe"):
        verify_vendor(tmp_path)


def test_packaging_manifests_are_offline_and_one_dir():
    root = Path(__file__).resolve().parents[1]
    spec = (root / "build" / "fat_binary.spec").read_text(encoding="utf-8")
    dockerfile = (root / "build" / "Dockerfile").read_text(encoding="utf-8")
    workflow = (root / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    assert "COLLECT(" in spec
    assert "onefile" not in spec.lower()
    assert 'name="HaradiBots"' in spec
    assert "Path(SPECPATH).resolve().parent" in spec
    assert "parent.parent" not in spec
    assert '".exe", ".dll", ".pyd"' in spec
    assert "datas.append" in spec
    assert "nvidia/cuda:12.2.0-runtime-ubuntu22.04" in dockerfile
    assert "USER haradibots" in dockerfile
    assert "cluster.api_server:app" in dockerfile
    assert "import-isolation:" in workflow
    assert "docker-build:" in workflow
