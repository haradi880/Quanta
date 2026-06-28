from pathlib import Path

import pytest

from build.verify_bundle import REQUIRED_STEMS, verify_vendor
from core.runtime import NATIVE_ENV, configure_native_runtime


def test_bundle_verifier_fails_closed_when_native_assets_are_missing(tmp_path):
    with pytest.raises(RuntimeError, match="offline bundle is incomplete"):
        verify_vendor(tmp_path)


def test_bundle_verifier_and_runtime_resolver_cover_all_native_tools(
    tmp_path,
    monkeypatch,
):
    for stem in REQUIRED_STEMS:
        (tmp_path / f"{stem}.exe").write_bytes(b"native")
    (tmp_path / "convert_hf_to_gguf.py").write_text("# converter", encoding="utf-8")
    verified = verify_vendor(tmp_path)
    for variable in NATIVE_ENV.values():
        monkeypatch.delenv(variable, raising=False)

    configured = configure_native_runtime(tmp_path)

    assert len(verified) == 5
    assert set(configured) == set(NATIVE_ENV.values())
    assert all(Path(value).is_file() for value in configured.values())


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
    assert "nvidia/cuda:12.2.0-runtime-ubuntu22.04" in dockerfile
    assert "USER haradibots" in dockerfile
    assert "cluster.api_server:app" in dockerfile
    assert "import-isolation:" in workflow
    assert "docker-build:" in workflow
