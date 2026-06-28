import sys

import pytest

from core.frozen_converter import run_bundled_converter


def test_frozen_converter_runs_pinned_script_with_bundled_gguf(tmp_path, monkeypatch):
    converter = tmp_path / "convert_hf_to_gguf.py"
    converter.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "Path(sys.argv[1]).write_text(sys.argv[2], encoding='utf-8')\n",
        encoding="utf-8",
    )
    (tmp_path / "gguf-py").mkdir()
    (tmp_path / "conversion").mkdir()
    output = tmp_path / "result.txt"
    original_path = list(sys.path)
    original_argv = list(sys.argv)
    monkeypatch.setattr(sys, "path", original_path)
    monkeypatch.setattr(sys, "argv", original_argv)

    assert run_bundled_converter([str(converter), str(output), "converted"]) == 0
    assert output.read_text(encoding="utf-8") == "converted"
    assert sys.path[0] == str((tmp_path / "gguf-py").resolve())


@pytest.mark.parametrize(
    "setup, message",
    [
        ("none", "path is required"),
        ("wrong-name", "path is invalid"),
        ("no-gguf", "gguf-py is missing"),
        ("no-conversion", "conversion is missing"),
    ],
)
def test_frozen_converter_fails_closed(tmp_path, setup, message):
    arguments = []
    if setup != "none":
        name = "wrong.py" if setup == "wrong-name" else "convert_hf_to_gguf.py"
        converter = tmp_path / name
        converter.write_text("", encoding="utf-8")
        arguments = [str(converter)]
        if setup == "no-conversion":
            (tmp_path / "gguf-py").mkdir()

    with pytest.raises(SystemExit, match=message):
        run_bundled_converter(arguments)
