"""AST enforcement of architecture import boundaries."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def boundary_violations(root: Path = ROOT) -> list[str]:
    violations: list[str] = []
    interface_forbidden = ("engines", "cluster", "core.accelerator")
    core_forbidden = ("ui", "cli", "notebooks")

    for directory in ("ui", "cli", "notebooks"):
        for path in (root / directory).rglob("*.py"):
            for module in imported_modules(path):
                if any(
                    module == prefix or module.startswith(prefix + ".")
                    for prefix in interface_forbidden
                ):
                    violations.append(f"{path.relative_to(root)} -> {module}")

    for directory in ("core", "engines"):
        for path in (root / directory).rglob("*.py"):
            for module in imported_modules(path):
                if any(
                    module == prefix or module.startswith(prefix + ".")
                    for prefix in core_forbidden
                ):
                    violations.append(f"{path.relative_to(root)} -> {module}")
    return violations


def test_import_isolation() -> None:
    assert boundary_violations() == []


def test_linter_detects_deliberate_bad_interface_import(tmp_path) -> None:
    for directory in ("ui", "cli", "notebooks", "core", "engines"):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "cli" / "bad.py").write_text(
        "from engines.gguf_worker import GGUFWorker\n",
        encoding="utf-8",
    )

    assert boundary_violations(tmp_path) == [
        f"{Path('cli') / 'bad.py'} -> engines.gguf_worker"
    ]
