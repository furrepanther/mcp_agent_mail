from __future__ import annotations

import ast
from pathlib import Path


CLI_SOURCE = Path(__file__).resolve().parents[1] / "src" / "mcp_agent_mail" / "cli.py"


def test_cli_never_imports_or_calls_the_browser_module() -> None:
    tree = ast.parse(CLI_SOURCE.read_text(encoding="utf-8"), filename=str(CLI_SOURCE))

    browser_imports = [
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Import)
            and any(alias.name == "webbrowser" for alias in node.names)
        )
        or (
            isinstance(node, ast.ImportFrom)
            and node.module == "webbrowser"
        )
    ]
    browser_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "webbrowser"
    ]

    assert browser_imports == []
    assert browser_calls == []


def test_preview_flag_documents_the_safe_manual_open_behavior() -> None:
    source = CLI_SOURCE.read_text(encoding="utf-8")

    assert "--open-browser/--no-open-browser" in source
    assert "automatic browser launch is disabled" in source
    assert "manually if needed" in source
