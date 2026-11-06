import asyncio
import hashlib
import sqlite3
import subprocess
import threading
from typing import Any

from sqlalchemy import text
from typer.testing import CliRunner

from mcp_agent_mail.cli import app
from mcp_agent_mail.db import ensure_schema, get_database_path, get_session
from mcp_agent_mail.models import Agent, Message, Project


def test_cli_lint(monkeypatch):
    runner = CliRunner()
    captured: list[list[str]] = []

    def fake_run(command: list[str]) -> None:
        captured.append(command)

    monkeypatch.setattr("mcp_agent_mail.cli._run_command", fake_run)
    result = runner.invoke(app, ["lint"])
    assert result.exit_code == 0
    assert captured == [["ruff", "check", "--fix", "--unsafe-fixes"]]


def test_cli_typecheck(monkeypatch):
    runner = CliRunner()
    captured: list[list[str]] = []

    def fake_run(command: list[str]) -> None:
        captured.append(command)

    monkeypatch.setattr("mcp_agent_mail.cli._run_command", fake_run)
    result = runner.invoke(app, ["typecheck"])
    assert result.exit_code == 0
    assert captured == [["uvx", "ty", "check"]]


def test_cli_serve_http_uses_settings(isolated_env, monkeypatch):
    runner = CliRunner()
    call_args: dict[str, Any] = {}

    def fake_uvicorn_run(app, host, port, log_level="info"):
        call_args["app"] = app
        call_args["host"] = host
        call_args["port"] = port
        call_args["log_level"] = log_level

    monkeypatch.setattr("uvicorn.run", fake_uvicorn_run)
    result = runner.invoke(app, ["serve-http"])
    assert result.exit_code == 0
    assert call_args["host"] == "127.0.0.1"
    assert call_args["port"] == 8765


def test_cli_serve_stdio(isolated_env, monkeypatch):
    """Test that serve-stdio invokes FastMCP.run with stdio transport."""
    runner = CliRunner()
    call_args: dict[str, Any] = {}

    def fake_run(self, transport="stdio", **kwargs):
        call_args["transport"] = transport
        call_args["kwargs"] = kwargs

    # Patch FastMCP.run on the class before build_mcp_server returns an instance
    from fastmcp import FastMCP

    monkeypatch.setattr(FastMCP, "run", fake_run)
    result = runner.invoke(app, ["serve-stdio"])
    assert result.exit_code == 0
    assert call_args["transport"] == "stdio"


def test_cli_migrate(monkeypatch):
    runner = CliRunner()
    invoked: dict[str, bool] = {"called": False}

    async def fake_migrate(settings):
        invoked["called"] = True

    monkeypatch.setattr("mcp_agent_mail.cli.ensure_schema", fake_migrate)
    result = runner.invoke(app, ["migrate"])
    assert result.exit_code == 0
    assert invoked["called"] is True


def test_cli_list_projects(isolated_env):
    runner = CliRunner()

    async def seed() -> None:
        await ensure_schema()
        async with get_session() as session:
            project = Project(slug="demo", human_key="Demo")
            session.add(project)
            await session.commit()
            await session.refresh(project)
            assert project.id is not None
            session.add(
                Agent(
                    project_id=project.id,
                    name="BlueLake",
                    program="codex",
                    model="gpt-5",
                    task_description="",
                )
            )
            await session.commit()

    asyncio.run(seed())
    result = runner.invoke(app, ["list-projects", "--include-agents"])
    assert result.exit_code == 0
    assert "demo" in result.stdout
    assert "BlueLake" not in result.stdout


def test_projects_adopt_refuses_attested_messages_before_archive_move(tmp_path, isolated_env, monkeypatch):
    repo = tmp_path / "repo"
    source_path = repo / "source-worktree"
    target_path = repo / "target-worktree"
    source_path.mkdir(parents=True)
    target_path.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)

    async def seed() -> tuple[int, int, int]:
        await ensure_schema()
        async with get_session() as session:
            source = Project(slug="source-worktree", human_key=str(source_path))
            target = Project(slug="target-worktree", human_key=str(target_path))
            session.add(source)
            session.add(target)
            await session.commit()
            await session.refresh(source)
            await session.refresh(target)
            assert source.id is not None and target.id is not None
            sender = Agent(
                project_id=source.id,
                name="RoseStone",
                program="codex",
                model="gpt-5",
                task_description="",
            )
            session.add(sender)
            await session.commit()
            await session.refresh(sender)
            assert sender.id is not None
            body = "attested archive guard"
            message = Message(
                project_id=source.id,
                sender_id=sender.id,
                subject="Do not re-key",
                body_md=body,
                body_sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
                attestation_version=1,
                revision=1,
            )
            session.add(message)
            await session.commit()
            assert message.id is not None
            return source.id, target.id, message.id

    source_id, _target_id, message_id = asyncio.run(seed())
    archive_called = False

    async def unexpected_ensure_archive(*_args, **_kwargs):
        nonlocal archive_called
        archive_called = True
        raise AssertionError("archive must not be opened after attestation preflight fails")

    monkeypatch.setattr("mcp_agent_mail.storage.ensure_archive", unexpected_ensure_archive)
    runner = CliRunner()
    result = runner.invoke(app, ["projects", "adopt", str(source_path), str(target_path), "--apply"])

    assert result.exit_code != 0
    assert archive_called is False

    async def verify() -> int:
        async with get_session() as session:
            return int(
                (
                    await session.execute(
                        text("SELECT project_id FROM messages WHERE id = :id"),
                        {"id": message_id},
                    )
                ).scalar_one()
            )

    assert asyncio.run(verify()) == source_id


def test_projects_adopt_quiesces_attested_insert_until_rekey(tmp_path, isolated_env, monkeypatch):
    repo = tmp_path / "repo"
    source_path = repo / "source-worktree"
    target_path = repo / "target-worktree"
    source_path.mkdir(parents=True)
    target_path.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)

    async def seed() -> tuple[int, int]:
        await ensure_schema()
        async with get_session() as session:
            source = Project(slug="source-worktree", human_key=str(source_path))
            target = Project(slug="target-worktree", human_key=str(target_path))
            session.add_all([source, target])
            await session.commit()
            await session.refresh(source)
            assert source.id is not None
            sender = Agent(
                project_id=source.id,
                name="RoseStone",
                program="codex",
                model="gpt-5",
                task_description="",
            )
            session.add(sender)
            await session.commit()
            await session.refresh(sender)
            assert sender.id is not None
            return source.id, sender.id

    source_id, sender_id = asyncio.run(seed())
    db_path = get_database_path()
    assert db_path is not None
    started = threading.Event()
    finished = threading.Event()
    writer_error: list[Exception] = []
    writer_thread: list[threading.Thread] = []
    commit_calls = 0
    body = "concurrent attested insert"

    def insert_attested() -> None:
        started.set()
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(db_path, timeout=2.0)
            connection.execute("PRAGMA busy_timeout = 2000")
            connection.execute(
                """
                INSERT INTO messages
                    (project_id, sender_id, subject, body_md, importance, ack_required,
                     created_ts, attachments, body_sha256, attestation_version, revision)
                VALUES (?, ?, ?, ?, 'normal', 0, datetime('now'), '[]', ?, 1, 1)
                """,
                (source_id, sender_id, "Concurrent insert", body, hashlib.sha256(body.encode("utf-8")).hexdigest()),
            )
            connection.commit()
        except Exception as exc:
            writer_error.append(exc)
        finally:
            # Closing a rejected raw SQLite write releases its transaction,
            # just as production sessions do on request exit.
            if connection is not None:
                connection.close()
            finished.set()

    async def gate_archive_commit(*_args, **_kwargs):
        nonlocal commit_calls
        # The first commit initializes the archive repository during plan
        # construction; the second is the locked adoption archive commit.
        if commit_calls == 1:
            thread = threading.Thread(target=insert_attested, daemon=True)
            writer_thread.append(thread)
            thread.start()
            assert started.wait(1.0)
            await asyncio.sleep(0.05)
            # SQLite either waits for the immediate transaction or rejects the
            # competing writer as locked; it must not commit before re-key.
            assert not (finished.is_set() and not writer_error)
        commit_calls += 1
        return None

    monkeypatch.setattr("mcp_agent_mail.storage._commit", gate_archive_commit)
    result = CliRunner().invoke(app, ["projects", "adopt", str(source_path), str(target_path), "--apply"])

    assert result.exit_code == 0
    assert writer_thread
    writer_thread[0].join(timeout=2.0)
    assert finished.is_set()
    if writer_error:
        assert "locked" in str(writer_error[0]).lower()

def test_projects_adopt_archive_failure_rolls_back_attestation_quiesce(tmp_path, isolated_env, monkeypatch):
    repo = tmp_path / "repo"
    source_path = repo / "source-worktree"
    target_path = repo / "target-worktree"
    source_path.mkdir(parents=True)
    target_path.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)

    async def seed() -> tuple[int, int]:
        await ensure_schema()
        async with get_session() as session:
            source = Project(slug="source-worktree", human_key=str(source_path))
            target = Project(slug="target-worktree", human_key=str(target_path))
            session.add_all([source, target])
            await session.commit()
            await session.refresh(source)
            assert source.id is not None
            sender = Agent(
                project_id=source.id,
                name="RoseStone",
                program="codex",
                model="gpt-5",
                task_description="",
            )
            session.add(sender)
            await session.commit()
            await session.refresh(sender)
            assert sender.id is not None
            return source.id, sender.id

    source_id, sender_id = asyncio.run(seed())
    archive_failure_called = False

    async def fail_adopt_archive_commit(_repo, _settings, message, _paths):
        nonlocal archive_failure_called
        if message.startswith("adopt: move"):
            archive_failure_called = True
            raise RuntimeError("forced archive failure")

    monkeypatch.setattr("mcp_agent_mail.storage._commit", fail_adopt_archive_commit)
    result = CliRunner().invoke(app, ["projects", "adopt", str(source_path), str(target_path), "--apply"])

    assert result.exit_code != 0
    assert archive_failure_called is True

    async def verify_rollback_releases_writer_lock() -> None:
        async with get_session() as session:
            body = "accepted after failed adoption"
            session.add(
                Message(
                    project_id=source_id,
                    sender_id=sender_id,
                    subject="Post-failure attested message",
                    body_md=body,
                    body_sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
                    attestation_version=1,
                    revision=1,
                )
            )
            await session.commit()

    asyncio.run(verify_rollback_releases_writer_lock())


def test_archive_save_defaults_to_archive_preset(tmp_path, isolated_env, monkeypatch):
    runner = CliRunner()
    archive_path = tmp_path / "state.zip"
    archive_path.write_bytes(b"zip")
    captured: dict[str, Any] = {}

    def fake_archive(**kwargs):
        captured.update(kwargs)
        metadata = {"scrub_preset": kwargs["scrub_preset"], "projects_requested": list(kwargs["project_filters"])}
        return archive_path, metadata

    monkeypatch.setattr("mcp_agent_mail.cli._create_mailbox_archive", fake_archive)
    result = runner.invoke(app, ["archive", "save"])
    assert result.exit_code == 0
    assert captured["scrub_preset"] == "archive"


def test_clear_and_reset_skips_archive_when_disabled(isolated_env, monkeypatch):
    runner = CliRunner()

    def _should_not_run(**_kwargs):  # pragma: no cover - defensive
        raise AssertionError("archive should not be invoked when --no-archive is supplied")

    monkeypatch.setattr("mcp_agent_mail.cli._create_mailbox_archive", _should_not_run)
    result = runner.invoke(app, ["clear-and-reset-everything", "--force", "--no-archive"])
    assert result.exit_code == 0
