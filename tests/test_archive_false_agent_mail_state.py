from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "archive_false_agent_mail_state.py"


def _create_fixture_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE projects (id INTEGER PRIMARY KEY, slug TEXT NOT NULL, human_key TEXT NOT NULL);
            CREATE TABLE agents (
                id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL, name TEXT NOT NULL,
                retired_at TEXT NULL
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL, sender_id INTEGER NOT NULL,
                created_ts TEXT NOT NULL, thread_id TEXT, subject TEXT NOT NULL, topic TEXT,
                external_sender_host TEXT, external_sender_name TEXT
            );
            CREATE TABLE message_recipients (message_id INTEGER NOT NULL, agent_id INTEGER NOT NULL);
            CREATE TABLE agent_mail_bridge_deliveries (
                id INTEGER PRIMARY KEY, local_message_id INTEGER, source_host TEXT NOT NULL,
                source_agent TEXT NOT NULL, dest_host TEXT NOT NULL, dest_agent TEXT NOT NULL,
                status TEXT NOT NULL
            );
            CREATE TABLE agent_links (
                id INTEGER PRIMARY KEY, a_agent_id INTEGER NOT NULL, b_agent_id INTEGER NOT NULL,
                status TEXT NOT NULL, expires_ts TEXT
            );
            CREATE TABLE agent_mail_remote_identities (
                id INTEGER PRIMARY KEY, host_key TEXT NOT NULL, agent_name TEXT NOT NULL,
                remote_project_key TEXT, status TEXT
            );
            CREATE TABLE window_identities (
                id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL, display_name TEXT NOT NULL
            );
            CREATE TABLE file_reservations (
                id INTEGER PRIMARY KEY, agent_id INTEGER NOT NULL, path_pattern TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE fts_messages USING fts5(message_id UNINDEXED, subject, body);
            CREATE TRIGGER fts_messages_ad
            AFTER DELETE ON messages
            BEGIN
                DELETE FROM fts_messages WHERE rowid = old.id;
            END;
            """
        )
        connection.execute("INSERT INTO projects VALUES (1, 'home-antigravity', '/home/Antigravity')")
        active = ["mister-fixit", "RoseStone", "HumanOverseer", "TealEagle", "PurpleCanyon"]
        connection.executemany(
            "INSERT INTO agents (id,project_id,name,retired_at) VALUES (?,1,?,NULL)",
            [(index + 1, name) for index, name in enumerate(active)],
        )
        connection.execute("INSERT INTO agents VALUES (6,1,'TealCove','2026-11-23T04:42:11')")
        connection.executemany(
            "INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?)",
            [
                (10, 1, 6, "2026-11-23T04:42:23", "false-thread", "false sender", "cross-system", None, None),
                (
                    11,
                    1,
                    5,
                    "2026-11-23T04:43:23",
                    "timeout-thread",
                    "[codex] timeout notice",
                    "timeout_notice",
                    None,
                    None,
                ),
                (
                    12,
                    1,
                    5,
                    "2026-11-23T04:44:23",
                    "valid-thread",
                    "valid record",
                    "cross-system",
                    "Primus",
                    "RoseStone",
                ),
                (
                    13,
                    1,
                    5,
                    "2026-11-23T04:45:23",
                    "false-recipient-thread",
                    "false recipient",
                    "local",
                    None,
                    None,
                ),
            ],
        )
        connection.execute(
            "INSERT INTO fts_messages(rowid,message_id,subject,body) SELECT id,id,subject,'' FROM messages"
        )
        connection.executemany(
            "INSERT INTO message_recipients VALUES (?,?)",
            [(10, 5), (11, 5), (12, 5), (13, 6)],
        )
        connection.executemany(
            "INSERT INTO agent_mail_bridge_deliveries VALUES (?,?,?,?,?,?,?)",
            [
                (1, 10, "Datacube", "RoseStone", "Primus", "SwiftCat", "delivered"),
                (2, 12, "Primus", "RoseStone", "Datacube", "PurpleCanyon", "delivered"),
                (3, None, "Datacube", "PurpleCanyon", "Primus", "RoseStone", "delivered"),
                (4, None, "Datacube", "RoseStone", "Primus", "PurpleCanyon", "delivered"),
                (5, None, "OtherHost", "ValidPeer", "Datacube", "PurpleCanyon", "delivered"),
            ],
        )
        connection.execute("INSERT INTO agent_links VALUES (1,6,5,'approved','2026-11-18T00:00:00')")
        connection.executemany(
            "INSERT INTO agent_mail_remote_identities VALUES (?,?,?,?,?)",
            [
                (1, "Primus", "SwiftCat", "", "discovered"),
                (2, "Primus", "RoseStone", "", "active"),
                (3, "OtherHost", "ValidPeer", "", "active"),
            ],
        )
        connection.execute("INSERT INTO window_identities VALUES (1,1,'TealCove')")
        connection.commit()
    finally:
        connection.close()


def _run(database: Path, archive_root: Path, quarantine_parent: Path, *extra: str) -> dict:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--database",
            str(database),
            "--archive-root",
            str(archive_root),
            "--quarantine-parent",
            str(quarantine_parent),
            "--service-unit",
            "agent-mail-test-unit-that-does-not-exist.service",
            *extra,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return json.loads(completed.stdout)


def _load_cleanup_module():
    module_name = "archive_false_agent_mail_state_test"
    specification = importlib.util.spec_from_file_location(module_name, SCRIPT)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def test_archive_false_agent_mail_state_is_dry_run_then_recoverable_cleanup(tmp_path):
    database = tmp_path / "storage.sqlite3"
    archive_root = tmp_path / "archive"
    quarantine_parent = tmp_path / "quarantine"
    _create_fixture_database(database)

    message_dir = archive_root / "projects" / "home-antigravity" / "messages" / "2026" / "11"
    message_dir.mkdir(parents=True)
    for message_id in (10, 11, 12, 13):
        (message_dir / f"{message_id}.md").write_text(f"message {message_id}\n", encoding="utf-8")
    profile = archive_root / "projects" / "home-antigravity" / "agents" / "TealCove"
    profile.mkdir(parents=True)
    (profile / "profile.json").write_text("{}\n", encoding="utf-8")

    dry_run = _run(database, archive_root, quarantine_parent)
    assert dry_run["mode"] == "dry-run"
    assert dry_run["counts"] == {
        "agent_links": 1,
        "bridge_deliveries": 2,
        "false_agents": 1,
        "file_reservations": 0,
        "messages": 3,
        "remote_identities": 1,
        "window_identities": 1,
    }
    assert (message_dir / "10.md").exists()
    assert (message_dir / "11.md").exists()

    applied = _run(database, archive_root, quarantine_parent, "--apply")
    assert applied["mode"] == "applied"
    quarantine = Path(applied["quarantine"])
    assert (quarantine / "mailbox-preimage.sqlite3").is_file()
    manifest = json.loads((quarantine / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert (
        quarantine / "archive-files" / "projects" / "home-antigravity" / "messages" / "2026" / "11" / "10.md"
    ).is_file()
    assert (
        quarantine / "archive-files" / "projects" / "home-antigravity" / "agents" / "TealCove" / "profile.json"
    ).is_file()

    connection = sqlite3.connect(database)
    try:
        assert connection.execute("SELECT count(*) FROM agents WHERE name='TealCove'").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM messages WHERE id IN (10,11,13)").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM messages WHERE id=12").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM fts_messages WHERE message_id IN (10,11,13)").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM fts_messages WHERE message_id=12").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM message_recipients WHERE agent_id=6").fetchone()[0] == 0
        assert (
            connection.execute("SELECT count(*) FROM agent_mail_bridge_deliveries WHERE id IN (1,4)").fetchone()[0] == 0
        )
        assert connection.execute("SELECT count(*) FROM agent_mail_bridge_deliveries WHERE id=2").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM agent_mail_bridge_deliveries WHERE id=3").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM agent_mail_bridge_deliveries WHERE id=5").fetchone()[0] == 1
        assert (
            connection.execute(
                "SELECT count(*) FROM agent_mail_remote_identities WHERE agent_name='SwiftCat'"
            ).fetchone()[0]
            == 0
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM agent_mail_remote_identities WHERE agent_name='RoseStone'"
            ).fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM agent_mail_remote_identities WHERE agent_name='ValidPeer'"
            ).fetchone()[0]
            == 1
        )
    finally:
        connection.close()
    assert not (message_dir / "10.md").exists()
    assert not (message_dir / "11.md").exists()
    assert not (message_dir / "13.md").exists()
    assert (message_dir / "12.md").exists()
    assert not profile.exists()


def test_apply_reserves_sqlite_write_access_before_snapshot(monkeypatch, tmp_path):
    database = tmp_path / "storage.sqlite3"
    archive_root = tmp_path / "archive"
    quarantine_parent = tmp_path / "quarantine"
    _create_fixture_database(database)
    archive_root.mkdir()

    module = _load_cleanup_module()

    original_backup = module._backup_database

    def backup_with_lock_probe(source: Path, destination: Path) -> str:
        contender = sqlite3.connect(source, timeout=0)
        try:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                contender.execute("INSERT INTO agents VALUES (99,1,'unexpected',NULL)")
        finally:
            contender.close()
        return original_backup(source, destination)

    monkeypatch.setattr(module, "_backup_database", backup_with_lock_probe)
    result = module.run(
        argparse.Namespace(
            database=database,
            archive_root=archive_root,
            quarantine_parent=quarantine_parent,
            service_unit="agent-mail-test-unit-that-does-not-exist.service",
            apply=True,
        )
    )
    assert result["mode"] == "applied"


def test_apply_records_post_commit_active_file_removal_failure(monkeypatch, tmp_path):
    database = tmp_path / "storage.sqlite3"
    archive_root = tmp_path / "archive"
    quarantine_parent = tmp_path / "quarantine"
    _create_fixture_database(database)
    archive_root.mkdir()
    module = _load_cleanup_module()

    def fail_removal(_paths):
        raise OSError("simulated active archive removal failure")

    monkeypatch.setattr(module, "_remove_copied_active_paths", fail_removal)
    with pytest.raises(RuntimeError, match="Database cleanup committed"):
        module.run(
            argparse.Namespace(
                database=database,
                archive_root=archive_root,
                quarantine_parent=quarantine_parent,
                service_unit="agent-mail-test-unit-that-does-not-exist.service",
                apply=True,
            )
        )

    quarantine = next(quarantine_parent.glob(".quarantine-*"))
    manifest = json.loads((quarantine / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "database_committed_active_file_removal_failed"
    assert "simulated active archive removal failure" in manifest["active_file_removal_error"]
    connection = sqlite3.connect(database)
    try:
        assert connection.execute("SELECT count(*) FROM agents WHERE name='TealCove'").fetchone()[0] == 0
    finally:
        connection.close()
