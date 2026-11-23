#!/usr/bin/env python3
"""Archive false Agent Mail identities and their non-canonical state safely.

The tool is intentionally fail-closed.  It takes a SQLite backup before any
mutation, requires the canonical active local roster, and holds planning and
database cleanup under one IMMEDIATE reservation.  It never deletes the
quarantine copy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

CANONICAL_LOCAL_IDENTITIES = frozenset({"mister-fixit", "RoseStone", "HumanOverseer", "TealEagle", "PurpleCanyon"})
CANONICAL_LOCAL_HOST = "datacube"
CANONICAL_REMOTE_HOST = "primus"
CANONICAL_REMOTE_IDENTITY = "rosestone"
CANONICAL_LOCAL_BRIDGE_RECIPIENT = "purplecanyon"
GENERIC_TIMEOUT_TOPIC = "timeout_notice"


@dataclass(frozen=True)
class CleanupPlan:
    false_agents: tuple[sqlite3.Row, ...]
    messages: tuple[sqlite3.Row, ...]
    bridge_deliveries: tuple[sqlite3.Row, ...]
    links: tuple[sqlite3.Row, ...]
    remote_identities: tuple[sqlite3.Row, ...]
    window_identities: tuple[sqlite3.Row, ...]
    reservations: tuple[sqlite3.Row, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    return (
        connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None
    )


def _require_schema(connection: sqlite3.Connection) -> None:
    required = {
        "agents",
        "messages",
        "message_recipients",
        "agent_links",
        "agent_mail_bridge_deliveries",
        "agent_mail_remote_identities",
        "window_identities",
        "file_reservations",
        "projects",
    }
    absent = sorted(name for name in required if not _table_exists(connection, name))
    if absent:
        raise RuntimeError(f"Agent Mail schema is incomplete; missing tables: {', '.join(absent)}")


def _placeholders(values: Iterable[object]) -> str:
    values = tuple(values)
    if not values:
        raise ValueError("Cannot build an empty SQL IN clause")
    return ",".join("?" for _ in values)


def _rows(connection: sqlite3.Connection, sql: str, params: tuple[object, ...] = ()) -> tuple[sqlite3.Row, ...]:
    return tuple(connection.execute(sql, params).fetchall())


def _verify_active_roster(connection: sqlite3.Connection) -> None:
    active_names = {
        str(row[0]) for row in connection.execute("SELECT name FROM agents WHERE retired_at IS NULL").fetchall()
    }
    if active_names != CANONICAL_LOCAL_IDENTITIES:
        raise RuntimeError(
            "Refusing cleanup because the active Agent Mail roster is not exactly canonical: "
            f"observed={sorted(active_names)!r} expected={sorted(CANONICAL_LOCAL_IDENTITIES)!r}"
        )


def _canonical_bridge_route_clause() -> tuple[str, tuple[str, ...]]:
    """Return the two valid host-qualified directions of the bridge route."""
    clause = (
        "((lower(source_host)=? AND lower(source_agent)=? AND lower(dest_host)=? AND lower(dest_agent)=?) "
        "OR (lower(source_host)=? AND lower(source_agent)=? AND lower(dest_host)=? AND lower(dest_agent)=?))"
    )
    return (
        clause,
        (
            CANONICAL_REMOTE_HOST,
            CANONICAL_REMOTE_IDENTITY,
            CANONICAL_LOCAL_HOST,
            CANONICAL_LOCAL_BRIDGE_RECIPIENT,
            CANONICAL_LOCAL_HOST,
            CANONICAL_LOCAL_BRIDGE_RECIPIENT,
            CANONICAL_REMOTE_HOST,
            CANONICAL_REMOTE_IDENTITY,
        ),
    )


def _stale_bridge_delivery_clause(false_names: tuple[str, ...]) -> tuple[str, tuple[object, ...]]:
    """Select only false routes, not unrelated peers that may be legitimate."""
    canonical_route, canonical_route_params = _canonical_bridge_route_clause()
    canonical_host_pair = (
        "((lower(source_host)=? AND lower(dest_host)=?) OR (lower(source_host)=? AND lower(dest_host)=?))"
    )
    clauses = [f"({canonical_host_pair} AND NOT {canonical_route})"]
    params: list[object] = [
        CANONICAL_LOCAL_HOST,
        CANONICAL_REMOTE_HOST,
        CANONICAL_REMOTE_HOST,
        CANONICAL_LOCAL_HOST,
        *canonical_route_params,
    ]
    if false_names:
        clauses.append(
            "("
            f"lower(source_agent) IN ({_placeholders(false_names)}) "
            "OR "
            f"lower(dest_agent) IN ({_placeholders(false_names)})"
            ")"
        )
        params.extend(false_names)
        params.extend(false_names)
    return "(" + " OR ".join(clauses) + ")", tuple(params)


def _stale_remote_identity_clause(false_names: tuple[str, ...]) -> tuple[str, tuple[object, ...]]:
    """Select Primus route drift and known false names, not arbitrary peers."""
    clauses = ["(lower(host_key)=? AND lower(agent_name) != ?)"]
    params: list[object] = [CANONICAL_REMOTE_HOST, CANONICAL_REMOTE_IDENTITY]
    if false_names:
        clauses.append(f"lower(agent_name) IN ({_placeholders(false_names)})")
        params.extend(false_names)
    return "(" + " OR ".join(clauses) + ")", tuple(params)


def build_plan(connection: sqlite3.Connection) -> CleanupPlan:
    _require_schema(connection)
    _verify_active_roster(connection)

    false_agents = _rows(
        connection,
        "SELECT id,name,project_id,retired_at FROM agents WHERE retired_at IS NOT NULL ORDER BY id",
    )
    false_ids = tuple(int(row["id"]) for row in false_agents)
    false_names = tuple(str(row["name"]).casefold() for row in false_agents)

    message_filters = [
        "topic = ?",
        "(external_sender_host IS NOT NULL AND NOT (lower(external_sender_host)=? AND lower(COALESCE(external_sender_name,''))=?))",
    ]
    message_params: list[object] = [GENERIC_TIMEOUT_TOPIC, CANONICAL_REMOTE_HOST, CANONICAL_REMOTE_IDENTITY]
    if false_ids:
        message_filters.insert(
            0,
            "("
            f"sender_id IN ({_placeholders(false_ids)}) "
            "OR id IN (SELECT message_id FROM message_recipients "
            f"WHERE agent_id IN ({_placeholders(false_ids)}))"
            ")",
        )
        message_params = [*false_ids, *false_ids, *message_params]
    messages = _rows(
        connection,
        "SELECT id,project_id,sender_id,created_ts,thread_id,subject,topic "
        "FROM messages WHERE " + " OR ".join(f"({item})" for item in message_filters) + " ORDER BY id",
        tuple(message_params),
    )
    message_ids = tuple(int(row["id"]) for row in messages)

    stale_bridge_clause, stale_bridge_params = _stale_bridge_delivery_clause(false_names)
    bridge_filters = [stale_bridge_clause]
    bridge_params: list[object] = list(stale_bridge_params)
    if message_ids:
        bridge_filters.insert(0, f"local_message_id IN ({_placeholders(message_ids)})")
        bridge_params = [*message_ids, *bridge_params]
    bridge_deliveries = _rows(
        connection,
        "SELECT id,local_message_id,source_host,source_agent,dest_host,dest_agent,status "
        "FROM agent_mail_bridge_deliveries WHERE "
        + " OR ".join(f"({item})" for item in bridge_filters)
        + " ORDER BY id",
        tuple(bridge_params),
    )

    if false_ids:
        links = _rows(
            connection,
            "SELECT id,a_agent_id,b_agent_id,status,expires_ts FROM agent_links WHERE "
            f"a_agent_id IN ({_placeholders(false_ids)}) OR b_agent_id IN ({_placeholders(false_ids)}) ORDER BY id",
            (*false_ids, *false_ids),
        )
        reservations = _rows(
            connection,
            "SELECT id,agent_id,path_pattern FROM file_reservations WHERE "
            f"agent_id IN ({_placeholders(false_ids)}) ORDER BY id",
            false_ids,
        )
    else:
        links = ()
        reservations = ()

    stale_remote_clause, stale_remote_params = _stale_remote_identity_clause(false_names)
    remote_identities = _rows(
        connection,
        "SELECT id,host_key,agent_name,remote_project_key,status FROM agent_mail_remote_identities "
        "WHERE " + stale_remote_clause + " ORDER BY id",
        stale_remote_params,
    )
    if false_names:
        window_identities = _rows(
            connection,
            "SELECT id,project_id,display_name FROM window_identities WHERE "
            f"lower(display_name) IN ({_placeholders(false_names)}) ORDER BY id",
            false_names,
        )
    else:
        window_identities = ()

    return CleanupPlan(
        false_agents=false_agents,
        messages=messages,
        bridge_deliveries=bridge_deliveries,
        links=links,
        remote_identities=remote_identities,
        window_identities=window_identities,
        reservations=reservations,
    )


def _row_list(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def _plan_summary(plan: CleanupPlan) -> dict[str, Any]:
    return {
        "false_agent_ids": [int(row["id"]) for row in plan.false_agents],
        "false_agent_names": [str(row["name"]) for row in plan.false_agents],
        "message_ids": [int(row["id"]) for row in plan.messages],
        "counts": {
            "false_agents": len(plan.false_agents),
            "messages": len(plan.messages),
            "bridge_deliveries": len(plan.bridge_deliveries),
            "agent_links": len(plan.links),
            "remote_identities": len(plan.remote_identities),
            "window_identities": len(plan.window_identities),
            "file_reservations": len(plan.reservations),
        },
    }


def _require_inactive_service(unit: str) -> None:
    result = subprocess.run(
        ["systemctl", "is-active", "--quiet", unit], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    if result.returncode == 0:
        raise RuntimeError(f"Refusing cleanup while {unit} is active; quiesce it first.")


def _backup_database(source: Path, destination: Path) -> str:
    source_connection = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    destination_connection = sqlite3.connect(destination)
    try:
        source_connection.backup(destination_connection)
    finally:
        destination_connection.close()
        source_connection.close()
    destination.chmod(0o600)
    return _sha256(destination)


def _parse_year_month(value: str) -> tuple[str, str] | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.strftime("%Y"), parsed.strftime("%m")


def _message_and_profile_paths(
    connection: sqlite3.Connection,
    archive_root: Path,
    plan: CleanupPlan,
) -> tuple[Path, ...]:
    project_slugs = {
        int(row["id"]): str(row["slug"]) for row in connection.execute("SELECT id,slug FROM projects").fetchall()
    }
    paths: set[Path] = set()
    for message in plan.messages:
        year_month = _parse_year_month(str(message["created_ts"]))
        slug = project_slugs.get(int(message["project_id"]))
        if year_month is None or not slug:
            continue
        year, month = year_month
        directory = archive_root / "projects" / slug / "messages" / year / month
        if not directory.is_dir():
            continue
        message_id = int(message["id"])
        legacy = directory / f"{message_id}.md"
        if legacy.is_file():
            paths.add(legacy)
        paths.update(path for path in directory.glob(f"*__*__{message_id}.md") if path.is_file())
    for agent in plan.false_agents:
        slug = project_slugs.get(int(agent["project_id"]))
        if not slug:
            continue
        profile = archive_root / "projects" / slug / "agents" / str(agent["name"])
        if profile.exists():
            paths.add(profile)
    return tuple(sorted(paths))


def _copy_paths_to_quarantine(paths: Iterable[Path], archive_root: Path, quarantine: Path) -> tuple[Path, ...]:
    copied: list[Path] = []
    for source in paths:
        relative = source.relative_to(archive_root)
        destination = quarantine / "archive-files" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
        copied.append(source)
    return tuple(copied)


def _delete_ids(connection: sqlite3.Connection, table: str, ids: Iterable[int]) -> None:
    values = tuple(ids)
    if values:
        connection.execute(f"DELETE FROM {table} WHERE id IN ({_placeholders(values)})", values)


def _apply_database_cleanup(connection: sqlite3.Connection, plan: CleanupPlan) -> None:
    message_ids = tuple(int(row["id"]) for row in plan.messages)
    if message_ids:
        connection.execute(
            f"DELETE FROM message_recipients WHERE message_id IN ({_placeholders(message_ids)})", message_ids
        )
    _delete_ids(connection, "agent_mail_bridge_deliveries", (int(row["id"]) for row in plan.bridge_deliveries))
    if message_ids:
        connection.execute(f"DELETE FROM messages WHERE id IN ({_placeholders(message_ids)})", message_ids)
    _delete_ids(connection, "agent_links", (int(row["id"]) for row in plan.links))
    _delete_ids(connection, "file_reservations", (int(row["id"]) for row in plan.reservations))
    _delete_ids(connection, "window_identities", (int(row["id"]) for row in plan.window_identities))
    _delete_ids(connection, "agent_mail_remote_identities", (int(row["id"]) for row in plan.remote_identities))
    _delete_ids(connection, "agents", (int(row["id"]) for row in plan.false_agents))


def _verify_post_cleanup(connection: sqlite3.Connection, plan: CleanupPlan) -> None:
    _verify_active_roster(connection)
    message_ids = tuple(int(row["id"]) for row in plan.messages)
    if message_ids:
        remaining = connection.execute(
            f"SELECT count(*) FROM messages WHERE id IN ({_placeholders(message_ids)})", message_ids
        ).fetchone()[0]
        if remaining:
            raise RuntimeError(f"Cleanup left {remaining} selected messages in the active database")
        if _table_exists(connection, "fts_messages"):
            remaining_fts = connection.execute(
                f"SELECT count(*) FROM fts_messages WHERE message_id IN ({_placeholders(message_ids)})",
                message_ids,
            ).fetchone()[0]
            if remaining_fts:
                raise RuntimeError(f"Cleanup left {remaining_fts} selected message(s) in full-text search")
    generic_timeouts = connection.execute(
        "SELECT count(*) FROM messages WHERE topic=?", (GENERIC_TIMEOUT_TOPIC,)
    ).fetchone()[0]
    if generic_timeouts:
        raise RuntimeError(f"Cleanup left {generic_timeouts} generic timeout notice(s) active")
    false_names = tuple(str(row["name"]).casefold() for row in plan.false_agents)
    stale_remote_clause, stale_remote_params = _stale_remote_identity_clause(false_names)
    noncanonical_remote = connection.execute(
        "SELECT count(*) FROM agent_mail_remote_identities WHERE " + stale_remote_clause,
        stale_remote_params,
    ).fetchone()[0]
    if noncanonical_remote:
        raise RuntimeError(f"Cleanup left {noncanonical_remote} non-canonical remote identity row(s)")
    stale_bridge_clause, stale_bridge_params = _stale_bridge_delivery_clause(false_names)
    noncanonical_bridge = connection.execute(
        "SELECT count(*) FROM agent_mail_bridge_deliveries WHERE " + stale_bridge_clause,
        stale_bridge_params,
    ).fetchone()[0]
    if noncanonical_bridge:
        raise RuntimeError(f"Cleanup left {noncanonical_bridge} non-canonical bridge delivery row(s)")
    dangling = connection.execute("PRAGMA foreign_key_check").fetchall()
    if dangling:
        raise RuntimeError(f"Foreign-key validation failed after cleanup: {dangling[:3]!r}")
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        raise RuntimeError(f"SQLite integrity check failed after cleanup: {integrity}")


def _remove_copied_active_paths(paths: Iterable[Path]) -> None:
    for path in sorted(paths, key=lambda item: len(item.parts), reverse=True):
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o600)


def _new_quarantine(parent: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = parent / f".quarantine-{stamp}-{uuid.uuid4()}"
    path.mkdir(parents=True, exist_ok=False)
    path.chmod(0o700)
    return path


def run(args: argparse.Namespace) -> dict[str, Any]:
    database = args.database.resolve()
    archive_root = args.archive_root.resolve()
    if not database.is_file():
        raise RuntimeError(f"Database does not exist: {database}")
    if not archive_root.is_dir():
        raise RuntimeError(f"Archive root does not exist: {archive_root}")
    _require_inactive_service(args.service_unit)

    if not args.apply:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            plan = build_plan(connection)
            archive_paths = _message_and_profile_paths(connection, archive_root, plan)
        finally:
            connection.close()
        summary = _plan_summary(plan)
        summary["archive_paths"] = [str(path.relative_to(archive_root)) for path in archive_paths]
        summary["service_unit"] = args.service_unit
        return {"mode": "dry-run", **summary}

    quarantine = _new_quarantine(args.quarantine_parent.resolve())
    preimage = quarantine / "mailbox-preimage.sqlite3"
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("BEGIN IMMEDIATE")
        plan = build_plan(connection)
        archive_paths = _message_and_profile_paths(connection, archive_root, plan)
        summary = _plan_summary(plan)
        summary["archive_paths"] = [str(path.relative_to(archive_root)) for path in archive_paths]
        summary["service_unit"] = args.service_unit
        # The reservation prevents a new writer from committing between the
        # plan and the snapshot. A separate read-only connection can still
        # obtain the last committed image; backing up from the write-holding
        # connection itself can deadlock in SQLite.
        preimage_sha256 = _backup_database(database, preimage)
        copied_paths = _copy_paths_to_quarantine(archive_paths, archive_root, quarantine)
        manifest = {
            "status": "prepared",
            "database": str(database),
            "archive_root": str(archive_root),
            "preimage": {"path": preimage.name, "sha256": preimage_sha256},
            "selection": summary,
            "records": {
                "false_agents": _row_list(plan.false_agents),
                "messages": _row_list(plan.messages),
                "bridge_deliveries": _row_list(plan.bridge_deliveries),
                "agent_links": _row_list(plan.links),
                "remote_identities": _row_list(plan.remote_identities),
                "window_identities": _row_list(plan.window_identities),
                "file_reservations": _row_list(plan.reservations),
            },
        }
        manifest_path = quarantine / "manifest.json"
        _write_manifest(manifest_path, manifest)
        _apply_database_cleanup(connection, plan)
        _verify_post_cleanup(connection, plan)
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()

    manifest["status"] = "database_committed_files_pending"
    manifest["post_cleanup_database_sha256"] = _sha256(database)
    manifest["pending_active_paths"] = [str(path.relative_to(archive_root)) for path in copied_paths]
    _write_manifest(manifest_path, manifest)
    try:
        _remove_copied_active_paths(copied_paths)
        remaining_paths = [path for path in copied_paths if path.exists()]
        if remaining_paths:
            raise RuntimeError(
                "Active archive path removal reported success but left: "
                + ", ".join(str(path.relative_to(archive_root)) for path in remaining_paths)
            )
    except Exception as exc:
        manifest["status"] = "database_committed_active_file_removal_failed"
        manifest["active_file_removal_error"] = f"{type(exc).__name__}: {exc}"
        _write_manifest(manifest_path, manifest)
        raise RuntimeError(
            f"Database cleanup committed but active archive file removal failed; recover only from {manifest_path}"
        ) from exc

    manifest["status"] = "completed"
    manifest["removed_archive_paths"] = [str(path.relative_to(archive_root)) for path in copied_paths]
    manifest.pop("pending_active_paths", None)
    _write_manifest(manifest_path, manifest)
    return {"mode": "applied", "quarantine": str(quarantine), **summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--quarantine-parent", type=Path, required=True)
    parser.add_argument("--service-unit", default="agent-mail.service")
    parser.add_argument("--apply", action="store_true", help="Perform the cleanup; default is a read-only dry run.")
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), sort_keys=True))
    except Exception as exc:
        print(json.dumps({"error": str(exc), "status": "blocked"}, sort_keys=True), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
