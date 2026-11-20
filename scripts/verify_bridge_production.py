#!/usr/bin/env python3
"""Read-only production verifier for the authenticated Agent Mail bridge.

This intentionally performs no migration, cleanup, network request, or token
rotation.  It proves that the deployed database and host-local secret wiring
are ready for the reviewed bridge code; a failed check is a promotion block.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import sqlite3
import stat
import sys
from pathlib import Path
from typing import Any


CANONICAL_AGENTS = {"datacube": "PurpleCanyon", "primus": "RoseStone"}
CANONICAL_PROJECTS = {"datacube": "/home/Antigravity", "primus": "/home/antigravity"}
STALE_AGENTS = {"swiftcat", "greenstone", "mister-fixit", "ambergrove", "tealcove"}
ACTIVE_DELIVERY_STATUSES = {"queued", "retry", "sending"}
ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _host_token_var(host_key: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", host_key.strip()).strip("_").upper()
    return f"AGENT_MAIL_BRIDGE_PEER_TOKEN_{normalized}"


def _parse_env_file(path: Path) -> tuple[dict[str, str], list[str]]:
    values: dict[str, str] = {}
    errors: list[str] = []
    if not path.exists():
        return values, [f"secret file missing: {path}"]
    if not path.is_file():
        return values, [f"secret path is not a regular file: {path}"]
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        errors.append("secret file is accessible by group/other users")
    for line_number, raw_line in enumerate(path.read_text(errors="strict").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, raw_value = line.partition("=")
        if not separator or not ENV_KEY_RE.fullmatch(key.strip()):
            errors.append(f"invalid environment assignment at line {line_number}")
            continue
        try:
            parsed = shlex.split(raw_value, comments=False, posix=True)
        except ValueError:
            errors.append(f"invalid shell quoting at line {line_number}")
            continue
        if len(parsed) != 1:
            errors.append(f"environment assignment at line {line_number} is not scalar")
            continue
        values[key.strip()] = parsed[0]
    return values, errors


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}


def _result(name: str, ok: bool, detail: str, **extra: Any) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail, **extra}


def verify(
    *,
    database: Path,
    identity_env: Path,
    token_env: Path,
    host_key: str,
    peer_host: str,
    peer_project: str,
    peer_api_url: str,
) -> dict[str, Any]:
    host = host_key.strip().lower()
    peer = peer_host.strip().lower()
    checks: list[dict[str, Any]] = []
    expected_agent = CANONICAL_AGENTS.get(host, "")
    expected_project = CANONICAL_PROJECTS.get(host, "")
    if not expected_agent or not expected_project:
        checks.append(_result("host_mapping", False, "host is not a canonical bridge host"))
    else:
        checks.append(_result("host_mapping", True, f"canonical host {host_key}"))

    identity, identity_errors = _parse_env_file(identity_env)
    identity_ok = not identity_errors
    if identity.get("AGENT_MAIL_HOST_KEY") != host_key:
        identity_ok = False
        identity_errors.append("identity env has no matching explicit AGENT_MAIL_HOST_KEY")
    if identity.get("AGENT_MAIL_AGENT") != expected_agent:
        identity_ok = False
        identity_errors.append(f"identity agent is not canonical {expected_agent}")
    if identity.get("AGENT_MAIL_PROJECT") != expected_project:
        identity_ok = False
        identity_errors.append(f"identity project is not canonical {expected_project}")
    checks.append(
        _result(
            "local_identity",
            identity_ok,
            "; ".join(identity_errors) if identity_errors else "explicit canonical host, agent, and project are present",
        )
    )

    token_values, token_errors = _parse_env_file(token_env)
    token_name = _host_token_var(peer_host)
    token = token_values.get(token_name, "")
    token_ok = not token_errors and bool(token)
    if not token:
        token_errors.append(f"missing host-scoped {token_name}")
    if any(key in token_values for key in ("AGENT_MAIL_BRIDGE_PEER_TOKEN", "ANTIGRAVITY_AGENT_MAIL_BRIDGE_PEER_TOKEN")):
        token_ok = False
        token_errors.append("unscoped peer-token variable is forbidden")
    checks.append(
        _result(
            "peer_token_file",
            token_ok,
            "; ".join(token_errors) if token_errors else "host-scoped peer token is present and file permissions are private",
            token_variable=token_name,
        )
    )

    if not database.exists():
        checks.append(_result("database", False, f"database is missing: {database}"))
        return {"ok": False, "checks": checks}

    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        checks.append(_result("database", False, f"database cannot be opened read-only: {type(exc).__name__}"))
        return {"ok": False, "checks": checks}

    try:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        required_tables = {
            "messages",
            "agent_mail_peers",
            "agent_mail_remote_identities",
            "agent_mail_bridge_deliveries",
        }
        missing_tables = sorted(required_tables - tables)
        checks.append(
            _result(
                "database_schema",
                not missing_tables,
                "required bridge tables are present" if not missing_tables else f"missing tables: {', '.join(missing_tables)}",
            )
        )
        if missing_tables:
            return {"ok": False, "checks": checks}

        message_columns = _table_columns(connection, "messages")
        required_message_columns = {
            "external_sender_host",
            "external_sender_name",
            "external_message_id",
            "external_project_key",
            "external_thread_id",
            "bridge_dedupe_id",
            "bridge_dedupe_scope",
            "bridge_revision_scope",
            "body_sha256",
            "attestation_version",
            "revision",
            "supersedes_message_id",
        }
        missing_message_columns = sorted(required_message_columns - message_columns)
        checks.append(
            _result(
                "message_schema",
                not missing_message_columns,
                "message provenance and attestation columns are present"
                if not missing_message_columns
                else f"missing message columns: {', '.join(missing_message_columns)}",
            )
        )

        remote_identity_columns = _table_columns(connection, "agent_mail_remote_identities")
        required_remote_identity_columns = {"host_key", "agent_name", "remote_project_key", "status"}
        missing_remote_identity_columns = sorted(required_remote_identity_columns - remote_identity_columns)
        checks.append(
            _result(
                "remote_identity_schema",
                not missing_remote_identity_columns,
                "remote identity routing columns are present"
                if not missing_remote_identity_columns
                else f"missing remote identity columns: {', '.join(missing_remote_identity_columns)}",
            )
        )

        peer_columns = _table_columns(connection, "agent_mail_peers")
        required_peer_columns = {
            "host_key",
            "api_url",
            "project_key",
            "auth_token_hash",
            "enabled",
            "bridge_protocol_version",
        }
        missing_peer_columns = sorted(required_peer_columns - peer_columns)
        if missing_peer_columns:
            checks.append(_result("peer_schema", False, f"missing peer columns: {', '.join(missing_peer_columns)}"))
        else:
            row = connection.execute(
                "SELECT host_key, api_url, project_key, auth_token_hash, enabled, bridge_protocol_version "
                "FROM agent_mail_peers WHERE lower(host_key)=lower(?) LIMIT 1",
                (peer_host,),
            ).fetchone()
            peer_errors: list[str] = []
            if row is None:
                peer_errors.append("canonical peer row is missing")
            else:
                stored_host, api_url, project_key, token_hash, enabled, protocol_version = row
                if str(stored_host).casefold() != peer:
                    peer_errors.append("peer host key is not canonical")
                if str(api_url or "").rstrip("/") != peer_api_url.rstrip("/"):
                    peer_errors.append("peer API URL differs from the canonical tunnel endpoint")
                if str(project_key or "") != peer_project:
                    peer_errors.append("peer project key differs from the canonical peer project")
                if not bool(enabled):
                    peer_errors.append("peer is disabled")
                if not token_hash:
                    peer_errors.append("peer auth token hash is missing")
                elif token and hashlib.sha256(token.encode("utf-8")).hexdigest() != str(token_hash):
                    peer_errors.append("configured peer token does not match the stored hash")
                if int(protocol_version or 0) < 1:
                    peer_errors.append("peer bridge_protocol_version does not support attested delivery")
            checks.append(_result("peer_configuration", not peer_errors, "; ".join(peer_errors) or "canonical enabled peer and token hash match"))

        identity_rows = connection.execute(
            "SELECT host_key, agent_name, remote_project_key, status "
            "FROM agent_mail_remote_identities WHERE lower(host_key)=lower(?)",
            (peer_host,),
        ).fetchall()
        active_stale = [
            (agent, project, status)
            for _host, agent, project, status in identity_rows
            if str(status or "").casefold() in {"active", "enabled"}
            and str(agent or "").casefold() in STALE_AGENTS
        ]
        checks.append(
            _result(
                "active_remote_identity",
                not active_stale,
                "no stale active remote identity remains"
                if not active_stale
                else f"stale active identities require quarantine: {len(active_stale)}",
                historical_rows=len(identity_rows),
            )
        )

        delivery_columns = _table_columns(connection, "agent_mail_bridge_deliveries")
        required_delivery_columns = {
            "dest_host",
            "dest_agent",
            "dest_project_key",
            "status",
            "body_sha256",
            "attestation_version",
            "message_revision",
            "supersedes_message_id",
        }
        missing_delivery_columns = sorted(required_delivery_columns - delivery_columns)
        if missing_delivery_columns:
            checks.append(_result("delivery_schema", False, f"missing delivery columns: {', '.join(missing_delivery_columns)}"))
        else:
            active_rows = connection.execute(
                "SELECT dest_host, dest_agent, dest_project_key, status "
                "FROM agent_mail_bridge_deliveries WHERE lower(dest_host)=lower(?) "
                "AND lower(status) IN ('queued','retry','sending')",
                (peer_host,),
            ).fetchall()
            bad_active = [
                row
                for row in active_rows
                if not str(row[2] or "").strip() or str(row[1] or "").casefold() in STALE_AGENTS
            ]
            checks.append(
                _result(
                    "active_delivery_routing",
                    not bad_active,
                    "all active deliveries have non-stale routed identities and projects"
                    if not bad_active
                    else f"active stale/blank deliveries require quarantine: {len(bad_active)}",
                    active_rows=len(active_rows),
                )
            )
    except (OSError, sqlite3.Error, UnicodeError) as exc:
        checks.append(_result("database_read", False, f"read-only verification failed: {type(exc).__name__}"))
    finally:
        connection.close()

    return {"ok": all(bool(item.get("ok")) for item in checks), "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--identity-env", type=Path, required=True)
    parser.add_argument("--token-env", type=Path, required=True)
    parser.add_argument("--host-key", required=True)
    parser.add_argument("--peer-host", default="Primus")
    parser.add_argument("--peer-project", default="/home/antigravity")
    parser.add_argument("--peer-api-url", default="http://127.0.0.1:18766/api/")
    args = parser.parse_args()
    result = verify(
        database=args.database,
        identity_env=args.identity_env,
        token_env=args.token_env,
        host_key=args.host_key,
        peer_host=args.peer_host,
        peer_project=args.peer_project,
        peer_api_url=args.peer_api_url,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
