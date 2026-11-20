from __future__ import annotations

import asyncio
import hashlib
import sqlite3

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

import mcp_agent_mail.app as app_module
from mcp_agent_mail.app import build_mcp_server
from mcp_agent_mail.db import ensure_schema, get_session
from mcp_agent_mail.models import AgentMailBridgeDelivery


def _tool_list(result):
    if hasattr(result, "structured_content") and isinstance(result.structured_content, dict):
        structured = result.structured_content.get("result")
        if isinstance(structured, list):
            return structured
    return list(getattr(result, "data", result))


async def _upsert_peer(client, host_key, token, project_key="/peer-project"):
    await client.call_tool(
        "upsert_bridge_peer",
        {
            "host_key": host_key,
            "api_url": "http://peer.invalid/api/",
            "project_key": project_key,
            "auth_token": token,
            "bridge_protocol_version": 1,
        },
    )


@pytest.mark.parametrize(
    ("host_key", "agent_name", "expected_error"),
    [
        ("Primus", "SwiftCat", "STALE_BRIDGE_IDENTITY"),
        ("Primus", "", "INVALID_BRIDGE_IDENTITY"),
        ("", "RoseStone", "INVALID_BRIDGE_IDENTITY"),
    ],
)
def test_bridge_identity_rejects_stale_or_blank_before_delivery(host_key, agent_name, expected_error):
    """Stale and incomplete peer identities fail before network or persistence."""
    with pytest.raises(app_module.ToolExecutionError) as exc_info:
        app_module._validate_bridge_peer_identity(
            host_key,
            agent_name,
            require_canonical=True,
            context="test bridge identity",
        )

    assert exc_info.value.error_type == expected_error


@pytest.mark.parametrize(
    ("host_key", "project_key", "expected"),
    [
        ("Datacube", "/home/antigravity", "/home/Antigravity"),
        ("Primus", "/home/Antigravity", "/home/antigravity"),
        ("Primus", "/srv/other-project", "/srv/other-project"),
    ],
)
def test_bridge_project_keys_are_canonical_per_host(host_key, project_key, expected):
    """Federated project paths are host-local and must preserve case semantics."""
    assert app_module._canonicalize_bridge_project_key(host_key, project_key) == expected


def test_outbound_peer_token_never_uses_unscoped_fallback(monkeypatch):
    """A generic token must not cross an accidentally renamed peer boundary."""
    monkeypatch.delenv("AGENT_MAIL_BRIDGE_PEER_TOKEN_PRIMUS", raising=False)
    monkeypatch.delenv("ANTIGRAVITY_AGENT_MAIL_BRIDGE_PEER_TOKEN_PRIMUS", raising=False)
    monkeypatch.setenv("AGENT_MAIL_BRIDGE_PEER_TOKEN", "wrong-peer-token")
    monkeypatch.setenv("ANTIGRAVITY_AGENT_MAIL_BRIDGE_PEER_TOKEN", "another-wrong-peer-token")

    assert app_module._bridge_outbound_peer_token("Primus") == ""


@pytest.mark.asyncio
async def test_same_thread_participant_lookup_survives_session_scope(isolated_env, monkeypatch):
    """Thread contact auto-allow must query recipients inside a live DB session."""
    monkeypatch.setenv("AGENT_MAIL_HOST_KEY", "Datacube")
    monkeypatch.setenv("CONTACT_ENFORCEMENT_ENABLED", "true")
    monkeypatch.setenv("MESSAGING_AUTO_HANDSHAKE_ON_BLOCK", "false")
    monkeypatch.setenv("CONTACT_AUTO_RETRY_ENABLED", "false")
    monkeypatch.setenv("CONTACT_AUTO_TTL_SECONDS", "0")
    from mcp_agent_mail.config import clear_settings_cache

    clear_settings_cache()
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/thread-contacts"})
        for name in ("GreenCastle", "BlueLake"):
            await client.call_tool(
                "register_agent",
                {
                    "project_key": "Thread-Contacts",
                    "program": "workx",
                    "model": "local",
                    "name": name,
                },
            )
        await client.call_tool(
            "set_contact_policy",
            {"project_key": "Thread-Contacts", "agent_name": "BlueLake", "policy": "open"},
        )
        seed = await client.call_tool(
            "send_message",
            {
                "project_key": "Thread-Contacts",
                "sender_name": "GreenCastle",
                "to": ["BlueLake"],
                "subject": "Seed",
                "body_md": "thread seed",
            },
        )
        payload = (seed.data.get("deliveries") or [])[0]["payload"]
        thread_id = payload.get("thread_id") or payload["id"]
        await client.call_tool(
            "set_contact_policy",
            {"project_key": "Thread-Contacts", "agent_name": "BlueLake", "policy": "contacts_only"},
        )
        followup = await client.call_tool(
            "send_message",
            {
                "project_key": "Thread-Contacts",
                "sender_name": "GreenCastle",
                "to": ["BlueLake"],
                "subject": "Thread follow-up",
                "body_md": "participant lookup remains available",
                "thread_id": str(thread_id),
            },
        )
        assert (followup.data.get("deliveries") or [])[0]["payload"]["subject"] == "Thread follow-up"


def test_parse_mcp_tool_payload_rejects_remote_tool_error():
    raw = '{"jsonrpc":"2.0","id":"1","result":{"isError":true,"content":[{"type":"text","text":"peer rejected bridge_receive_message"}]}}'

    with pytest.raises(RuntimeError, match="MCP tool error"):
        app_module._parse_mcp_tool_payload(raw)


@pytest.mark.asyncio
async def test_name_at_host_queues_remote_delivery(isolated_env, monkeypatch):
    monkeypatch.setenv("AGENT_MAIL_HOST_KEY", "Datacube")
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/home/Antigravity"})
        await client.call_tool(
            "register_agent",
            {
                "project_key": "/home/Antigravity",
                "program": "codex",
                "model": "gpt-5",
                "name": "PurpleCanyon",
            },
        )

        sent = await client.call_tool(
            "send_message",
            {
                "project_key": "/home/Antigravity",
                "sender_name": "PurpleCanyon",
                "to": ["RoseStone@Primus"],
                "subject": "Bridge smoke",
                "body_md": "queued body",
                "thread_id": "bridge-smoke",
                "topic": "cross-system",
            },
        )

        assert sent.data["remote_delivery_state"] == "queued_remote"
        bridge = sent.data["bridge_deliveries"][0]
        assert bridge["source_host"] == "Datacube"
        assert bridge["dest_host"] == "Primus"
        assert bridge["dest_agent"] == "RoseStone"
        assert bridge["status"] == "queued"

        identities = await client.call_tool(
            "list_identities",
            {"project_key": "/home/Antigravity", "include_remote": True},
        )
        assert any(
            item["host_key"] == "Primus" and item["agent_name"] == "RoseStone"
            for item in identities.data["remote"]
        )


@pytest.mark.asyncio
async def test_remote_send_requires_explicit_host_pin(isolated_env, monkeypatch):
    monkeypatch.delenv("AGENT_MAIL_HOST_KEY", raising=False)
    monkeypatch.delenv("ANTIGRAVITY_PLATFORM_ID", raising=False)
    monkeypatch.delenv("AG_PLATFORM_ID", raising=False)
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/home/Antigravity"})
        await client.call_tool(
            "register_agent",
            {
                "project_key": "/home/Antigravity",
                "program": "codex",
                "model": "gpt-5",
                "name": "PurpleCanyon",
            },
        )
        failed = await client.call_tool_mcp(
            "send_message",
            {
                "project_key": "/home/Antigravity",
                "sender_name": "PurpleCanyon",
                "to": ["RoseStone@Primus"],
                "subject": "Implicit identity must fail",
                "body_md": "The host pin is intentionally absent.",
            },
        )
        assert failed.isError is True
        error_text = " ".join(getattr(content, "text", "") for content in failed.content)
        assert "outbound bridge source requires a non-empty host" in error_text


@pytest.mark.asyncio
async def test_reply_to_bridged_message_routes_to_remote_sender_and_preserves_thread(isolated_env, monkeypatch):
    monkeypatch.setenv("AGENT_MAIL_HOST_KEY", "Datacube")
    calls = []

    async def fake_call_bridge_peer_tool(**kwargs):
        calls.append(kwargs)
        return {"id": 4243}

    monkeypatch.setattr(app_module, "_call_bridge_peer_tool", fake_call_bridge_peer_tool)
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/home/Antigravity"})
        await client.call_tool(
            "register_agent",
            {
                "project_key": "/home/Antigravity",
                "program": "codex",
                "model": "gpt-5",
                "name": "PurpleCanyon",
            },
        )
        await client.call_tool(
            "upsert_bridge_peer",
            {
                "host_key": "Primus",
                "auth_token": "test-primus-peer-token",
                "api_url": "http://127.0.0.1:18766/api/",
                "project_key": "/peer-default-project",
                "bridge_protocol_version": 1,
            },
        )
        inbound = await client.call_tool(
            "bridge_receive_message",
            {
                "project_key": "/home/Antigravity",
                "to": ["PurpleCanyon"],
                "subject": "Remote work order",
                "body_md": "Please verify the bridge reply path.",
                "source_host": "Primus",
                "peer_token": "test-primus-peer-token",
                "source_agent": "RoseStone",
                "source_project_key": "/remote-origin-project",
                "external_message_id": "primus-42",
                "external_thread_id": "primus-thread-42",
                "bridge_dedupe_id": "primus:42:datacube",
            },
        )

        reply_body = "Bridge reply repaired: human-readable evidence is attached to the original thread."
        replied = await client.call_tool(
            "reply_message",
            {
                "project_key": "/home/Antigravity",
                "message_id": inbound.data["id"],
                "sender_name": "PurpleCanyon",
                "body_md": reply_body,
            },
        )

        assert replied.data["reply_to"] == inbound.data["id"]
        assert replied.data["thread_id"] == "primus-thread-42"
        assert replied.data["remote_delivery_state"] == "queued_remote"
        bridge = replied.data["bridge_deliveries"][0]
        assert bridge["dest_host"] == "Primus"
        assert bridge["dest_agent"] == "RoseStone"
        assert bridge["dest_project_key"] == "/remote-origin-project"

        flushed = await client.call_tool("flush_bridge_deliveries", {"dest_host": "Primus", "limit": 1})
        assert flushed.data["delivered"] == 1
        assert calls[0]["tool_name"] == "bridge_receive_message"
        args = calls[0]["arguments"]
        assert args["to"] == ["RoseStone"]
        assert args["project_key"] == "/remote-origin-project"
        assert args["body_md"] == reply_body
        assert args["external_thread_id"] == "primus-thread-42"


@pytest.mark.asyncio
async def test_bridge_receive_without_remote_project_fails_closed(isolated_env, monkeypatch):
    monkeypatch.setenv("AGENT_MAIL_HOST_KEY", "Datacube")
    bridge_calls = []
    error_types = []

    def capture_tool_error(_tool_name, error):
        error_types.append(error.error_type)

    monkeypatch.setattr(app_module, "_record_tool_error", capture_tool_error)
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/home/Antigravity"})
        await client.call_tool(
            "register_agent",
            {
                "project_key": "/home/Antigravity",
                "program": "codex",
                "model": "gpt-5",
                "name": "PurpleCanyon",
            },
        )
        await client.call_tool(
            "upsert_bridge_peer",
            {
                "host_key": "Primus",
                "auth_token": "test-primus-peer-token",
                "api_url": "http://127.0.0.1:18766/api/",
                "project_key": "/peer-default-project",
                "bridge_protocol_version": 1,
            },
        )
        failed = await client.call_tool_mcp(
            "bridge_receive_message",
            {
                "project_key": "/home/Antigravity",
                "to": ["PurpleCanyon"],
                "subject": "Remote work order without project provenance",
                "body_md": "This input intentionally omits source_project_key.",
                "source_host": "Primus",
                "peer_token": "test-primus-peer-token",
                "source_agent": "RoseStone",
                "external_message_id": "primus-missing-project",
                "external_thread_id": "primus-thread-missing-project",
                "bridge_dedupe_id": "primus:missing-project:datacube",
            },
        )
        error_text = " ".join(getattr(content, "text", "") for content in failed.content)

        assert failed.isError is True
        assert error_types == ["INVALID_BRIDGE_PROJECT"]
        assert "source project key" in error_text
        assert bridge_calls == []


@pytest.mark.asyncio
async def test_bridge_receive_preserves_remote_provenance(isolated_env, monkeypatch):
    monkeypatch.setenv("AGENT_MAIL_HOST_KEY", "Primus")
    server = build_mcp_server()
    async with Client(server) as client:
        await _upsert_peer(client, "Datacube", "test-datacube-peer-token")
        await client.call_tool("ensure_project", {"human_key": "/home/antigravity"})
        registration = await client.call_tool(
            "register_agent",
            {
                "project_key": "/home/antigravity",
                "program": "codex",
                "model": "gpt-5",
                "name": "RoseStone",
            },
        )
        token = registration.data["registration_token"]

        received = await client.call_tool(
            "bridge_receive_message",
            {
                "project_key": "/home/antigravity",
                "to": ["RoseStone"],
                "subject": "Inbound",
                "body_md": "from datacube",
                "source_host": "Datacube",
                "peer_token": "test-datacube-peer-token",
                "source_agent": "PurpleCanyon",
                "source_project_key": "/home/Antigravity",
                "external_message_id": "11192",
                "external_thread_id": "thread-1",
                "bridge_dedupe_id": "datacube:11192:swiftcat",
                "topic": "cross-system",
            },
        )

        assert received.data["from"] == "PurpleCanyon@Datacube"
        assert received.data["external_sender"]["host"] == "Datacube"
        assert received.data["external_sender"]["name"] == "PurpleCanyon"
        assert received.data["external_sender"]["project_key"] == "/home/Antigravity"
        assert received.data["attestation"]["status"] == "legacy_unattested"
        assert received.data["attestation"]["version"] == 0

        inbox = await client.call_tool(
            "fetch_inbox",
            {
                "project_key": "/home/antigravity",
                "agent_name": "RoseStone",
                "registration_token": token,
                "include_bodies": True,
                "topic": "cross-system",
            },
        )
        inbox_items = _tool_list(inbox)
        assert inbox_items[0]["from"] == "PurpleCanyon@Datacube"
        assert inbox_items[0]["bridge_agent"] == "RoseStone"
        assert inbox_items[0]["external_sender"]["name"] == "PurpleCanyon"
        assert inbox_items[0]["external_sender"]["host"] == "Datacube"

        duplicate = await client.call_tool(
            "bridge_receive_message",
            {
                "project_key": "/home/antigravity",
                "to": ["RoseStone"],
                "subject": "Inbound",
                "body_md": "from datacube",
                "source_host": "Datacube",
                "peer_token": "test-datacube-peer-token",
                "source_agent": "PurpleCanyon",
                "source_project_key": "/home/Antigravity",
                "external_message_id": "11192",
                "bridge_dedupe_id": "datacube:11192:swiftcat",
            },
        )
        assert duplicate.data["duplicate"] is True
        assert duplicate.data["id"] == received.data["id"]


@pytest.mark.asyncio
async def test_bridge_receive_attested_digest_is_verified_and_idempotent(isolated_env, monkeypatch):
    monkeypatch.setenv("AGENT_MAIL_HOST_KEY", "Datacube")
    error_types = []

    def capture_tool_error(_tool_name, error):
        error_types.append(error.error_type)

    monkeypatch.setattr(app_module, "_record_tool_error", capture_tool_error)
    server = build_mcp_server()
    async with Client(server) as client:
        await _upsert_peer(client, "Datacube", "test-datacube-peer-token")
        await client.call_tool("ensure_project", {"human_key": "/home/antigravity"})
        await client.call_tool(
            "register_agent",
            {
                "project_key": "/home/antigravity",
                "program": "codex",
                "model": "gpt-5",
                "name": "RoseStone",
            },
        )
        body = "exact UTF-8 body: cafe\u0301"
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        args = {
            "project_key": "/home/antigravity",
            "to": ["RoseStone"],
            "subject": "Attested inbound",
            "body_md": body,
            "source_host": "Datacube",
            "peer_token": "test-datacube-peer-token",
            "source_agent": "PurpleCanyon",
            "source_project_key": "/home/Antigravity",
            "external_message_id": "attested-100",
            "bridge_dedupe_id": "datacube:attested-100:swiftcat",
            "body_sha256": digest,
            "attestation_version": 1,
            "message_revision": 1,
        }
        received = await client.call_tool("bridge_receive_message", args)
        assert received.data["attestation"] == {
            "status": "attested",
            "version": 1,
            "body_sha256": digest,
            "revision": 1,
            "supersedes_message_id": None,
        }

        duplicate = await client.call_tool("bridge_receive_message", args)
        assert duplicate.data["duplicate"] is True
        assert duplicate.data["id"] == received.data["id"]

        revision_body = "attested revision two"
        non_monotonic = await client.call_tool_mcp(
            "bridge_receive_message",
            {
                **args,
                "external_message_id": "attested-101",
                "bridge_dedupe_id": "datacube:attested-101:swiftcat",
                "body_md": revision_body,
                "body_sha256": hashlib.sha256(revision_body.encode("utf-8")).hexdigest(),
                "message_revision": 1,
                "supersedes_message_id": "attested-100",
            },
        )
        non_monotonic_error = " ".join(getattr(content, "text", "") for content in non_monotonic.content)
        assert non_monotonic.isError is True
        assert "must be greater than the superseded message revision" in non_monotonic_error

        revised = await client.call_tool(
            "bridge_receive_message",
            {
                **args,
                "external_message_id": "attested-101",
                "bridge_dedupe_id": "datacube:attested-101:swiftcat",
                "body_md": revision_body,
                "body_sha256": hashlib.sha256(revision_body.encode("utf-8")).hexdigest(),
                "message_revision": 2,
                "supersedes_message_id": "attested-100",
            },
        )
        assert revised.data["attestation"]["revision"] == 2
        assert revised.data["attestation"]["supersedes_message_id"] == "attested-100"

        same_revision = await client.call_tool(
            "bridge_receive_message",
            {
                **args,
                "external_message_id": "attested-102",
                "bridge_dedupe_id": "datacube:attested-102:swiftcat",
                "body_md": revision_body,
                "body_sha256": hashlib.sha256(revision_body.encode("utf-8")).hexdigest(),
                "message_revision": 2,
                "supersedes_message_id": "attested-100",
            },
        )
        assert same_revision.data["duplicate"] is True
        assert same_revision.data["id"] == revised.data["id"]

        conflicting_revision_body = "conflicting revision two"
        conflicting_revision = await client.call_tool_mcp(
            "bridge_receive_message",
            {
                **args,
                "external_message_id": "attested-103",
                "bridge_dedupe_id": "datacube:attested-103:swiftcat",
                "body_md": conflicting_revision_body,
                "body_sha256": hashlib.sha256(conflicting_revision_body.encode("utf-8")).hexdigest(),
                "message_revision": 2,
                "supersedes_message_id": "attested-100",
            },
        )
        conflicting_error = " ".join(getattr(content, "text", "") for content in conflicting_revision.content)
        assert conflicting_revision.isError is True
        assert "different message at this revision" in conflicting_error

        late_revision_body = "arrives before its parent"
        late_revision = await client.call_tool(
            "bridge_receive_message",
            {
                **args,
                "external_message_id": "late-revision",
                "bridge_dedupe_id": "datacube:late-revision:swiftcat",
                "body_md": late_revision_body,
                "body_sha256": hashlib.sha256(late_revision_body.encode("utf-8")).hexdigest(),
                "message_revision": 2,
                "supersedes_message_id": "late-root",
            },
        )
        late_parent_body = "parent arrives later"
        late_parent = await client.call_tool(
            "bridge_receive_message",
            {
                **args,
                "external_message_id": "late-root",
                "bridge_dedupe_id": "datacube:late-root:swiftcat",
                "body_md": late_parent_body,
                "body_sha256": hashlib.sha256(late_parent_body.encode("utf-8")).hexdigest(),
                "message_revision": 1,
                "supersedes_message_id": None,
            },
        )
        assert late_revision.data["attestation"]["revision"] == 2
        assert late_parent.data["attestation"]["revision"] == 1

        changed_body = "same key, different body"
        changed = await client.call_tool_mcp(
            "bridge_receive_message",
            {
                **args,
                "body_md": changed_body,
                "body_sha256": hashlib.sha256(changed_body.encode("utf-8")).hexdigest(),
            },
        )
        changed_error = " ".join(getattr(content, "text", "") for content in changed.content)
        assert changed.isError is True
        assert "different body attestation or revision data" in changed_error

        bad_digest = await client.call_tool_mcp(
            "bridge_receive_message",
            {
                **args,
                "bridge_dedupe_id": "datacube:bad-digest:swiftcat",
                "body_sha256": "0" * 64,
            },
        )
        bad_error = " ".join(getattr(content, "text", "") for content in bad_digest.content)
        assert bad_digest.isError is True
        assert "does not match the exact UTF-8 body_md payload" in bad_error
        assert error_types == [
            "NON_MONOTONIC_MESSAGE_REVISION",
            "CONFLICTING_MESSAGE_REVISION",
            "BRIDGE_DEDUPE_ATTESTATION_MISMATCH",
            "BODY_DIGEST_MISMATCH",
        ]


@pytest.mark.asyncio
async def test_schema_migrates_existing_sqlite_bridge_and_message_tables(isolated_env, tmp_path):
    db_path = tmp_path / "test.sqlite3"
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(
            """
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY,
                project_id INTEGER NOT NULL,
                sender_id INTEGER NOT NULL,
                thread_id VARCHAR(128),
                subject VARCHAR(512) NOT NULL,
                body_md TEXT NOT NULL,
                importance VARCHAR(16) NOT NULL DEFAULT 'normal',
                ack_required BOOLEAN NOT NULL DEFAULT 0,
                created_ts DATETIME NOT NULL,
                attachments JSON NOT NULL DEFAULT '[]'
            );
            CREATE TABLE agent_mail_bridge_deliveries (
                id INTEGER PRIMARY KEY,
                bridge_id VARCHAR(128) NOT NULL,
                local_message_id INTEGER,
                source_host VARCHAR(64) NOT NULL,
                source_agent VARCHAR(128) NOT NULL,
                source_project_key VARCHAR(255) NOT NULL DEFAULT '',
                dest_host VARCHAR(64) NOT NULL,
                dest_agent VARCHAR(128) NOT NULL,
                dest_project_key VARCHAR(255) NOT NULL DEFAULT '',
                remote_message_id VARCHAR(128),
                status VARCHAR(32) NOT NULL DEFAULT 'queued',
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error VARCHAR(2048) NOT NULL DEFAULT '',
                created_ts DATETIME NOT NULL,
                updated_ts DATETIME NOT NULL,
                delivered_ts DATETIME,
                read_ts DATETIME,
                ack_ts DATETIME,
                completed_ts DATETIME
            );
            """
        )
        connection.commit()
    finally:
        connection.close()

    await ensure_schema()
    async with get_session() as session:
        message_columns = {row[1] for row in (await session.execute(text("PRAGMA table_info(messages)"))).all()}
        delivery_columns = {
            row[1] for row in (await session.execute(text("PRAGMA table_info(agent_mail_bridge_deliveries)"))).all()
        }
        peer_columns = {row[1] for row in (await session.execute(text("PRAGMA table_info(agent_mail_peers)"))).all()}
        triggers = {
            row[0]
            for row in (
                await session.execute(
                    text(
                        "SELECT name FROM sqlite_master "
                        "WHERE type = 'trigger' "
                        "AND name IN ('messages_legacy_attestation_immutable', 'messages_attested_immutable')"
                    )
                )
            ).all()
        }
    assert {
        "body_sha256",
        "attestation_version",
        "revision",
        "supersedes_message_id",
        "bridge_dedupe_scope",
        "bridge_revision_scope",
    }.issubset(message_columns)
    assert {"body_sha256", "attestation_version", "message_revision", "supersedes_message_id"}.issubset(
        delivery_columns
    )
    assert "bridge_protocol_version" in peer_columns
    assert triggers == {"messages_legacy_attestation_immutable", "messages_attested_immutable"}


@pytest.mark.asyncio
async def test_bridge_revision_lineage_is_scoped_to_project_and_remote_provenance(isolated_env, monkeypatch):
    monkeypatch.setenv("AGENT_MAIL_HOST_KEY", "Datacube")
    server = build_mcp_server()
    async with Client(server) as client:
        await _upsert_peer(client, "Datacube", "test-datacube-peer-token")
        for project_key in ("/home/alpha", "/home/beta"):
            await client.call_tool("ensure_project", {"human_key": project_key})
            await client.call_tool(
                "register_agent",
                {
                    "project_key": project_key,
                    "program": "codex",
                    "model": "gpt-5",
                    "name": "RoseStone",
                },
            )

        first_body = "first project revision one"
        await client.call_tool(
            "bridge_receive_message",
            {
                "project_key": "/home/alpha",
                "to": ["RoseStone"],
                "subject": "Alpha history",
                "body_md": first_body,
                "source_host": "Datacube",
                "peer_token": "test-datacube-peer-token",
                "source_agent": "PurpleCanyon",
                "source_project_key": "/remote/alpha",
                "external_message_id": "shared-12655",
                "bridge_dedupe_id": "datacube:alpha:shared-12655",
                "body_sha256": hashlib.sha256(first_body.encode("utf-8")).hexdigest(),
                "attestation_version": 1,
                "message_revision": 1,
            },
        )

        beta_body = "beta legacy correction"
        beta = await client.call_tool(
            "bridge_receive_message",
            {
                "project_key": "/home/beta",
                "to": ["RoseStone"],
                "subject": "Beta correction",
                "body_md": beta_body,
                "source_host": "Datacube",
                "peer_token": "test-datacube-peer-token",
                "source_agent": "PurpleCanyon",
                "source_project_key": "/remote/beta",
                "external_message_id": "shared-12660",
                "bridge_dedupe_id": "datacube:beta:shared-12660",
                "body_sha256": hashlib.sha256(beta_body.encode("utf-8")).hexdigest(),
                "attestation_version": 1,
                "message_revision": 1,
                "supersedes_message_id": "shared-12655",
            },
        )

        assert beta.data["attestation"]["revision"] == 1
        assert beta.data["attestation"]["supersedes_message_id"] == "shared-12655"
        assert beta.data["external_sender"] == {
            "host": "Datacube",
            "name": "PurpleCanyon",
            "message_id": "shared-12660",
            "project_key": "/remote/beta",
            "thread_id": None,
            "bridge_dedupe_id": "datacube:beta:shared-12660",
        }


@pytest.mark.asyncio
async def test_bridge_dedupe_scope_prevents_cross_project_remote_collision(isolated_env, monkeypatch):
    monkeypatch.setenv("AGENT_MAIL_HOST_KEY", "Datacube")
    server = build_mcp_server()
    async with Client(server) as client:
        await _upsert_peer(client, "Datacube", "test-datacube-peer-token")
        for project_key in ("/home/alpha", "/home/beta"):
            await client.call_tool("ensure_project", {"human_key": project_key})
            await client.call_tool(
                "register_agent",
                {
                    "project_key": project_key,
                    "program": "codex",
                    "model": "gpt-5",
                    "name": "RoseStone",
                },
            )

        alpha_body = "alpha attested root"
        await client.call_tool(
            "bridge_receive_message",
            {
                "project_key": "/home/alpha",
                "to": ["RoseStone"],
                "subject": "Alpha root",
                "body_md": alpha_body,
                "source_host": "Datacube",
                "peer_token": "test-datacube-peer-token",
                "source_agent": "PurpleCanyon",
                "source_project_key": "/remote/alpha",
                "external_message_id": "shared-12655",
                "bridge_dedupe_id": "datacube:collision",
                "body_sha256": hashlib.sha256(alpha_body.encode("utf-8")).hexdigest(),
                "attestation_version": 1,
                "message_revision": 1,
            },
        )

        beta_body = "beta attested correction"
        beta = await client.call_tool(
            "bridge_receive_message",
            {
                "project_key": "/home/beta",
                "to": ["RoseStone"],
                "subject": "Beta correction",
                "body_md": beta_body,
                "source_host": "Datacube",
                "peer_token": "test-datacube-peer-token",
                "source_agent": "PurpleCanyon",
                "source_project_key": "/remote/beta",
                "external_message_id": "shared-12660",
                "bridge_dedupe_id": "datacube:collision",
                "body_sha256": hashlib.sha256(beta_body.encode("utf-8")).hexdigest(),
                "attestation_version": 1,
                "message_revision": 1,
                "supersedes_message_id": "shared-12655",
            },
        )

        assert beta.data["attestation"]["revision"] == 1
        assert beta.data["external_sender"]["project_key"] == "/remote/beta"
        assert beta.data["external_sender"]["message_id"] == "shared-12660"


@pytest.mark.asyncio
async def test_bridge_receive_concurrent_idempotent_retry_rereads_scoped_duplicate(isolated_env, monkeypatch):
    monkeypatch.setenv("AGENT_MAIL_HOST_KEY", "Datacube")
    server = build_mcp_server()
    original_create_message = app_module._create_message
    injected_race = False

    async def create_then_simulate_unique_race(*args, **kwargs):
        nonlocal injected_race
        if not injected_race:
            injected_race = True
            await original_create_message(*args, **kwargs)
            raise IntegrityError("INSERT", {}, Exception("UNIQUE constraint failed: messages.bridge_dedupe_id"))
        return await original_create_message(*args, **kwargs)

    async with Client(server) as client:
        await _upsert_peer(client, "Datacube", "test-datacube-peer-token")
        await client.call_tool("ensure_project", {"human_key": "/home/antigravity"})
        await client.call_tool(
            "register_agent",
            {
                "project_key": "/home/antigravity",
                "program": "codex",
                "model": "gpt-5",
                "name": "RoseStone",
            },
        )
        monkeypatch.setattr(app_module, "_create_message", create_then_simulate_unique_race)
        body = "race-safe attested inbound"
        raced = await client.call_tool(
            "bridge_receive_message",
            {
                "project_key": "/home/antigravity",
                "to": ["RoseStone"],
                "subject": "Concurrent retry",
                "body_md": body,
                "source_host": "Datacube",
                "peer_token": "test-datacube-peer-token",
                "source_agent": "PurpleCanyon",
                "source_project_key": "/remote/antigravity",
                "external_message_id": "race-12665",
                "bridge_dedupe_id": "datacube:race-12665",
                "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
                "attestation_version": 1,
                "message_revision": 1,
            },
        )

    assert raced.data["duplicate"] is True
    assert injected_race is True


@pytest.mark.asyncio
async def test_legacy_history_can_be_superseded_but_not_retroactively_attested(isolated_env, monkeypatch):
    monkeypatch.setenv("AGENT_MAIL_HOST_KEY", "Datacube")
    server = build_mcp_server()
    async with Client(server) as client:
        await _upsert_peer(client, "Datacube", "test-datacube-peer-token")
        await client.call_tool("ensure_project", {"human_key": "/home/antigravity"})
        for name in ("RoseStone", "AmberGrove"):
            await client.call_tool(
                "register_agent",
                {
                    "project_key": "/home/antigravity",
                    "program": "codex",
                    "model": "gpt-5",
                    "name": name,
                },
            )
        legacy = await client.call_tool(
            "bridge_receive_message",
            {
                "project_key": "/home/antigravity",
                "to": ["RoseStone"],
                "subject": "Legacy 12655",
                "body_md": "legacy body",
                "source_host": "Datacube",
                "peer_token": "test-datacube-peer-token",
                "source_agent": "PurpleCanyon",
                "source_project_key": "/home/Antigravity",
                "external_message_id": "12655",
                "bridge_dedupe_id": "datacube:12655:swiftcat",
            },
        )
        legacy_id = legacy.data["id"]
        assert legacy.data["attestation"]["status"] == "legacy_unattested"

        async with get_session() as session:
            await session.execute(text("UPDATE messages SET body_md = 'legacy amendment' WHERE id = :id"), {"id": legacy_id})
            await session.commit()
        async with get_session() as session:
            with pytest.raises(IntegrityError):
                await session.execute(
                    text("UPDATE messages SET attestation_version = 1 WHERE id = :id"),
                    {"id": legacy_id},
                )
            await session.rollback()
        async with get_session() as session:
            with pytest.raises(IntegrityError):
                await session.execute(
                    text(
                        "UPDATE messages SET body_sha256 = :digest, revision = 1 "
                        "WHERE id = :id"
                    ),
                    {"digest": "0" * 64, "id": legacy_id},
                )
            await session.rollback()

        superseding = await client.call_tool(
            "send_message",
            {
                "project_key": "/home/antigravity",
                "sender_name": "RoseStone",
                "to": ["RoseStone"],
                "subject": "Supersedes legacy 12655",
                "body_md": "attested replacement",
                "supersedes_message_id": legacy_id,
            },
        )
        payload = superseding.data["deliveries"][0]["payload"]
        assert payload["attestation"]["status"] == "attested"
        assert payload["attestation"]["revision"] == 1
        assert payload["attestation"]["supersedes_message_id"] == str(legacy_id)

        async with get_session() as session:
            with pytest.raises(IntegrityError):
                await session.execute(
                    text("UPDATE messages SET body_md = 'tampered' WHERE id = :id"),
                    {"id": payload["id"]},
                )
            await session.rollback()


@pytest.mark.asyncio
async def test_bridge_flush_refuses_source_body_mutated_after_queue(isolated_env, monkeypatch):
    monkeypatch.setenv("AGENT_MAIL_HOST_KEY", "Datacube")
    bridge_calls = []

    async def fake_call_bridge_peer_tool(**kwargs):
        bridge_calls.append(kwargs)
        return {"id": 99}

    monkeypatch.setattr(app_module, "_call_bridge_peer_tool", fake_call_bridge_peer_tool)
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/home/Antigravity"})
        await client.call_tool(
            "register_agent",
            {
                "project_key": "/home/Antigravity",
                "program": "codex",
                "model": "gpt-5",
                "name": "PurpleCanyon",
            },
        )
        await client.call_tool(
            "upsert_bridge_peer",
            {
                "host_key": "Primus",
                "auth_token": "test-primus-peer-token",
                "api_url": "http://127.0.0.1:18766/api/",
                "project_key": "/home/antigravity",
                "bridge_protocol_version": 1,
            },
        )
        sent = await client.call_tool(
            "send_message",
            {
                "project_key": "/home/Antigravity",
                "sender_name": "PurpleCanyon",
                "to": ["RoseStone@Primus"],
                "subject": "Queued body mutation",
                "body_md": "original queued body",
            },
        )
        delivery = sent.data["bridge_deliveries"][0]
        async with get_session() as session:
            # Simulate a pre-trigger or out-of-band modification. Production
            # attested rows are protected by the trigger tested above.
            await session.execute(text("DROP TRIGGER messages_attested_immutable"))
            await session.execute(
                text("UPDATE messages SET body_md = 'mutated after queue' WHERE id = :id"),
                {"id": delivery["local_message_id"]},
            )
            await session.commit()

        flushed = await client.call_tool("flush_bridge_deliveries", {"limit": 5})
        assert flushed.data["failed"] == 1
        assert bridge_calls == []
        listed = await client.call_tool("list_bridge_deliveries", {"status": "failed"})
        failed = _tool_list(listed)[0]
        assert failed["bridge_id"] == delivery["bridge_id"]
        assert failed["last_error"] == "source message attestation no longer matches the queued bridge snapshot"


@pytest.mark.asyncio
async def test_bridge_flush_blocks_attested_delivery_for_legacy_peer(isolated_env, monkeypatch):
    monkeypatch.setenv("AGENT_MAIL_HOST_KEY", "Datacube")
    bridge_calls = []

    async def fake_call_bridge_peer_tool(**kwargs):
        bridge_calls.append(kwargs)
        return {"id": 99}

    monkeypatch.setattr(app_module, "_call_bridge_peer_tool", fake_call_bridge_peer_tool)
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/home/Antigravity"})
        await client.call_tool(
            "register_agent",
            {
                "project_key": "/home/Antigravity",
                "program": "codex",
                "model": "gpt-5",
                "name": "PurpleCanyon",
            },
        )
        # The default protocol version is deliberately legacy/unknown.
        await client.call_tool(
            "upsert_bridge_peer",
            {
                "host_key": "Primus",
                "auth_token": "test-primus-peer-token",
                "api_url": "http://127.0.0.1:18766/api/",
                "project_key": "/home/antigravity",
            },
        )
        await client.call_tool(
            "send_message",
            {
                "project_key": "/home/Antigravity",
                "sender_name": "PurpleCanyon",
                "to": ["RoseStone@Primus"],
                "subject": "Attested must not downgrade",
                "body_md": "attested body",
            },
        )

        flushed = await client.call_tool("flush_bridge_deliveries", {"limit": 5})

    assert flushed.data["blocked"] == 1
    assert bridge_calls == []
    assert flushed.data["results"][0]["last_error"] == "bridge peer does not support body attestation"


@pytest.mark.asyncio
async def test_flush_migrated_legacy_delivery_uses_legacy_wire_shape(isolated_env, monkeypatch):
    monkeypatch.setenv("AGENT_MAIL_HOST_KEY", "Datacube")
    bridge_calls = []

    async def fake_call_bridge_peer_tool(**kwargs):
        bridge_calls.append(kwargs)
        return {"id": 99}

    monkeypatch.setattr(app_module, "_call_bridge_peer_tool", fake_call_bridge_peer_tool)
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/home/Antigravity"})
        await client.call_tool(
            "register_agent",
            {
                "project_key": "/home/Antigravity",
                "program": "codex",
                "model": "gpt-5",
                "name": "RoseStone",
            },
        )
        await client.call_tool(
            "register_agent",
            {
                "project_key": "/home/Antigravity",
                "program": "codex",
                "model": "gpt-5",
                "name": "RoseStone",
            },
        )
        await client.call_tool(
            "upsert_bridge_peer",
            {
                "host_key": "Primus",
                "auth_token": "test-primus-peer-token",
                "api_url": "http://127.0.0.1:18766/api/",
                "project_key": "/home/antigravity",
            },
        )
        legacy = await client.call_tool(
            "bridge_receive_message",
            {
                "project_key": "/home/Antigravity",
                "to": ["RoseStone"],
                "subject": "Migrated legacy source",
                "body_md": "legacy body",
                "source_host": "Primus",
                "peer_token": "test-primus-peer-token",
                "source_agent": "RoseStone",
                "source_project_key": "/home/primus",
                "external_message_id": "legacy-1",
                "bridge_dedupe_id": "primus:legacy-1",
            },
        )
        async with get_session() as session:
            session.add(
                AgentMailBridgeDelivery(
                    bridge_id="migrated-legacy-bridge",
                    local_message_id=legacy.data["id"],
                    source_host="Datacube",
                    source_agent="PurpleCanyon",
                    source_project_key="/home/Antigravity",
                    dest_host="Primus",
                    dest_agent="RoseStone",
                    dest_project_key="/home/antigravity",
                    status="queued",
                )
            )
            await session.commit()

        flushed = await client.call_tool("flush_bridge_deliveries", {"limit": 5})

    assert flushed.data["delivered"] == 1
    assert flushed.data["results"][0]["attestation"]["status"] == "legacy_unattested"
    wire_args = bridge_calls[0]["arguments"]
    assert not {"body_sha256", "attestation_version", "message_revision", "supersedes_message_id"}.intersection(
        wire_args
    )


@pytest.mark.asyncio
async def test_bridge_rejects_remote_export_of_cross_origin_supersession(isolated_env, monkeypatch):
    monkeypatch.setenv("AGENT_MAIL_HOST_KEY", "Datacube")
    bridge_calls = []

    async def fake_call_bridge_peer_tool(**kwargs):
        bridge_calls.append(kwargs)
        return {"id": 99}

    monkeypatch.setattr(app_module, "_call_bridge_peer_tool", fake_call_bridge_peer_tool)
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/home/Antigravity"})
        for name in ("RoseStone", "PurpleCanyon", "AmberGrove"):
            await client.call_tool(
                "register_agent",
                {
                    "project_key": "/home/Antigravity",
                    "program": "codex",
                    "model": "gpt-5",
                    "name": name,
                },
            )
        await client.call_tool(
            "upsert_bridge_peer",
            {
                "host_key": "Primus",
                "auth_token": "test-primus-peer-token",
                "api_url": "http://127.0.0.1:18766/api/",
                "project_key": "/home/primus",
                "bridge_protocol_version": 1,
            },
        )
        inbound_body = "remote attested root"
        inbound = await client.call_tool(
            "bridge_receive_message",
            {
                "project_key": "/home/Antigravity",
                "to": ["RoseStone"],
                "subject": "Remote root",
                "body_md": inbound_body,
                "source_host": "Primus",
                "peer_token": "test-primus-peer-token",
                "source_agent": "RoseStone",
                "source_project_key": "/home/primus",
                "external_message_id": "primus-12655",
                "bridge_dedupe_id": "primus:12655",
                "body_sha256": hashlib.sha256(inbound_body.encode("utf-8")).hexdigest(),
                "attestation_version": 1,
                "message_revision": 1,
            },
        )
        rejected = await client.call_tool_mcp(
            "send_message",
            {
                "project_key": "/home/Antigravity",
                "sender_name": "PurpleCanyon",
                "to": ["RoseStone@Primus"],
                "subject": "Local correction",
                "body_md": "local correction body",
                "supersedes_message_id": inbound.data["id"],
            },
        )

    error_text = " ".join(getattr(content, "text", "") for content in rejected.content)
    assert rejected.isError is True
    assert "cannot be bridged" in error_text
    assert bridge_calls == []
    async with get_session() as session:
        stored_supersedes = (
            await session.execute(
                text("SELECT supersedes_message_id FROM messages WHERE subject = 'Local correction'")
            )
        ).scalar_one()
    assert stored_supersedes == str(inbound.data["id"])


@pytest.mark.asyncio
async def test_bridge_receive_peer_token_enforced_when_configured(isolated_env, monkeypatch):
    monkeypatch.setenv("AGENT_MAIL_HOST_KEY", "Datacube")
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/home/antigravity"})
        await client.call_tool(
            "register_agent",
            {
                "project_key": "/home/antigravity",
                "program": "codex",
                "model": "gpt-5",
                "name": "RoseStone",
            },
        )
        await client.call_tool(
            "upsert_bridge_peer",
            {"host_key": "Datacube", "display_name": "Datacube", "auth_token": "secret"},
        )

        with pytest.raises(ToolError):
            await client.call_tool(
                "bridge_receive_message",
                {
                    "project_key": "/home/antigravity",
                    "to": ["RoseStone"],
                    "subject": "Inbound",
                    "body_md": "from datacube",
                    "source_host": "Datacube",
                    "source_agent": "PurpleCanyon",
                    "source_project_key": "/home/Antigravity",
                    "peer_token": "wrong",
                },
            )

        ok = await client.call_tool(
            "bridge_receive_message",
            {
                "project_key": "/home/antigravity",
                "to": ["RoseStone"],
                "subject": "Inbound",
                "body_md": "from datacube",
                "source_host": "Datacube",
                "source_agent": "PurpleCanyon",
                "source_project_key": "/home/Antigravity",
                "peer_token": "secret",
            },
        )
        assert ok.data["from"] == "PurpleCanyon@Datacube"


@pytest.mark.asyncio
async def test_bridge_receive_fails_closed_for_unknown_or_missing_peer_token(isolated_env):
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/home/antigravity"})
        await client.call_tool(
            "register_agent",
            {
                "project_key": "/home/antigravity",
                "program": "codex",
                "model": "gpt-5",
                "name": "PurpleCanyon",
            },
        )
        unknown_peer = await client.call_tool_mcp(
            "bridge_receive_message",
            {
                "project_key": "/home/antigravity",
                "to": ["PurpleCanyon"],
                "subject": "Unknown peer",
                "body_md": "must not be accepted",
                "source_host": "Datacube",
                "source_agent": "PurpleCanyon",
                "source_project_key": "/home/Antigravity",
            },
        )
        unknown_text = " ".join(getattr(content, "text", "") for content in unknown_peer.content)
        assert unknown_peer.isError is True
        assert "configured enabled peer" in unknown_text

        await _upsert_peer(client, "Datacube", "test-datacube-peer-token")
        missing_token = await client.call_tool_mcp(
            "bridge_receive_message",
            {
                "project_key": "/home/antigravity",
                "to": ["PurpleCanyon"],
                "subject": "Missing token",
                "body_md": "must not be accepted",
                "source_host": "Datacube",
                "source_agent": "PurpleCanyon",
                "source_project_key": "/home/Antigravity",
            },
        )
        missing_text = " ".join(getattr(content, "text", "") for content in missing_token.content)
        assert missing_token.isError is True
        assert "matching peer_token" in missing_text


@pytest.mark.asyncio
async def test_bridge_rejects_stale_and_blank_federated_identities(isolated_env, monkeypatch):
    monkeypatch.setenv("AGENT_MAIL_HOST_KEY", "Datacube")
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/home/Antigravity"})
        await client.call_tool(
            "register_agent",
            {
                "project_key": "/home/Antigravity",
                "program": "codex",
                "model": "gpt-5",
                "name": "PurpleCanyon",
            },
        )
        for target in ("SwiftCat@Primus", "SwiftCat@datacube", "@Primus"):
            failed = await client.call_tool_mcp(
                "send_message",
                {
                    "project_key": "/home/Antigravity",
                    "sender_name": "PurpleCanyon",
                    "to": [target],
                    "subject": "Invalid identity",
                    "body_md": "must be rejected before queueing",
                },
            )
            error_text = " ".join(getattr(content, "text", "") for content in failed.content)
            assert failed.isError is True
            if target.startswith("SwiftCat"):
                assert "stale agent identity" in error_text
            else:
                assert "non-empty agent and host" in error_text

        stale_inbound = await client.call_tool_mcp(
            "bridge_receive_message",
            {
                "project_key": "/home/Antigravity",
                "to": ["PurpleCanyon"],
                "subject": "Stale inbound",
                "body_md": "must be rejected before authentication",
                "source_host": "Primus",
                "source_agent": "SwiftCat",
                "source_project_key": "/home/antigravity",
            },
        )
        stale_text = " ".join(getattr(content, "text", "") for content in stale_inbound.content)
        assert stale_inbound.isError is True
        assert "stale agent identity" in stale_text


@pytest.mark.asyncio
async def test_bridge_rejects_unknown_canonical_host_before_persistence(isolated_env, monkeypatch):
    monkeypatch.setenv("AGENT_MAIL_HOST_KEY", "Datacube")
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/home/Antigravity"})
        await client.call_tool(
            "register_agent",
            {
                "project_key": "/home/Antigravity",
                "program": "workx",
                "model": "local",
                "name": "PurpleCanyon",
            },
        )
        failed = await client.call_tool_mcp(
            "bridge_receive_message",
            {
                "project_key": "/home/Antigravity",
                "to": ["PurpleCanyon"],
                "subject": "Unknown source host",
                "body_md": "must fail closed",
                "source_host": "UntrustedHost",
                "source_agent": "RoseStone",
                "source_project_key": "/home/Antigravity",
                "peer_token": "test-unknown-peer-token",
                "external_message_id": "unknown-host-1",
                "bridge_dedupe_id": "unknown-host:unknown-host-1",
            },
        )

        error_text = " ".join(getattr(content, "text", "") for content in failed.content)
        assert failed.isError is True
        assert "only canonical bridge hosts are accepted" in error_text
        async with get_session() as session:
            count = (
                await session.execute(
                    text("SELECT COUNT(*) FROM messages WHERE subject = 'Unknown source host'")
                )
            ).scalar_one()
        assert count == 0


@pytest.mark.asyncio
async def test_bridge_rejects_stale_body_provenance_before_persist(isolated_env, monkeypatch):
    monkeypatch.setenv("AGENT_MAIL_HOST_KEY", "Datacube")
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/home/Antigravity"})
        await client.call_tool(
            "register_agent",
            {
                "project_key": "/home/Antigravity",
                "program": "workx",
                "model": "local",
                "name": "PurpleCanyon",
            },
        )
        failed = await client.call_tool_mcp(
            "bridge_receive_message",
            {
                "project_key": "/home/Antigravity",
                "to": ["PurpleCanyon"],
                "subject": "Stale body provenance",
                "body_md": (
                    "<!-- ag-bridge-id: primus:stale-body -->\n\n"
                    "Cross-system request from Primus agent `SwiftCat` to `PurpleCanyon`.\n"
                ),
                "source_host": "Primus",
                "source_agent": "RoseStone",
                "source_project_key": "/home/antigravity",
                "peer_token": "test-primus-peer-token",
                "external_message_id": "stale-body-1",
                "bridge_dedupe_id": "primus:stale-body-1",
            },
        )

        error_text = " ".join(getattr(content, "text", "") for content in failed.content)
        assert failed.isError is True
        assert "stale agent identity" in error_text
        async with get_session() as session:
            count = (
                await session.execute(
                    text("SELECT COUNT(*) FROM messages WHERE subject = 'Stale body provenance'")
                )
            ).scalar_one()
        assert count == 0


@pytest.mark.asyncio
async def test_send_message_rejects_bridge_envelope_without_transport_provenance(isolated_env, monkeypatch):
    monkeypatch.setenv("AGENT_MAIL_HOST_KEY", "Datacube")
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/home/Antigravity"})
        await client.call_tool(
            "register_agent",
            {
                "project_key": "/home/Antigravity",
                "program": "workx",
                "model": "local",
                "name": "PurpleCanyon",
            },
        )
        failed = await client.call_tool_mcp(
            "send_message",
            {
                "project_key": "/home/Antigravity",
                "sender_name": "PurpleCanyon",
                "to": ["PurpleCanyon"],
                "subject": "Envelope through local path",
                "body_md": (
                    "<!-- ag-bridge-id: primus:local-path -->\n\n"
                    "Cross-system request from Primus agent `RoseStone` to `PurpleCanyon`.\n"
                ),
            },
        )

        error_text = " ".join(getattr(content, "text", "") for content in failed.content)
        assert failed.isError is True
        assert "bridge_receive_message" in error_text
        async with get_session() as session:
            count = (
                await session.execute(
                    text("SELECT COUNT(*) FROM messages WHERE subject = 'Envelope through local path'")
                )
            ).scalar_one()
        assert count == 0


@pytest.mark.asyncio
async def test_bridge_receive_reports_notification_failure_after_persist(isolated_env, monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_MAIL_HOST_KEY", "Datacube")
    monkeypatch.setenv("NOTIFICATIONS_ENABLED", "true")
    monkeypatch.setenv("NOTIFICATIONS_SIGNALS_DIR", str(tmp_path / "signals"))
    monkeypatch.setenv("NOTIFICATIONS_DEBOUNCE_MS", "0")
    from mcp_agent_mail.config import clear_settings_cache

    clear_settings_cache()

    async def fail_notification(*_args, **_kwargs):
        raise OSError("signal directory unavailable")

    monkeypatch.setattr(app_module, "emit_notification_signal", fail_notification)
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/home/Antigravity"})
        await client.call_tool(
            "register_agent",
            {
                "project_key": "/home/Antigravity",
                "program": "workx",
                "model": "local",
                "name": "PurpleCanyon",
            },
        )
        await _upsert_peer(client, "Primus", "test-primus-peer-token", project_key="/home/antigravity")
        received = await client.call_tool(
            "bridge_receive_message",
            {
                "project_key": "/home/Antigravity",
                "to": ["PurpleCanyon"],
                "subject": "Persist before notification",
                "body_md": "durable body",
                "source_host": "Primus",
                "source_agent": "RoseStone",
                "source_project_key": "/home/antigravity",
                "peer_token": "test-primus-peer-token",
                "external_message_id": "notification-failure-1",
                "bridge_dedupe_id": "primus:notification-failure-1",
            },
        )

        assert received.data["notification_state"] == "failed_after_persist"
        assert "signal directory unavailable" in received.data["notification_errors"][0]
        async with get_session() as session:
            stored = (
                await session.execute(
                    text("SELECT COUNT(*) FROM messages WHERE id = :id"),
                    {"id": received.data["id"]},
                )
            ).scalar_one()
        assert stored == 1


@pytest.mark.asyncio
async def test_flush_blocks_missing_outbound_peer_token_durably(isolated_env, monkeypatch):
    for name in (
        "AGENT_MAIL_BRIDGE_PEER_TOKEN_PRIMUS",
        "ANTIGRAVITY_AGENT_MAIL_BRIDGE_PEER_TOKEN_PRIMUS",
        "AGENT_MAIL_BRIDGE_PEER_TOKEN",
        "ANTIGRAVITY_AGENT_MAIL_BRIDGE_PEER_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    calls = []

    async def fake_call_bridge_peer_tool(**kwargs):
        calls.append(kwargs)
        return {"id": "should-not-be-called"}

    monkeypatch.setattr(app_module, "_call_bridge_peer_tool", fake_call_bridge_peer_tool)
    monkeypatch.setenv("AGENT_MAIL_HOST_KEY", "Datacube")
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/home/Antigravity"})
        await client.call_tool(
            "register_agent",
            {
                "project_key": "/home/Antigravity",
                "program": "codex",
                "model": "gpt-5",
                "name": "PurpleCanyon",
            },
        )
        await client.call_tool(
            "upsert_bridge_peer",
            {
                "host_key": "Primus",
                "api_url": "http://127.0.0.1:18766/api/",
                "project_key": "/home/antigravity",
                "auth_token": "configured-but-unavailable",
                "bridge_protocol_version": 1,
            },
        )
        queued = await client.call_tool(
            "send_message",
            {
                "project_key": "/home/Antigravity",
                "sender_name": "PurpleCanyon",
                "to": ["RoseStone@Primus"],
                "subject": "Token gate",
                "body_md": "must remain queued until token is available",
            },
        )
        assert queued.data["remote_delivery_state"] == "queued_remote"

        flushed = await client.call_tool("flush_bridge_deliveries", {"dest_host": "Primus", "limit": 1})
        assert flushed.data["blocked"] == 1
        assert flushed.data["delivered"] == 0
        assert calls == []
        assert "token is unavailable" in flushed.data["results"][0]["last_error"]

        monkeypatch.setenv("AGENT_MAIL_BRIDGE_PEER_TOKEN_PRIMUS", "configured-but-unavailable")
        recovered = await client.call_tool("flush_bridge_deliveries", {"dest_host": "Primus", "limit": 1})
        assert recovered.data["delivered"] == 1
        assert recovered.data["results"][0]["remote_message_id"] == "should-not-be-called"
        assert len(calls) == 1


@pytest.mark.asyncio
async def test_flush_transport_failure_is_visible_and_recovers(isolated_env, monkeypatch):
    calls = []

    async def flaky_call_bridge_peer_tool(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("peer unreachable during test")
        return {"id": "recovered-4242"}

    monkeypatch.setattr(app_module, "_call_bridge_peer_tool", flaky_call_bridge_peer_tool)
    monkeypatch.setenv("AGENT_MAIL_HOST_KEY", "Datacube")
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/home/Antigravity"})
        await client.call_tool(
            "register_agent",
            {
                "project_key": "/home/Antigravity",
                "program": "codex",
                "model": "gpt-5",
                "name": "PurpleCanyon",
            },
        )
        await client.call_tool(
            "upsert_bridge_peer",
            {
                "host_key": "Primus",
                "api_url": "http://127.0.0.1:18766/api/",
                "project_key": "/home/antigravity",
                "auth_token": "test-primus-peer-token",
                "bridge_protocol_version": 1,
            },
        )
        await client.call_tool(
            "send_message",
            {
                "project_key": "/home/Antigravity",
                "sender_name": "PurpleCanyon",
                "to": ["RoseStone@Primus"],
                "subject": "Recovery",
                "body_md": "transport should recover without losing the queue",
            },
        )

        first = await client.call_tool("flush_bridge_deliveries", {"dest_host": "Primus", "limit": 1})
        assert first.data["retry"] == 1
        assert first.data["results"][0]["last_error"] == "peer unreachable during test"

        second = await client.call_tool("flush_bridge_deliveries", {"dest_host": "Primus", "limit": 1})
        assert second.data["delivered"] == 1
        assert second.data["results"][0]["remote_message_id"] == "recovered-4242"
        assert len(calls) == 2


@pytest.mark.asyncio
async def test_concurrent_flushers_claim_delivery_once(isolated_env, monkeypatch):
    monkeypatch.setenv("AGENT_MAIL_HOST_KEY", "Datacube")
    calls = []
    network_entered = asyncio.Event()
    release_network = asyncio.Event()

    async def blocked_call_bridge_peer_tool(**kwargs):
        calls.append(kwargs)
        network_entered.set()
        await release_network.wait()
        return {"id": "concurrent-4242"}

    monkeypatch.setattr(app_module, "_call_bridge_peer_tool", blocked_call_bridge_peer_tool)
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/home/Antigravity"})
        await client.call_tool(
            "register_agent",
            {
                "project_key": "/home/Antigravity",
                "program": "workx",
                "model": "local",
                "name": "PurpleCanyon",
            },
        )
        await client.call_tool(
            "upsert_bridge_peer",
            {
                "host_key": "Primus",
                "auth_token": "test-primus-peer-token",
                "api_url": "http://127.0.0.1:18766/api/",
                "project_key": "/home/antigravity",
                "bridge_protocol_version": 1,
            },
        )
        await client.call_tool(
            "send_message",
            {
                "project_key": "/home/Antigravity",
                "sender_name": "PurpleCanyon",
                "to": ["RoseStone@Primus"],
                "subject": "Concurrent flush claim",
                "body_md": "only one flusher may own this delivery",
            },
        )

        first_task = asyncio.create_task(
            app_module._flush_bridge_deliveries_once(dest_host="Primus", limit=1)
        )
        await asyncio.wait_for(network_entered.wait(), timeout=2)
        second_task = asyncio.create_task(
            app_module._flush_bridge_deliveries_once(dest_host="Primus", limit=1)
        )
        await asyncio.sleep(0.05)
        release_network.set()
        first, second = await asyncio.gather(first_task, second_task)

    assert len(calls) == 1
    assert first["delivered"] + second["delivered"] == 1
    assert first["processed"] + second["processed"] == 1


@pytest.mark.asyncio
async def test_flush_requires_durable_peer_receipt_and_recovers(isolated_env, monkeypatch):
    calls = []

    async def empty_then_durable_receipt(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return {}
        return {"id": "durable-receipt-4242"}

    monkeypatch.setattr(app_module, "_call_bridge_peer_tool", empty_then_durable_receipt)
    monkeypatch.setenv("AGENT_MAIL_HOST_KEY", "Datacube")
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/home/Antigravity"})
        await client.call_tool(
            "register_agent",
            {
                "project_key": "/home/Antigravity",
                "program": "workx",
                "model": "local",
                "name": "PurpleCanyon",
            },
        )
        await client.call_tool(
            "upsert_bridge_peer",
            {
                "host_key": "Primus",
                "auth_token": "test-primus-peer-token",
                "api_url": "http://127.0.0.1:18766/api/",
                "project_key": "/home/antigravity",
                "bridge_protocol_version": 1,
            },
        )
        await client.call_tool(
            "send_message",
            {
                "project_key": "/home/Antigravity",
                "sender_name": "PurpleCanyon",
                "to": ["RoseStone@Primus"],
                "subject": "Receipt recovery",
                "body_md": "peer must return a durable receipt",
            },
        )

        first = await client.call_tool("flush_bridge_deliveries", {"dest_host": "Primus", "limit": 1})
        assert first.data["retry"] == 1
        assert first.data["delivered"] == 0
        assert first.data["results"][0]["last_error"] == (
            "peer bridge_receive_message returned no durable message receipt"
        )

        second = await client.call_tool("flush_bridge_deliveries", {"dest_host": "Primus", "limit": 1})
        assert second.data["delivered"] == 1
        assert second.data["results"][0]["remote_message_id"] == "durable-receipt-4242"
        assert len(calls) == 2


@pytest.mark.asyncio
async def test_flush_blocks_stale_queued_destination_before_network(isolated_env, monkeypatch):
    monkeypatch.setenv("AGENT_MAIL_HOST_KEY", "Datacube")
    calls = []

    async def fake_call_bridge_peer_tool(**kwargs):
        calls.append(kwargs)
        return {"id": "must-not-be-called"}

    monkeypatch.setattr(app_module, "_call_bridge_peer_tool", fake_call_bridge_peer_tool)
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/home/Antigravity"})
        await client.call_tool(
            "register_agent",
            {
                "project_key": "/home/Antigravity",
                "program": "workx",
                "model": "local",
                "name": "PurpleCanyon",
            },
        )
        await client.call_tool(
            "upsert_bridge_peer",
            {
                "host_key": "Primus",
                "auth_token": "test-primus-peer-token",
                "api_url": "http://127.0.0.1:18766/api/",
                "project_key": "/home/Antigravity",
                "bridge_protocol_version": 1,
            },
        )
        sent = await client.call_tool(
            "send_message",
            {
                "project_key": "/home/Antigravity",
                "sender_name": "PurpleCanyon",
                "to": ["RoseStone@Primus"],
                "subject": "Stale queued destination",
                "body_md": "must be quarantined before transport",
            },
        )
        assert sent.data["bridge_deliveries"][0]["dest_project_key"] == "/home/antigravity"
        bridge_id = sent.data["bridge_deliveries"][0]["bridge_id"]
        async with get_session() as session:
            await session.execute(
                text(
                    "UPDATE agent_mail_bridge_deliveries "
                    "SET dest_agent = 'SwiftCat' WHERE bridge_id = :bridge_id"
                ),
                {"bridge_id": bridge_id},
            )
            await session.commit()

        blocked = await client.call_tool("flush_bridge_deliveries", {"limit": 1})
        assert blocked.data["blocked"] == 1
        assert blocked.data["delivered"] == 0
        assert calls == []
        assert blocked.data["results"][0]["status"] == "blocked"
        assert "STALE_BRIDGE_IDENTITY" in blocked.data["results"][0]["last_error"]

        second = await client.call_tool("flush_bridge_deliveries", {"limit": 1})
        assert second.data["processed"] == 0
        assert calls == []


@pytest.mark.asyncio
async def test_flush_blocks_blank_queued_destination_project_before_network(isolated_env, monkeypatch):
    monkeypatch.setenv("AGENT_MAIL_HOST_KEY", "Datacube")
    calls = []

    async def fake_call_bridge_peer_tool(**kwargs):
        calls.append(kwargs)
        return {"id": "must-not-be-called"}

    monkeypatch.setattr(app_module, "_call_bridge_peer_tool", fake_call_bridge_peer_tool)
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/home/Antigravity"})
        await client.call_tool(
            "register_agent",
            {
                "project_key": "/home/Antigravity",
                "program": "workx",
                "model": "local",
                "name": "PurpleCanyon",
            },
        )
        await client.call_tool(
            "upsert_bridge_peer",
            {
                "host_key": "Primus",
                "auth_token": "test-primus-peer-token",
                "api_url": "http://127.0.0.1:18766/api/",
                "project_key": "/home/antigravity",
                "bridge_protocol_version": 1,
            },
        )
        sent = await client.call_tool(
            "send_message",
            {
                "project_key": "/home/Antigravity",
                "sender_name": "PurpleCanyon",
                "to": ["RoseStone@Primus"],
                "subject": "Blank queued project",
                "body_md": "must be quarantined before transport",
            },
        )
        bridge_id = sent.data["bridge_deliveries"][0]["bridge_id"]
        async with get_session() as session:
            await session.execute(
                text(
                    "UPDATE agent_mail_bridge_deliveries "
                    "SET dest_project_key = '' WHERE bridge_id = :bridge_id"
                ),
                {"bridge_id": bridge_id},
            )
            await session.commit()

        blocked = await client.call_tool("flush_bridge_deliveries", {"limit": 1})
        assert blocked.data["blocked"] == 1
        assert blocked.data["delivered"] == 0
        assert calls == []
        assert blocked.data["results"][0]["status"] == "blocked"
        assert "BLANK_BRIDGE_PROJECT" in blocked.data["results"][0]["last_error"]

        second = await client.call_tool("flush_bridge_deliveries", {"limit": 1})
        assert second.data["processed"] == 0
        assert calls == []


@pytest.mark.asyncio
async def test_flush_bridge_deliveries_pushes_to_configured_peer(isolated_env, monkeypatch):
    monkeypatch.setenv("AGENT_MAIL_HOST_KEY", "Datacube")
    calls = []

    async def fake_call_bridge_peer_tool(**kwargs):
        calls.append(kwargs)
        return {"id": 4242}

    monkeypatch.setattr(app_module, "_call_bridge_peer_tool", fake_call_bridge_peer_tool)
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/home/Antigravity"})
        await client.call_tool(
            "register_agent",
            {
                "project_key": "/home/Antigravity",
                "program": "codex",
                "model": "gpt-5",
                "name": "PurpleCanyon",
            },
        )
        await client.call_tool(
            "upsert_bridge_peer",
            {
                "host_key": "Primus",
                "auth_token": "test-primus-peer-token",
                "api_url": "http://127.0.0.1:18766/api/",
                "project_key": "/home/antigravity",
                "bridge_protocol_version": 1,
            },
        )
        sent = await client.call_tool(
            "send_message",
            {
                "project_key": "/home/Antigravity",
                "sender_name": "PurpleCanyon",
                "to": ["RoseStone@Primus"],
                "subject": "Bridge flush",
                "body_md": "deliver body",
                "thread_id": "bridge-flush",
                "topic": "cross-system",
                "ack_required": True,
            },
        )
        bridge_id = sent.data["bridge_deliveries"][0]["bridge_id"]
        local_message_id = sent.data["bridge_deliveries"][0]["local_message_id"]
        snapshot = sent.data["bridge_deliveries"][0]["attestation"]

        flushed = await client.call_tool("flush_bridge_deliveries", {"limit": 5})

        assert flushed.data["processed"] == 1
        assert flushed.data["delivered"] == 1
        assert calls[0]["api_url"] == "http://127.0.0.1:18766/api/"
        assert calls[0]["tool_name"] == "bridge_receive_message"
        args = calls[0]["arguments"]
        assert args["project_key"] == "/home/antigravity"
        assert args["to"] == ["RoseStone"]
        assert args["source_host"] == "Datacube"
        assert args["source_agent"] == "PurpleCanyon"
        assert args["bridge_id"] == bridge_id
        assert args["bridge_dedupe_id"] == f"Datacube:{local_message_id}:Primus:RoseStone"
        assert args["body_sha256"] == snapshot["body_sha256"]
        assert args["attestation_version"] == 1
        assert args["message_revision"] == 1
        assert args["supersedes_message_id"] is None

        deliveries = await client.call_tool("list_bridge_deliveries", {"status": "delivered"})
        delivered = _tool_list(deliveries)[0]
        assert delivered["remote_message_id"] == "4242"
        assert delivered["completed_ts"] is not None
