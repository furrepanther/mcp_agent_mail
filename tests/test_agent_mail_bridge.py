from __future__ import annotations

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

import mcp_agent_mail.app as app_module
from mcp_agent_mail.app import build_mcp_server


def _tool_list(result):
    if hasattr(result, "structured_content") and isinstance(result.structured_content, dict):
        structured = result.structured_content.get("result")
        if isinstance(structured, list):
            return structured
    return list(getattr(result, "data", result))


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
                "name": "RoseStone",
            },
        )

        sent = await client.call_tool(
            "send_message",
            {
                "project_key": "/home/Antigravity",
                "sender_name": "RoseStone",
                "to": ["SwiftCat@Primus"],
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
        assert bridge["dest_agent"] == "SwiftCat"
        assert bridge["status"] == "queued"

        identities = await client.call_tool(
            "list_identities",
            {"project_key": "/home/Antigravity", "include_remote": True},
        )
        assert any(
            item["host_key"] == "Primus" and item["agent_name"] == "SwiftCat"
            for item in identities.data["remote"]
        )


@pytest.mark.asyncio
async def test_bridge_receive_preserves_remote_provenance(isolated_env, monkeypatch):
    monkeypatch.setenv("AGENT_MAIL_HOST_KEY", "Primus")
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/home/antigravity"})
        registration = await client.call_tool(
            "register_agent",
            {
                "project_key": "/home/antigravity",
                "program": "codex",
                "model": "gpt-5",
                "name": "SwiftCat",
            },
        )
        token = registration.data["registration_token"]

        received = await client.call_tool(
            "bridge_receive_message",
            {
                "project_key": "/home/antigravity",
                "to": ["SwiftCat"],
                "subject": "Inbound",
                "body_md": "from datacube",
                "source_host": "Datacube",
                "source_agent": "RoseStone",
                "source_project_key": "/home/Antigravity",
                "external_message_id": "11192",
                "external_thread_id": "thread-1",
                "bridge_dedupe_id": "datacube:11192:swiftcat",
                "topic": "cross-system",
            },
        )

        assert received.data["from"] == "RoseStone@Datacube"
        assert received.data["external_sender"]["host"] == "Datacube"
        assert received.data["external_sender"]["name"] == "RoseStone"
        assert received.data["external_sender"]["project_key"] == "/home/Antigravity"

        inbox = await client.call_tool(
            "fetch_inbox",
            {
                "project_key": "/home/antigravity",
                "agent_name": "SwiftCat",
                "registration_token": token,
                "include_bodies": True,
                "topic": "cross-system",
            },
        )
        inbox_items = _tool_list(inbox)
        assert inbox_items[0]["from"] == "AmberGrove"
        assert inbox_items[0]["external_sender"]["name"] == "RoseStone"
        assert inbox_items[0]["external_sender"]["host"] == "Datacube"

        duplicate = await client.call_tool(
            "bridge_receive_message",
            {
                "project_key": "/home/antigravity",
                "to": ["SwiftCat"],
                "subject": "Inbound",
                "body_md": "from datacube",
                "source_host": "Datacube",
                "source_agent": "RoseStone",
                "external_message_id": "11192",
                "bridge_dedupe_id": "datacube:11192:swiftcat",
            },
        )
        assert duplicate.data["duplicate"] is True
        assert duplicate.data["id"] == received.data["id"]


@pytest.mark.asyncio
async def test_bridge_receive_peer_token_enforced_when_configured(isolated_env):
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/home/antigravity"})
        await client.call_tool(
            "register_agent",
            {
                "project_key": "/home/antigravity",
                "program": "codex",
                "model": "gpt-5",
                "name": "SwiftCat",
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
                    "to": ["SwiftCat"],
                    "subject": "Inbound",
                    "body_md": "from datacube",
                    "source_host": "Datacube",
                    "source_agent": "RoseStone",
                    "peer_token": "wrong",
                },
            )

        ok = await client.call_tool(
            "bridge_receive_message",
            {
                "project_key": "/home/antigravity",
                "to": ["SwiftCat"],
                "subject": "Inbound",
                "body_md": "from datacube",
                "source_host": "Datacube",
                "source_agent": "RoseStone",
                "peer_token": "secret",
            },
        )
        assert ok.data["from"] == "RoseStone@Datacube"


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
                "name": "RoseStone",
            },
        )
        await client.call_tool(
            "upsert_bridge_peer",
            {
                "host_key": "Primus",
                "api_url": "http://127.0.0.1:18766/api/",
                "project_key": "/home/antigravity",
            },
        )
        sent = await client.call_tool(
            "send_message",
            {
                "project_key": "/home/Antigravity",
                "sender_name": "RoseStone",
                "to": ["SwiftCat@Primus"],
                "subject": "Bridge flush",
                "body_md": "deliver body",
                "thread_id": "bridge-flush",
                "topic": "cross-system",
                "ack_required": True,
            },
        )
        bridge_id = sent.data["bridge_deliveries"][0]["bridge_id"]
        local_message_id = sent.data["bridge_deliveries"][0]["local_message_id"]

        flushed = await client.call_tool("flush_bridge_deliveries", {"limit": 5})

        assert flushed.data["processed"] == 1
        assert flushed.data["delivered"] == 1
        assert calls[0]["api_url"] == "http://127.0.0.1:18766/api/"
        assert calls[0]["tool_name"] == "bridge_receive_message"
        args = calls[0]["arguments"]
        assert args["project_key"] == "/home/antigravity"
        assert args["to"] == ["SwiftCat"]
        assert args["source_host"] == "Datacube"
        assert args["source_agent"] == "RoseStone"
        assert args["bridge_id"] == bridge_id
        assert args["bridge_dedupe_id"] == f"Datacube:{local_message_id}:Primus:SwiftCat"

        deliveries = await client.call_tool("list_bridge_deliveries", {"status": "delivered"})
        delivered = _tool_list(deliveries)[0]
        assert delivered["remote_message_id"] == "4242"
        assert delivered["completed_ts"] is not None
