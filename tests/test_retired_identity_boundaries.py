from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from sqlalchemy import select

from mcp_agent_mail.app import build_mcp_server
from mcp_agent_mail.config import clear_settings_cache
from mcp_agent_mail.db import get_session
from mcp_agent_mail.models import AgentLink


@pytest.mark.asyncio
async def test_retired_identity_cannot_reuse_bound_session_or_registration_token(isolated_env):
    """Retirement must stop both session reuse and fresh token authentication."""
    server = build_mcp_server()
    project_key = "/identity/retired-boundary"

    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": project_key})
        agent = await client.call_tool(
            "register_agent",
            {
                "project_key": project_key,
                "program": "test",
                "model": "test",
                "name": "TealCove",
            },
        )
        token = agent.data["registration_token"]

        await client.call_tool(
            "retire_agent",
            {
                "project_key": project_key,
                "agent_name": "TealCove",
                "registration_token": token,
            },
        )

        with pytest.raises(ToolError, match="retired and cannot"):
            await client.call_tool(
                "send_message",
                {
                    "project_key": project_key,
                    "sender_name": "TealCove",
                    "to": ["TealCove"],
                    "subject": "must not send",
                    "body_md": "retired session reuse must fail",
                },
            )

    async with Client(server) as fresh_client:
        with pytest.raises(ToolError, match="retired and cannot"):
            await fresh_client.call_tool(
                "send_message",
                {
                    "project_key": project_key,
                    "sender_name": "TealCove",
                    "sender_token": token,
                    "to": ["TealCove"],
                    "subject": "must not send",
                    "body_md": "retired token reuse must fail",
                },
            )

        with pytest.raises(ToolError, match="retired and cannot"):
            await fresh_client.call_tool(
                "register_agent",
                {
                    "project_key": project_key,
                    "program": "test",
                    "model": "test",
                    "name": "TealCove",
                    "registration_token": token,
                },
            )

        restored = await fresh_client.call_tool(
            "unretire_agent",
            {
                "project_key": project_key,
                "agent_name": "TealCove",
                "registration_token": token,
            },
        )
        assert restored.data["status"] == "active"

        sent = await fresh_client.call_tool(
            "send_message",
            {
                "project_key": project_key,
                "sender_name": "TealCove",
                "sender_token": token,
                "to": ["TealCove"],
                "subject": "restored identity may send",
                "body_md": "explicit restoration is the only recovery path",
            },
        )
        assert sent.data["count"] == 1


@pytest.mark.asyncio
async def test_expired_approved_contact_link_cannot_authorize_delivery(isolated_env, monkeypatch):
    """A contact approval expires; stale rows must not retain delivery authority."""
    monkeypatch.setenv("CONTACT_ENFORCEMENT_ENABLED", "true")
    clear_settings_cache()
    server = build_mcp_server()
    project_key = "/identity/expired-contact"

    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": project_key})
        sender = await client.call_tool(
            "register_agent",
            {
                "project_key": project_key,
                "program": "test",
                "model": "test",
                "name": "GreenCastle",
            },
        )
        recipient = await client.call_tool(
            "register_agent",
            {
                "project_key": project_key,
                "program": "test",
                "model": "test",
                "name": "BlueLake",
            },
        )
        await client.call_tool(
            "set_contact_policy",
            {
                "project_key": project_key,
                "agent_name": "BlueLake",
                "policy": "contacts_only",
            },
        )
        await client.call_tool(
            "macro_contact_handshake",
            {
                "project_key": project_key,
                "requester": "GreenCastle",
                "target": "BlueLake",
                "auto_accept": True,
                "requester_registration_token": sender.data["registration_token"],
                "target_registration_token": recipient.data["registration_token"],
            },
        )

        async with get_session() as session:
            result = await session.execute(
                select(AgentLink).where(
                    AgentLink.a_agent_id == sender.data["id"],
                    AgentLink.b_agent_id == recipient.data["id"],
                )
            )
            link = result.scalars().one()
            link.expires_ts = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=1)
            session.add(link)
            await session.commit()

    async with Client(server) as fresh_client:
        with pytest.raises(ToolError, match="Contact approval required"):
            await fresh_client.call_tool(
                "send_message",
                {
                    "project_key": project_key,
                    "sender_name": "GreenCastle",
                    "sender_token": sender.data["registration_token"],
                    "to": ["BlueLake"],
                    "subject": "expired link must not authorize",
                    "body_md": "stale contact links are not authority",
                },
            )
