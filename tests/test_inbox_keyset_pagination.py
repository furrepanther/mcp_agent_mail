from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from mcp_agent_mail.app import build_mcp_server

sys.path.insert(0, str(Path(__file__).parent))
from test_message_delivery_regression import get_inbox_items, setup_project_with_agents


@pytest.mark.asyncio
async def test_fetch_inbox_keyset_pages_are_disjoint_with_same_timestamp(isolated_env):
    server = build_mcp_server()
    async with Client(server) as client:
        project = "/test/keyset-inbox"
        sender, receiver = await setup_project_with_agents(client, project, count=2)
        for index in range(25):
            await client.call_tool(
                "send_message",
                {
                    "project_key": project,
                    "sender_name": sender,
                    "to": [receiver],
                    "subject": f"Message {index}",
                    "body_md": "body",
                },
            )

        first = get_inbox_items(
            await client.call_tool(
                "fetch_inbox",
                {"project_key": project, "agent_name": receiver, "limit": 10},
            )
        )
        assert len(first) == 10
        boundary = first[-1]
        second = get_inbox_items(
            await client.call_tool(
                "fetch_inbox",
                {
                    "project_key": project,
                    "agent_name": receiver,
                    "limit": 10,
                    "before_ts": boundary["created_ts"],
                    "before_id": boundary["id"],
                },
            )
        )
        assert len(second) == 10
        assert {item["id"] for item in first}.isdisjoint({item["id"] for item in second})
        assert all(
            (item["created_ts"], item["id"]) < (boundary["created_ts"], boundary["id"])
            for item in second
        )


@pytest.mark.asyncio
async def test_fetch_inbox_rejects_partial_keyset_boundary(isolated_env):
    server = build_mcp_server()
    async with Client(server) as client:
        project = "/test/keyset-invalid"
        _, receiver = await setup_project_with_agents(client, project, count=2)
        with pytest.raises(ToolError, match="before_id requires before_ts"):
            await client.call_tool(
                "fetch_inbox",
                {
                    "project_key": project,
                    "agent_name": receiver,
                    "before_id": 4,
                },
            )
        with pytest.raises(ToolError, match="before_ts requires before_id"):
            await client.call_tool(
                "fetch_inbox",
                {
                    "project_key": project,
                    "agent_name": receiver,
                    "before_ts": "2026-11-20T17:00:00Z",
                },
            )
