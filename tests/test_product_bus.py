import json

import pytest
from fastmcp import Client

from mcp_agent_mail.app import build_mcp_server
from mcp_agent_mail.config import clear_settings_cache
from mcp_agent_mail.db import ensure_schema, reset_database_state


@pytest.mark.asyncio
async def test_ensure_product_and_link_project(tmp_path, monkeypatch) -> None:
    # Enable gated features for product bus
    monkeypatch.setenv("WORKTREES_ENABLED", "1")
    clear_settings_cache()
    reset_database_state()
    await ensure_schema()
    # Ensure product (unique ids to avoid cross-run collisions)
    unique = "_prod_" + hex(hash(str(tmp_path)) & 0xFFFFF)[2:]
    server = build_mcp_server()
    async with Client(server) as client:
        product_result = await client.call_tool(
            "ensure_product",
            {"product_key": f"my-product{unique}", "name": f"My Product{unique}"},
        )
        product = product_result.data
        assert product["product_uid"]
        project_result = await client.call_tool("ensure_project", {"human_key": str(tmp_path)})
        project = project_result.data
        slug = project.get("slug") or project["project"]["slug"]
        link_result = await client.call_tool(
            "products_link",
            {"product_key": product["product_uid"], "project_key": slug},
        )
        assert link_result.data["linked"] is True
        resources = await client.read_resource(f"resource://product/{product['product_uid']}")
        assert resources and resources[0].text
        payload = json.loads(resources[0].text)
        assert any(item["slug"] == slug for item in payload.get("projects", []))

