from __future__ import annotations

import pytest

from mcp_agent_mail.app import ToolExecutionError, _instrument_tool


@pytest.mark.asyncio
async def test_instrumented_tool_returns_result_after_finally_cleanup(isolated_env) -> None:
    @_instrument_tool("test_instrumented_return_contract", cluster="infrastructure")
    async def sample_tool() -> dict[str, bool]:
        return {"ok": True}

    assert await sample_tool() == {"ok": True}


@pytest.mark.asyncio
async def test_permission_error_is_not_misclassified_as_recoverable_database_error(isolated_env) -> None:
    @_instrument_tool("test_instrumented_permission_contract", cluster="infrastructure")
    async def sample_tool() -> None:
        raise PermissionError("database permission denied")

    with pytest.raises(ToolExecutionError) as exc_info:
        await sample_tool()

    assert exc_info.value.error_type == "OS_ERROR"
    assert exc_info.value.recoverable is False
