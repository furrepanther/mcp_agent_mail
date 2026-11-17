from __future__ import annotations

from mcp_agent_mail.app import _is_machine_coordination_notice


def test_primus_outage_share_is_not_mirrored() -> None:
    payload = {"subject": "Primus core service outage: Memory Fabric tunnel"}
    body = "share_id: agshare:primus:abc123\nsource_system: primus"

    assert _is_machine_coordination_notice(payload, body)


def test_primus_work_order_share_is_still_mirrored() -> None:
    payload = {"subject": "Please repair the Memory Fabric tunnel"}
    body = "share_id: agshare:primus:abc123\nag-human-visible-request: true"

    assert not _is_machine_coordination_notice(payload, body)


def test_non_share_informational_mail_is_still_mirrored() -> None:
    payload = {"subject": "Status update"}
    body = "Informational cross-system coordination; no action is requested."

    assert not _is_machine_coordination_notice(payload, body)
