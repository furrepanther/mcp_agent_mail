from __future__ import annotations

import pytest

from mcp_agent_mail.bridge_identity import (
    BridgeIdentityError,
    configured_contract,
    validate_local_identity,
    validate_peer_identity,
    validate_remote_recipient,
)


@pytest.mark.parametrize(
    ("host_key", "local_agent", "peer_host", "peer_agent"),
    [
        ("Datacube", "PurpleCanyon", "Primus", "RoseStone"),
        ("Primus", "RoseStone", "Datacube", "PurpleCanyon"),
    ],
)
def test_canonical_host_contract(host_key, local_agent, peer_host, peer_agent, monkeypatch):
    monkeypatch.setenv("AGENT_MAIL_HOST_KEY", host_key)
    contract = configured_contract()
    assert contract.local_agent == local_agent
    assert contract.peer_host == peer_host
    assert contract.peer_agent == peer_agent
    assert validate_local_identity(local_agent) == contract
    assert validate_peer_identity(peer_host, peer_agent) == contract
    assert validate_remote_recipient(peer_host, peer_agent) == contract


def test_host_guessing_is_rejected(monkeypatch):
    monkeypatch.delenv("AGENT_MAIL_HOST_KEY", raising=False)
    with pytest.raises(BridgeIdentityError, match="AGENT_MAIL_HOST_KEY is required"):
        configured_contract()


@pytest.mark.parametrize(
    "bad_agent",
    ["SwiftCat", "TealCove", "AmberGrove", "GreenStone", "mister-fixit"],
)
def test_retired_identity_is_rejected_on_both_directions(bad_agent, monkeypatch):
    monkeypatch.setenv("AGENT_MAIL_HOST_KEY", "Datacube")
    with pytest.raises(BridgeIdentityError):
        validate_local_identity(bad_agent)
    with pytest.raises(BridgeIdentityError):
        validate_remote_recipient("Primus", bad_agent)


def test_wrong_peer_identity_and_host_are_rejected(monkeypatch):
    monkeypatch.setenv("AGENT_MAIL_HOST_KEY", "Datacube")
    with pytest.raises(BridgeIdentityError, match="retired"):
        validate_peer_identity("Primus", "TealCove")
    with pytest.raises(BridgeIdentityError, match="not the configured peer"):
        validate_peer_identity("Windows", "RoseStone")

