"""Canonical cross-host Agent Mail bridge identity contract.

The bridge has one mailbox identity per host.  The transport process is not an
Agent Mail identity and must never be allowed to mint a replacement identity
when a peer is unavailable.  Keeping this contract in a small, dependency-free
module makes the admission and outbound paths use the same rules.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

LEGACY_IDENTITIES = frozenset(
    {
        "AmberGrove",
        "GreenStone",
        "SwiftCat",
        "TealCove",
        "mister-fixit",
    }
)


@dataclass(frozen=True, slots=True)
class BridgePeerContract:
    local_host: str
    local_agent: str
    peer_host: str
    peer_agent: str


_CONTRACTS = {
    "datacube": BridgePeerContract(
        local_host="Datacube",
        local_agent="PurpleCanyon",
        peer_host="Primus",
        peer_agent="RoseStone",
    ),
    "datacubewsl": BridgePeerContract(
        local_host="Datacube",
        local_agent="PurpleCanyon",
        peer_host="Primus",
        peer_agent="RoseStone",
    ),
    "primus": BridgePeerContract(
        local_host="Primus",
        local_agent="RoseStone",
        peer_host="Datacube",
        peer_agent="PurpleCanyon",
    ),
    "primuswsl": BridgePeerContract(
        local_host="Primus",
        local_agent="RoseStone",
        peer_host="Datacube",
        peer_agent="PurpleCanyon",
    ),
}


class BridgeIdentityError(ValueError):
    """A fail-closed bridge identity violation."""

    def __init__(self, code: str, message: str, *, data: dict[str, str] | None = None):
        super().__init__(message)
        self.code = code
        self.data = data or {}


def normalize_host(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").strip().lower())


def is_legacy_identity(value: str) -> bool:
    return (value or "").strip().casefold() in {item.casefold() for item in LEGACY_IDENTITIES}


def configured_contract() -> BridgePeerContract:
    raw = os.getenv("AGENT_MAIL_HOST_KEY", "").strip()
    if not raw:
        raise BridgeIdentityError(
            "BRIDGE_HOST_UNCONFIGURED",
            "AGENT_MAIL_HOST_KEY is required for cross-host Agent Mail; refusing hostname guessing.",
        )
    contract = _CONTRACTS.get(normalize_host(raw))
    if contract is None:
        raise BridgeIdentityError(
            "BRIDGE_HOST_UNSUPPORTED",
            f"AGENT_MAIL_HOST_KEY={raw!r} is not a canonical Datacube or Primus host.",
            data={"configured_host": raw},
        )
    return contract


def validate_peer_identity(source_host: str, source_agent: str) -> BridgePeerContract:
    contract = configured_contract()
    normalized_host = normalize_host(source_host)
    normalized_agent = (source_agent or "").strip().casefold()
    if not normalized_host or not normalized_agent:
        raise BridgeIdentityError(
            "BRIDGE_SOURCE_IDENTITY_REQUIRED",
            "source_host and source_agent are required for cross-host delivery.",
        )
    if is_legacy_identity(source_agent):
        raise BridgeIdentityError(
            "BRIDGE_RETIRED_IDENTITY",
            f"Bridge source identity {source_agent!r} is retired; refusing stale-mail admission.",
            data={"source_host": source_host, "source_agent": source_agent},
        )
    if normalized_host != normalize_host(contract.peer_host):
        raise BridgeIdentityError(
            "BRIDGE_PEER_HOST_MISMATCH",
            f"Bridge source host {source_host!r} is not the configured peer {contract.peer_host!r}.",
            data={"expected_host": contract.peer_host, "source_host": source_host},
        )
    if normalized_agent != contract.peer_agent.casefold():
        raise BridgeIdentityError(
            "BRIDGE_PEER_IDENTITY_MISMATCH",
            f"Bridge source identity {source_agent!r} is not canonical for {contract.peer_host}: "
            f"expected {contract.peer_agent!r}.",
            data={"expected_agent": contract.peer_agent, "source_agent": source_agent},
        )
    return contract


def validate_local_identity(agent_name: str) -> BridgePeerContract:
    contract = configured_contract()
    normalized = (agent_name or "").strip().casefold()
    if is_legacy_identity(agent_name):
        raise BridgeIdentityError(
            "BRIDGE_RETIRED_IDENTITY",
            f"Bridge local identity {agent_name!r} is retired; refusing stale-mail transmission.",
            data={"agent_name": agent_name},
        )
    if normalized != contract.local_agent.casefold():
        raise BridgeIdentityError(
            "BRIDGE_LOCAL_IDENTITY_MISMATCH",
            f"Bridge local identity {agent_name!r} is not canonical for {contract.local_host}: "
            f"expected {contract.local_agent!r}.",
            data={"expected_agent": contract.local_agent, "agent_name": agent_name},
        )
    return contract


def validate_remote_recipient(host_key: str, agent_name: str) -> BridgePeerContract:
    contract = configured_contract()
    normalized_host = normalize_host(host_key)
    normalized_agent = (agent_name or "").strip().casefold()
    if is_legacy_identity(agent_name):
        raise BridgeIdentityError(
            "BRIDGE_RETIRED_IDENTITY",
            f"Bridge destination identity {agent_name!r} is retired; refusing stale-mail transmission.",
            data={"dest_host": host_key, "dest_agent": agent_name},
        )
    if normalized_host != normalize_host(contract.peer_host) or normalized_agent != contract.peer_agent.casefold():
        raise BridgeIdentityError(
            "BRIDGE_DESTINATION_IDENTITY_MISMATCH",
            f"Bridge destination must be {contract.peer_agent}@{contract.peer_host}; "
            f"received {agent_name}@{host_key}.",
            data={
                "expected_host": contract.peer_host,
                "expected_agent": contract.peer_agent,
                "dest_host": host_key,
                "dest_agent": agent_name,
            },
        )
    return contract
