"""Client matching. "Rajendra Jewellers" and "Rajendra Jwellers" are the same shop;
two near-identical names are ambiguous and the rep is asked, never guessed for."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Literal

from app.db import TenantRepository
from app.models import Client

LIKELY_MIN = 0.88  # confident match, but the read-back still shows the stored name
LIKELY_MARGIN = 0.06  # best must beat the runner-up by this much
AMBIGUOUS_MIN = 0.72

_JUNK = re.compile(r"[^\w\sऀ-ॿ]", re.UNICODE)


def canonical_name(name: str) -> str:
    """Casefold, drop punctuation, and fold the many spellings of 'jewellers'."""
    text = unicodedata.normalize("NFKC", name).casefold().replace("&", " and ")
    tokens = []
    for tok in _JUNK.sub(" ", text).split():
        if tok.startswith(("jew", "jw")) or tok.startswith("ज्वै") or tok.startswith("ज्वेल"):
            tok = "jewel"
        tokens.append(tok)
    return " ".join(tokens)


@dataclass
class ClientMatch:
    status: Literal["exact", "likely", "ambiguous", "new"]
    client: Client | None = None
    candidates: list[Client] = field(default_factory=list)


def _score(canon: str, client: Client) -> float:
    names = [client.normalized_name, *(canonical_name(a) for a in client.aliases)]
    return max(SequenceMatcher(None, canon, n).ratio() for n in names)


async def match_client(repo: TenantRepository, name: str) -> ClientMatch:
    canon = canonical_name(name)
    clients = await repo.list(Client)
    scored = sorted(((_score(canon, c), c) for c in clients), key=lambda sc: -sc[0])
    exact = [c for s, c in scored if s == 1.0]
    if len(exact) == 1:
        return ClientMatch("exact", exact[0])
    if len(exact) > 1:  # duplicate clients already on file
        return ClientMatch("ambiguous", candidates=exact[:3])
    if scored and scored[0][0] >= LIKELY_MIN:
        runner_up = scored[1][0] if len(scored) > 1 else 0.0
        if scored[0][0] - runner_up >= LIKELY_MARGIN:
            return ClientMatch("likely", scored[0][1])
    near = [c for s, c in scored if s >= AMBIGUOUS_MIN][:3]
    if near:
        return ClientMatch("ambiguous", candidates=near)
    return ClientMatch("new")


def create_client(repo: TenantRepository, name: str) -> Client:
    name = " ".join(name.split())
    return repo.add(Client(name=name, normalized_name=canonical_name(name), aliases=[]))


def remember_alias(client: Client, spoken: str) -> bool:
    """Store a new spelling so the same name is an exact match next time."""
    spoken = " ".join(spoken.split())
    if canonical_name(spoken) == client.normalized_name or spoken in client.aliases:
        return False
    client.aliases = [*client.aliases, spoken]  # reassign: JSON columns are not mutation-tracked
    return True
