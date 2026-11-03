"""Vertical pack loader. One pack ships today: field_sales_jewellery."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PACK_DIR = Path(__file__).resolve().parent
DEFAULT_PACK = "field_sales_jewellery"

# Every string the code asks for. Loading fails fast if the YAML drops one.
REQUIRED_STRINGS: frozenset[str] = frozenset(
    {
        "btn_yes", "btn_no", "btn_accept", "btn_decline",
        "consent_notice", "consent_accepted", "consent_declined", "consent_required",
        "consent_withdrawn", "consent_toast",
        "not_registered", "account_inactive", "invite_invalid", "welcome_back",
        "welcome_bound", "rate_limited", "error_generic", "unsupported", "echo",
        "help_worker", "help_boss", "unknown_command", "boss_only",
        "addworker_usage", "addworker_bad_name", "addworker_confirm", "addworker_done",
        "addworker_cancelled", "action_expired", "action_toast_done", "action_toast_cancelled",
        "team_header", "team_line", "team_status_invited", "team_status_pending_consent",
        "team_status_active", "team_status_withdrawn", "team_empty",
        "policy_rate_below_floor", "policy_discount_over_cap", "policy_credit_over_limit",
        "policy_order_over_max", "policy_benchmark_missing", "policy_benchmark_stale",
        "policy_benchmark_future", "policy_unit_mismatch", "policy_fact_missing",
    }
)  # fmt: skip

REQUIRED_STRINGS_EN: frozenset[str] = frozenset(
    k for k in REQUIRED_STRINGS if k.startswith("policy_")
) | {"policy_config_invalid"}
# Strings added later are protected by tests/test_pack_and_migrations.py, which fails if any
# key used in code (pack.t / ctx.t / pack.te) is missing from the YAML.


@dataclass(frozen=True)
class Pack:
    name: str
    version: int
    strings: dict[str, str]
    strings_en: dict[str, str] = field(default_factory=dict)
    bounds: dict[str, int] = field(default_factory=dict)
    vocabulary: tuple[str, ...] = ()
    months: tuple[str, ...] = ()

    def t(self, key: str, **values: Any) -> str:
        """Hindi string. Values are substituted as data; they are never re-parsed."""
        return self.strings[key].format(**values)

    def te(self, key: str, **values: Any) -> str:
        """English string (policy reasons, for the dashboard and logs)."""
        return self.strings_en[key].format(**values)


def load_pack(name: str = DEFAULT_PACK) -> Pack:
    raw = yaml.safe_load((PACK_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
    strings = {k: str(v).rstrip("\n") for k, v in raw["strings"].items()}
    strings_en = {k: str(v).rstrip("\n") for k, v in raw.get("strings_en", {}).items()}
    missing = REQUIRED_STRINGS - strings.keys()
    missing_en = REQUIRED_STRINGS_EN - strings_en.keys()
    if missing or missing_en:
        raise ValueError(
            f"pack {name!r} is missing strings: {sorted(missing)} / English: {sorted(missing_en)}"
        )
    bounds = {str(k): int(v) for k, v in raw.get("bounds", {}).items()}
    return Pack(
        name=raw["name"],
        version=int(raw["version"]),
        strings=strings,
        strings_en=strings_en,
        bounds=bounds,
        vocabulary=tuple(str(v) for v in raw.get("vocabulary", [])),
        months=tuple(str(m) for m in raw.get("months_en", [])),
    )
