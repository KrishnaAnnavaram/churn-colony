"""Settings from environment variables (and an optional local .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_PREFIX = "CHURN_COLONY_"


def load_env_file(path: str | os.PathLike = ".env") -> int:
    """Read KEY=VALUE lines into os.environ. Existing variables win. Returns the count of new keys."""
    p = Path(path)
    if not p.is_file():
        return 0
    added = 0
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ and value != "":
            os.environ[key] = value
            added += 1
    return added


def _get(name: str, default: str) -> str:
    value = os.environ.get(ENV_PREFIX + name, "")
    return value if value.strip() else default


@dataclass(frozen=True)
class Settings:
    data_path: str | None
    seed: int
    output_dir: str
    contact_cost: float
    offer_success_rate: float
    value_months: float
    threads: int

    @classmethod
    def from_env(cls) -> "Settings":
        data = _get("DATA", "")
        settings = cls(
            data_path=data or None,
            seed=int(_get("SEED", "42")),
            output_dir=_get("OUTPUT", "runs"),
            contact_cost=float(_get("CONTACT_COST", "25")),
            offer_success_rate=float(_get("OFFER_SUCCESS", "0.25")),
            value_months=float(_get("VALUE_MONTHS", "6")),
            threads=int(_get("THREADS", "4")),
        )
        settings.check()
        return settings

    def check(self) -> None:
        if self.contact_cost < 0:
            raise ValueError("CHURN_COLONY_CONTACT_COST must be 0 or more")
        if not 0 < self.offer_success_rate <= 1:
            raise ValueError("CHURN_COLONY_OFFER_SUCCESS must be in (0, 1]")
        if self.value_months <= 0:
            raise ValueError("CHURN_COLONY_VALUE_MONTHS must be more than 0")
        if self.threads < 1:
            raise ValueError("CHURN_COLONY_THREADS must be 1 or more")
