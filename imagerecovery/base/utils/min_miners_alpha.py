from datetime import datetime, timezone

from imagerecovery.const import (
    MINERS_MINIMUM_ALPHA_BASE,
    MINERS_MINIMUM_ALPHA_DAILY_INCREASE,
    MINERS_MINIMUM_ALPHA_ENABLE_DATE,
)


def calculate_minimum_miner_alpha() -> int:
    considering_days = (datetime.now(tz=timezone.utc) - datetime.fromisoformat(MINERS_MINIMUM_ALPHA_ENABLE_DATE)).days
    return MINERS_MINIMUM_ALPHA_BASE + MINERS_MINIMUM_ALPHA_DAILY_INCREASE * considering_days
