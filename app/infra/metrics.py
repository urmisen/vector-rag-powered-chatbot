import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

from app.infra.logger import logger


def _metrics_log_path() -> Path:
    """Return the path where bootstrap latency metrics are stored."""
    log_path = os.getenv("BOOTSTRAP_LATENCY_LOG", "logs/bootstrap_latency.jsonl")
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def record_latency_metric(
    label: str,
    stages: Iterable[Tuple[str, float]],
    total_seconds: float,
    extra: Dict[str, object] | None = None,
) -> None:
    """Persist a latency metric as JSON Lines for offline analysis."""
    try:
        record = {
            "timestamp": time.time(),
            "label": label,
            "total_ms": round(total_seconds * 1000, 3),
            "stages_ms": {name: round(duration * 1000, 3) for name, duration in stages},
        }
        if extra:
            record.update(extra)

        log_path = _metrics_log_path()
        with log_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record))
            fp.write("\n")
    except Exception as exc:
        logger.getChild("Metrics").warning("Failed to record latency metric: %s", exc)

