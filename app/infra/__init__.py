"""
Lightweight infra package initializer.

To avoid circular imports, import directly from submodules, for example:

    from app.infra.logger import logger
    from app.infra.bigquery import BigQueryManager
    from app.infra.metrics import record_latency_metric
    from app.infra.background_client import get_background_client_manager
"""


