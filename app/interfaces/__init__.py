"""
User-facing interfaces for the Pay Regulations Chatbot.

This package groups:
- API interfaces (FastAPI + Mattermost) under app.interfaces.api
- Web UI (Streamlit frontend) under app.interfaces.web
"""

from . import api, web  # noqa: F401

__all__ = ["api", "web"]


