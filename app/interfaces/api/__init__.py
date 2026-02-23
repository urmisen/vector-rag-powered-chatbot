"""
API interfaces for the Pay Regulations Chatbot.

This package exposes the FastAPI application in a discoverable place.

Example:
    from app.interfaces.api import app
"""

from app.interfaces.api.api import app  # FastAPI application instance

__all__ = ["app"]


