"""
Web (Streamlit) interface for the Pay Regulations Chatbot.

This module exposes the high-level frontend helpers in a single place.

Example:
    from app.interfaces.web import run_frontend
"""

from app.interfaces.web.frontend import run_frontend, cleanup

__all__ = ["run_frontend", "cleanup"]


