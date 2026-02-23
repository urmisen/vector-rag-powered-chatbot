"""
Core domain and RAG logic for the Pay Regulations Chatbot.

This package provides a single, beginner-friendly place to import the
main RAG manager and shared models without changing the existing API.

Example:
    from app.core import RAGManager, ChatRequest
"""

from app.core.rag_manager import RAGManager
from app.core.models import (
    RegulationItem,
    ChatRequest,
    ChatResponse,
    VectorStoreMetadata,
    ToolCallRequest,
    ToolResponse,
    ConversationLog,
)

__all__ = [
    "RAGManager",
    "RegulationItem",
    "ChatRequest",
    "ChatResponse",
    "VectorStoreMetadata",
    "ToolCallRequest",
    "ToolResponse",
    "ConversationLog",
]


