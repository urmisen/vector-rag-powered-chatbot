###Pydantic is Python's most popular data validation library. It leverages Python's type hints to ensure that any data you accept (especially in APIs) matches the structure and type you expect. If data is missing or the wrong type, Pydantic will immediately raise a descriptive error and stop the operation, meaning your API will never silently accept bad data—this dramatically reduces bugs and security issues.
from pydantic import BaseModel          
from typing import Optional, List, Dict

class RegulationItem(BaseModel):
    title: str
    description: str
    category: str
    effective_date: Optional[str] = None
    reference: Optional[str] = None

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    query: str
    conversation_id: Optional[str] = None
    context: Optional[Dict] = None

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str
    sources: Optional[List[str]] = None
    conversation_id: Optional[str] = None

class VectorStoreMetadata(BaseModel):
    """Model for vector store document metadata"""
    file_identifier: str
    file_name: str
    chunk_index: int
    chunk_text: str

class ToolCallRequest(BaseModel):
    """Model for MCP tool calls"""
    tool_name: str
    parameters: Dict[str, str]

class ToolResponse(BaseModel):
    """Model for MCP tool responses"""
    content: str
    is_error: bool = False

class ConversationLog(BaseModel):
    """Model for storing conversation history"""
    messages: List[Dict[str, str]]
    timestamp: str