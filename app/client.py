import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from app.core import RAGManager
from app.infra.logger import logger
import os
import json
import asyncio
from datetime import datetime
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.auth import default
from typing import Optional, Dict, List
from dataclasses import dataclass
import uuid

# Add BigQuery import
from app.infra.bigquery import BigQueryManager, get_bigquery_manager

@dataclass
class ChatMessage:
    role: str  # "user" or "assistant"
    content: str
    timestamp: str  # Add timestamp field

class MCPClient:
    """MCP Client for handling user queries and AI interactions with context window support"""
    
    def __init__(
        self,
        authenticated_user_email: str = None,
        rag_manager: Optional[RAGManager] = None,
        bigquery_manager: Optional[BigQueryManager] = None,
    ):
        self.rag = rag_manager or RAGManager()
        self.logger = logger.getChild("MCPClient")
        self._chat_history: List[ChatMessage] = []  # Always start fresh
        self._context_window = int(os.getenv("CONTEXT_WINDOW_SIZE", "5"))  # Configurable context window size
        
        # Initialize BigQuery manager
        self.bq_manager = bigquery_manager or get_bigquery_manager()
        self.bq_max_concurrency = int(os.getenv("BQ_MAX_CONCURRENCY", "3"))
        self._bq_semaphore = None
        self._pending_bq_tasks = []
        
        # Check deployment environment
        deployment = os.getenv("DEPLOYMENT", "dev")
        
        # Use authenticated user email if provided, otherwise fall back to GCP service account
        if authenticated_user_email:
            self.gcp_username = self._format_user_email(authenticated_user_email)
            self.authenticated_user = authenticated_user_email
            # Create user_id from email for BigQuery
            self.user_id = authenticated_user_email.split("@")[0] if authenticated_user_email else "unknown"
            self.logger.info(f"New session for authenticated user: {authenticated_user_email}")
        elif deployment == "dev":
            # In dev mode, use a default dev user
            dev_user_email = os.getenv("DEV_USER_EMAIL", "dev@example.com")
            self.gcp_username = self._format_user_email(dev_user_email)
            self.authenticated_user = dev_user_email
            self.user_id = dev_user_email.split("@")[0] if dev_user_email else "dev_user"
            self.logger.info(f"New session for dev user: {dev_user_email}")
        else:
            self.gcp_username = self._get_gcp_username()
            self.authenticated_user = None
            self.user_id = self.gcp_username
            self.logger.info(f"New session for GCP user: {self.gcp_username}")
        
        # Create session-specific conversation ID for BigQuery
        self.conversation_id = str(uuid.uuid4())
        
        # Track if conversation session has been saved
        self._conversation_session_saved = False
        
        # Track the number of messages saved to conversation session
        self._conversation_session_message_count = 0
        
        self.credentials = self._initialize_gcp_auth()
        
        # Cache for conversation sessions
        self._conversation_session_cache = {}
        self._conversation_session_cache_timestamp = {}

    def reset_for_user(self, authenticated_user_email: Optional[str]) -> None:
        """Reinitialize lightweight session state for a new authenticated user."""
        if authenticated_user_email:
            self.authenticated_user = authenticated_user_email
            self.gcp_username = self._format_user_email(authenticated_user_email)
            self.user_id = authenticated_user_email.split("@")[0] if authenticated_user_email else "unknown"
            self.logger.info("Resetting session for authenticated user: %s", authenticated_user_email)
        else:
            deployment = os.getenv("DEPLOYMENT", "dev")
            if deployment == "dev":
                dev_user_email = os.getenv("DEV_USER_EMAIL", "dev@example.com")
                self.authenticated_user = dev_user_email
                self.gcp_username = self._format_user_email(dev_user_email)
                self.user_id = dev_user_email.split("@")[0] if dev_user_email else "dev_user"
                self.logger.info("Resetting session for dev user: %s", dev_user_email)
            else:
                self.authenticated_user = None
                self.gcp_username = self._get_gcp_username()
                self.user_id = self.gcp_username
                self.logger.info("Resetting session for anonymous GCP user: %s", self.gcp_username)

        # Reset in-memory conversation context
        self._chat_history = []
        self.conversation_id = str(uuid.uuid4())
        self._conversation_session_saved = False
        self._conversation_session_message_count = 0

    def _format_user_email(self, email: str) -> str:
        """Format user email to be GCP resource name safe"""
        if not email:
            return "anonymous"
        # Replace special characters with underscores
        safe_email = email.replace('@', '_at_').replace('.', '_').replace('-', '_')
        return safe_email

    def _get_gcp_username(self) -> str:
        """Get username from service account or environment"""
        try:
            # Try to get from service account email
            service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if service_account_path and os.path.exists(service_account_path):
                with open(service_account_path, 'r') as f:
                    sa_info = json.load(f)
                    email = sa_info.get('client_email', '')
                    if email:
                        username = email.split('@')[0].replace('-', '_').replace('.', '_')
                        self.logger.info(f"Using service account: {username}")
                        return username
            
            # Fallback to project ID
            project_id = os.getenv("GCP_PROJECT_ID", "unknown_project")
            username = f"sa_{project_id}".replace('-', '_').replace('.', '_')
            self.logger.info(f"Using project-based username: {username}")
            return username
            
        except Exception as e:
            self.logger.error(f"Failed to get GCP username: {str(e)}")
            return "unknown_user"

    def _initialize_gcp_auth(self):
        """Initialize GCP credentials using service account"""
        try:
            service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if not service_account_path or not os.path.exists(service_account_path):
                raise FileNotFoundError(f"Service account file not found: {service_account_path}")
            
            # Use service account credentials
            credentials = service_account.Credentials.from_service_account_file(
                service_account_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            
            self.logger.info("GCP authentication successful using service account")
            return credentials
            
        except Exception as e:
            self.logger.error(f"GCP auth failed: {str(e)}")
            # Don't raise error, just return None to allow application to continue
            self.logger.warning("Continuing without GCP authentication")
            return None

    async def initialize(self):
        """Initialize client (no-op; kept for API compatibility)."""
        self.logger.info("Client initialized")

    def _get_bq_semaphore(self) -> asyncio.Semaphore:
        """Lazily initialize the BigQuery semaphore."""
        if self._bq_semaphore is None:
            self._bq_semaphore = asyncio.Semaphore(self.bq_max_concurrency)
        return self._bq_semaphore
    
    def _enqueue_bq_write(self, description: str, func, *args, **kwargs):
        """Schedule a BigQuery write on a background thread to avoid blocking responses."""
        if not self.bq_manager:
            self.logger.warning(f"Skipping BigQuery task '{description}': manager not available")
            return
        
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No active event loop (e.g., synchronous script execution); fall back to direct call
            try:
                func(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"BigQuery task '{description}' failed in synchronous fallback: {str(e)}")
            return
        
        async def runner():
            try:
                async with self._get_bq_semaphore():
                    await asyncio.to_thread(func, *args, **kwargs)
            except Exception as e:
                self.logger.error(f"BigQuery task '{description}' failed: {str(e)}")
        
        task = loop.create_task(runner())
        self._pending_bq_tasks.append(task)
        task.add_done_callback(
            lambda t: self._pending_bq_tasks.remove(t) if t in self._pending_bq_tasks else None
        )
    
    async def _drain_bq_tasks(self):
        """Await any pending BigQuery tasks during graceful shutdown."""
        pending = [task for task in self._pending_bq_tasks if not task.done()]
        if not pending:
            return
        
        self.logger.info(f"Waiting for {len(pending)} pending BigQuery task(s) to finish")
        results = await asyncio.gather(*pending, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Pending BigQuery task raised error: {result}")
        self._pending_bq_tasks = [task for task in self._pending_bq_tasks if not task.done()]

    async def process_query(self, query: str) -> str:
        """Process query with fresh context each session"""
        # Add timestamp to the message
        timestamp = datetime.utcnow().isoformat() + 'Z'
        self._add_message("user", query, timestamp)
        
        # Save user message to BigQuery (individual message storage)
        user_message_id = str(uuid.uuid4())
        self._enqueue_bq_write(
            "save_user_message",
            self.bq_manager.save_conversation_message,
            conversation_id=self.conversation_id,
            user_id=self.user_id,
            message_id=user_message_id,
            role="user",
            content=query,
            metadata={"session_id": self.conversation_id}
        )
        
        # Check if user is asking for references
        if self._is_reference_request(query):
            references_response = self.rag.format_references()
            self._add_message("assistant", references_response, datetime.utcnow().isoformat() + 'Z')
            # Save assistant message to BigQuery
            self._save_assistant_message(references_response)
            return references_response
        
        try:
            context = self._get_conversation_context()
            rag_response = await self.rag.query(query, context)
            if rag_response:
                self._add_message("assistant", rag_response, datetime.utcnow().isoformat() + 'Z')
                # Save assistant message to BigQuery
                self._save_assistant_message(rag_response)
                return rag_response

            no_result_msg = "I couldn't find relevant information for your query. Please try rephrasing or asking a different question."
            self._add_message("assistant", no_result_msg, datetime.utcnow().isoformat() + 'Z')
            self._save_assistant_message(no_result_msg)
            return no_result_msg

        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            self._add_message("assistant", error_msg, datetime.utcnow().isoformat() + 'Z')
            # Save assistant message to BigQuery
            self._save_assistant_message(error_msg)
            return error_msg

    def _save_assistant_message(self, content: str):
        """Queue assistant message persistence in BigQuery."""
        assistant_message_id = str(uuid.uuid4())
        self._enqueue_bq_write(
            "save_assistant_message",
            self.bq_manager.save_conversation_message,
            conversation_id=self.conversation_id,
            user_id=self.user_id,
            message_id=assistant_message_id,
            role="assistant",
            content=content,
            metadata={"session_id": self.conversation_id}
        )

    def _add_message(self, role: str, content: str, timestamp: str):
        """Add message to current session history with timestamp"""
        self._chat_history.append(ChatMessage(role=role, content=content, timestamp=timestamp))
        # Limit the chat history to context window size * 2 + 2 (for safety margin)
        max_history_size = self._context_window * 2 + 2
        if len(self._chat_history) > max_history_size:
            self._chat_history = self._chat_history[-max_history_size:]

    # Public helpers for background client usage
    def add_message_entry(self, role: str, content: str, timestamp: str):
        self._add_message(role, content, timestamp)

    def save_conversation_session_now(self):
        self._save_conversation_session_at_transition()

    def is_conversation_session_saved(self) -> bool:
        return self._conversation_session_saved

    def set_conversation_session_saved(self, value: bool):
        self._conversation_session_saved = value

    def set_conversation_id(self, conversation_id: str):
        self.conversation_id = conversation_id

    def get_conversation_id(self) -> str:
        return self.conversation_id

    def _get_conversation_context(self) -> str:
        """Get context from current session only"""
        if not self._chat_history:
            return ""
        
        # Limit context to the last context_window * 2 messages (user and assistant pairs)
        context_messages = self._chat_history[-(self._context_window * 2):]
        return "\n".join(
            f"{msg.role}: {msg.content}" 
            for msg in context_messages
        )

    async def close(self):
        """Cleanup resources and save complete conversation session"""
        try:
            # Save the conversation session before closing
            if self._chat_history:
                self._save_conversation_session_at_transition()

            await self._drain_bq_tasks()
            self.logger.info("Client shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
            raise

    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get current session history"""
        return [msg.__dict__ for msg in self._chat_history]

    def clear_chat_history(self):
        """Clear current session history and create new conversation"""
        # Save the current conversation session before clearing
        if self._chat_history:
            self._save_conversation_session_at_transition()
        
        # Clear the chat history
        self._chat_history = []
        
        # Generate a new conversation ID for the new session
        old_conversation_id = self.conversation_id
        self.conversation_id = str(uuid.uuid4())
        
        # Reset the saved flag for the new conversation
        self._conversation_session_saved = False
        self._conversation_session_message_count = 0
        
        # Clear folder filters when starting new chat
        self.clear_folder_filters()
        
        self.logger.info(f"Chat history cleared. Ended conversation {old_conversation_id[:8]}, started new conversation {self.conversation_id[:8]}")

    def get_available_folders(self) -> List[str]:
        """
        Get list of available folders for document filtering.
        
        Returns:
            List of folder names available in the document collection
        """
        try:
            folders = self.rag.get_available_folders()
            self.logger.info(f"Retrieved {len(folders)} available folders")
            return folders
        except Exception as e:
            self.logger.error(f"Error getting available folders: {e}")
            return []
    
    def set_folder_filters(self, folder_names: List[str]):
        """
        Set the active folder filters for document search.
        
        Args:
            folder_names: List of folder names to filter by
        """
        try:
            self.rag.set_folder_filters(folder_names)
            self.logger.info(f"Folder filters set to: {folder_names}")
        except Exception as e:
            self.logger.error(f"Error setting folder filters: {e}")
    
    def add_folder_filter(self, folder_name: str):
        """
        Add a folder to the active filters.
        
        Args:
            folder_name: The folder name to add
        """
        try:
            self.rag.add_folder_filter(folder_name)
            self.logger.info(f"Added folder filter: {folder_name}")
        except Exception as e:
            self.logger.error(f"Error adding folder filter: {e}")
    
    def remove_folder_filter(self, folder_name: str):
        """
        Remove a folder from the active filters.
        
        Args:
            folder_name: The folder name to remove
        """
        try:
            self.rag.remove_folder_filter(folder_name)
            self.logger.info(f"Removed folder filter: {folder_name}")
        except Exception as e:
            self.logger.error(f"Error removing folder filter: {e}")
    
    def get_current_folder_filters(self) -> List[str]:
        """
        Get the currently active folder filters.
        
        Returns:
            List of active folder filter names
        """
        try:
            return self.rag.get_current_folder_filters()
        except Exception as e:
            self.logger.error(f"Error getting current folder filters: {e}")
            return []
    
    def clear_folder_filters(self):
        """Clear all active folder filters."""
        try:
            self.rag.clear_folder_filters()
            self.logger.info("Folder filters cleared")
        except Exception as e:
            self.logger.error(f"Error clearing folder filters: {e}")

    def load_conversation_session(self, conversation_id: str):
        """Load a conversation session from BigQuery with caching"""
        try:
            # Check cache first
            import time
            current_time = time.time()
            cache_timeout = 300  # 5 minutes
            
            # Check if we have a cached version that's still valid
            if (conversation_id in self._conversation_session_cache and 
                conversation_id in self._conversation_session_cache_timestamp and
                (current_time - self._conversation_session_cache_timestamp[conversation_id]) < cache_timeout):
                session_data = self._conversation_session_cache[conversation_id]
                self.logger.info(f"Loaded conversation session {conversation_id[:8]} from cache")
            else:
                # Get the complete conversation session from BigQuery
                session_data = self.bq_manager.get_conversation_session(conversation_id)
                
                # Cache the result
                self._conversation_session_cache[conversation_id] = session_data
                self._conversation_session_cache_timestamp[conversation_id] = current_time
                self.logger.info(f"Cached conversation session {conversation_id[:8]}")
            
            if session_data and 'messages' in session_data:
                # Clear current chat history
                self._chat_history = []
                
                # Load messages into the current session
                messages = session_data['messages']
                for msg in messages:
                    # Create ChatMessage objects from the stored data
                    chat_msg = ChatMessage(
                        role=msg['role'],
                        content=msg['content'],
                        timestamp=msg.get('timestamp', datetime.utcnow().isoformat() + 'Z')
                    )
                    self._chat_history.append(chat_msg)
                
                # Update the conversation ID to match the loaded session
                self.conversation_id = conversation_id
                
                # Mark the session as loaded (not saved yet in terms of new changes)
                self._conversation_session_saved = False
                
                self.logger.info(f"Loaded conversation session {conversation_id[:8]} with {len(messages)} messages")
                return True
            else:
                self.logger.warning(f"No messages found for conversation session {conversation_id[:8]}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading conversation session {conversation_id[:8]}: {str(e)}")
            return False

    def refresh_conversation_session_cache(self, conversation_id: str):
        """Force refresh the cache for a specific conversation session"""
        try:
            # Get the complete conversation session from BigQuery
            session_data = self.bq_manager.get_conversation_session(conversation_id)
            
            # Cache the result
            import time
            self._conversation_session_cache[conversation_id] = session_data
            self._conversation_session_cache_timestamp[conversation_id] = time.time()
            
            self.logger.info(f"Refreshed cache for conversation session {conversation_id[:8]}")
            return session_data
        except Exception as e:
            self.logger.error(f"Error refreshing cache for conversation session {conversation_id[:8]}: {str(e)}")
            return None

    def clear_conversation_session_cache(self):
        """Clear the conversation session cache"""
        self._conversation_session_cache.clear()
        self._conversation_session_cache_timestamp.clear()
        self.logger.info("Cleared conversation session cache")

    def _save_conversation_session_at_transition(self):
        """Save the conversation session when transitioning between conversations"""
        try:
            if not self._chat_history:
                self.logger.info("No chat history to save, skipping conversation session save")
                return
            
            # Convert ChatMessage objects to dictionaries for JSON serialization
            messages_data = [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp
                }
                for msg in self._chat_history
            ]
            
            # Add metadata for UI display
            session_metadata = {
                "context_window": self._context_window,
                "gcp_username": self.gcp_username,
                "authenticated_user": self.authenticated_user,
                "session_id": self.conversation_id,
                "message_count": len(self._chat_history),
                "end_reason": "user_initiated_new_conversation"
            }
            
            # Generate a summary of the conversation (first user message as topic)
            summary = None
            for msg in self._chat_history:
                if msg.role == "user":
                    summary = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                    break
            
            # Save the complete conversation session as a single record asynchronously
            self._enqueue_bq_write(
                "save_conversation_session",
                self.bq_manager.save_conversation_session,
                conversation_id=self.conversation_id,
                user_id=self.user_id,
                messages=messages_data,
                metadata=session_metadata,
                summary=summary
            )
            
            self._conversation_session_saved = True
            self.logger.info(f"Saved conversation session {self.conversation_id[:8]} with {len(self._chat_history)} messages at conversation transition")
            
        except Exception as e:
            self.logger.error(f"Failed to save conversation session at transition: {str(e)}")

    def get_context_window_size(self) -> int:
        """Get the current context window size"""
        return self._context_window

    def set_context_window_size(self, size: int):
        """Set the context window size"""
        if size > 0:
            self._context_window = size
            self.logger.info(f"Context window size set to {size}")
        else:
            self.logger.warning("Invalid context window size. Must be greater than 0.")

    def _is_reference_request(self, query: str) -> bool:
        """Check if the query is asking for references"""
        reference_keywords = ["reference", "source", "cite", "citation"]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in reference_keywords)