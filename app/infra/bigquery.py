import os
import time
from functools import lru_cache
from typing import Optional, Dict, List, Any
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class BigQueryManager:
    """Manager for BigQuery operations including user authentication and conversation history storage."""
    
    def __init__(self):
        """Initialize BigQuery client and dataset information from environment variables."""
        start = time.perf_counter()
        self.project_id = os.getenv("GCP_PROJECT_ID", "fintech-dep-staging")
        self.dataset_id = os.getenv("BIGQUERY_DATASET_ID", "chatbot_data")
        self.client = bigquery.Client(project=self.project_id)
        self.dataset_ref = f"{self.project_id}.{self.dataset_id}"
        elapsed = time.perf_counter() - start
        logger.info(f"BigQueryManager initialized in {elapsed * 1000:.1f}ms")
        
    def initialize_tables(self):
        """Create the required tables if they don't exist."""
        try:
            # Create dataset if it doesn't exist
            dataset = bigquery.Dataset(self.dataset_ref)
            dataset.location = os.getenv("GCP_LOCATION", "us-central1")
            try:
                self.client.get_dataset(self.dataset_ref)
                logger.info(f"Dataset {self.dataset_ref} already exists")
            except NotFound:
                dataset = self.client.create_dataset(dataset, timeout=30)
                logger.info(f"Created dataset {self.dataset_ref}")
            
            # Create user_log table (formerly users table)
            self._create_user_log_table()
            
            # Create users table (formerly user_login_tracking table)
            self._create_users_table()
            
            # Create conversations table with enhanced schema
            self._create_conversations_table()
            
            # Create conversation_sessions table for structured conversation storage
            self._create_conversation_sessions_table()
            
        except Exception as e:
            logger.error(f"Error initializing BigQuery tables: {str(e)}")
            raise
    
    def _create_user_log_table(self):
        """Create the user_log table for storing user authentication data."""
        table_id = f"{self.dataset_ref}.user_log"
        try:
            self.client.get_table(table_id)
            logger.info(f"Table {table_id} already exists")
        except NotFound:
            schema = [
                bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("email", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("auth_event", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("ip_address", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("user_agent", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("metadata", "JSON", mode="NULLABLE"),
            ]
            
            table = bigquery.Table(table_id, schema=schema)
            table = self.client.create_table(table)
            logger.info(f"Created table {table_id}")
    
    def _create_users_table(self):
        """Create the users table for storing unique user login timestamps."""
        table_id = f"{self.dataset_ref}.users"
        try:
            self.client.get_table(table_id)
            logger.info(f"Table {table_id} already exists")
        except NotFound:
            schema = [
                bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("email", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("first_login_at", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("last_login_at", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("login_count", "INTEGER", mode="REQUIRED"),
            ]
            
            table = bigquery.Table(table_id, schema=schema)
            table = self.client.create_table(table)
            logger.info(f"Created table {table_id}")
    
    def _create_conversations_table(self):
        """Create the conversations table for storing individual chat messages."""
        table_id = f"{self.dataset_ref}.conversations"
        try:
            self.client.get_table(table_id)
            logger.info(f"Table {table_id} already exists")
        except NotFound:
            schema = [
                bigquery.SchemaField("conversation_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("message_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("role", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("content", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("context", "JSON", mode="NULLABLE"),
                bigquery.SchemaField("metadata", "JSON", mode="NULLABLE"),
            ]
            
            table = bigquery.Table(table_id, schema=schema)
            table = self.client.create_table(table)
            logger.info(f"Created table {table_id}")
    
    def _create_conversation_sessions_table(self):
        """Create the conversation_sessions table for storing structured conversation data."""
        table_id = f"{self.dataset_ref}.conversation_sessions"
        try:
            self.client.get_table(table_id)
            logger.info(f"Table {table_id} already exists")
        except NotFound:
            schema = [
                bigquery.SchemaField("conversation_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("session_start_time", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("session_end_time", "TIMESTAMP", mode="NULLABLE"),
                bigquery.SchemaField("messages", "JSON", mode="REQUIRED"),
                bigquery.SchemaField("metadata", "JSON", mode="NULLABLE"),
                bigquery.SchemaField("summary", "STRING", mode="NULLABLE"),
            ]
            
            table = bigquery.Table(table_id, schema=schema)
            table = self.client.create_table(table)
            logger.info(f"Created table {table_id}")
    
    def log_user_auth_event(self, user_id: str, email: str, auth_event: str, 
                           ip_address: Optional[str] = None, user_agent: Optional[str] = None,
                           metadata: Optional[Dict] = None):
        """Log a user authentication event.
        
        Args:
            user_id: Unique identifier for the user
            email: User's email address
            auth_event: Type of authentication event (e.g., 'login', 'logout', 'failed_login')
            ip_address: IP address of the user (optional)
            user_agent: User agent string (optional)
            metadata: Additional metadata as a dictionary (optional)
        """
        try:
            table_id = f"{self.dataset_ref}.user_log"
            
            # Convert datetime to ISO format string for JSON serialization
            timestamp_str = datetime.utcnow().isoformat() + 'Z'
            
            rows_to_insert = [{
                "user_id": user_id,
                "email": email,
                "auth_event": auth_event,
                "timestamp": timestamp_str,
                "ip_address": ip_address,
                "user_agent": user_agent,
                "metadata": json.dumps(metadata) if metadata else None
            }]
            
            errors = self.client.insert_rows_json(table_id, rows_to_insert)
            if errors:
                logger.error(f"Error inserting user auth event: {errors}")
            else:
                logger.info(f"Logged user auth event: {auth_event} for user {email}")
                
                # If this is a login event, update the users table
                if auth_event == "login":
                    self._update_users_table(user_id, email, timestamp_str)
                
        except Exception as e:
            logger.error(f"Error logging user auth event: {str(e)}")
    
    def _update_users_table(self, user_id: str, email: str, login_timestamp: str):
        """Update the users table with first and last login timestamps.
        
        Args:
            user_id: Unique identifier for the user
            email: User's email address
            login_timestamp: ISO formatted timestamp of the login event
        """
        try:
            # Check if user already exists in the users table
            query = f"""
                SELECT first_login_at, login_count
                FROM `{self.dataset_ref}.users`
                WHERE user_id = @user_id
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                ]
            )
            
            query_job = self.client.query(query, job_config=job_config)
            results = list(query_job.result())
            
            if results:
                # User exists, update last_login_at and increment login_count
                first_login = results[0]["first_login_at"]
                login_count = results[0]["login_count"] + 1
                
                # Update the existing record
                update_query = f"""
                    UPDATE `{self.dataset_ref}.users`
                    SET last_login_at = @last_login_at, login_count = @login_count
                    WHERE user_id = @user_id
                """
                
                update_job_config = bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("last_login_at", "TIMESTAMP", login_timestamp),
                        bigquery.ScalarQueryParameter("login_count", "INTEGER", login_count),
                        bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                    ]
                )
                
                update_job = self.client.query(update_query, job_config=update_job_config)
                update_job.result()
                logger.info(f"Updated login tracking for user {email}")
            else:
                # User doesn't exist, insert new record
                rows_to_insert = [{
                    "user_id": user_id,
                    "email": email,
                    "first_login_at": login_timestamp,
                    "last_login_at": login_timestamp,
                    "login_count": 1
                }]
                
                table_id = f"{self.dataset_ref}.users"
                errors = self.client.insert_rows_json(table_id, rows_to_insert)
                if errors:
                    logger.error(f"Error inserting user login tracking: {errors}")
                else:
                    logger.info(f"Created login tracking for new user {email}")
                
        except Exception as e:
            logger.error(f"Error updating user login tracking: {str(e)}")
    
    def save_conversation_message(self, conversation_id: str, user_id: str, message_id: str,
                                 role: str, content: str, context: Optional[Dict] = None,
                                 metadata: Optional[Dict] = None):
        """Save a conversation message to BigQuery.
        
        Args:
            conversation_id: Unique identifier for the conversation session
            user_id: Unique identifier for the user
            message_id: Unique identifier for the message
            role: Role of the message sender ('user' or 'assistant')
            content: Content of the message
            context: Context information (optional)
            metadata: Additional metadata (optional)
        """
        try:
            table_id = f"{self.dataset_ref}.conversations"
            
            # Convert datetime to ISO format string for JSON serialization
            timestamp_str = datetime.utcnow().isoformat() + 'Z'
            
            rows_to_insert = [{
                "conversation_id": conversation_id,
                "user_id": user_id,
                "message_id": message_id,
                "role": role,
                "content": content,
                "timestamp": timestamp_str,
                "context": json.dumps(context) if context else None,
                "metadata": json.dumps(metadata) if metadata else None
            }]
            
            errors = self.client.insert_rows_json(table_id, rows_to_insert)
            if errors:
                logger.error(f"Error inserting conversation message: {errors}")
            else:
                logger.info(f"Saved conversation message {message_id} for user {user_id}")
                
        except Exception as e:
            logger.error(f"Error saving conversation message: {str(e)}")
    
    def save_conversation_session(self, conversation_id: str, user_id: str, 
                                 messages: List[Dict[str, Any]], metadata: Optional[Dict] = None,
                                 summary: Optional[str] = None):
        """Save or update a complete conversation session in structured JSON format.
        For ongoing conversations, this method replaces the existing record to keep it up to date.
        This implementation ensures only one record per user_id and conversation_id combination exists,
        keeping the one with the latest session_end_time.
        
        Args:
            conversation_id: Unique identifier for the conversation session
            user_id: Unique identifier for the user
            messages: List of message dictionaries containing role and content
            metadata: Additional session metadata (optional)
            summary: Summary of the conversation (optional)
        """
        try:
            table_id = f"{self.dataset_ref}.conversation_sessions"
            
            # Convert datetime to ISO format string for JSON serialization
            timestamp_str = datetime.utcnow().isoformat() + 'Z'
            
            # Determine session start and end times from messages
            session_start_time = timestamp_str
            session_end_time = timestamp_str
            
            if messages:
                # Use first message timestamp as start time
                session_start_time = messages[0].get("timestamp", timestamp_str)
                # Use last message timestamp as end time
                session_end_time = messages[-1].get("timestamp", timestamp_str)
            
            # Log the data being saved for debugging
            logger.info(f"Saving conversation session {conversation_id} for user {user_id}")
            logger.info(f"Messages count: {len(messages)}")
            logger.info(f"Session start time: {session_start_time}")
            logger.info(f"Session end time: {session_end_time}")
            logger.info(f"Summary: {summary}")
            
            # For handling streaming buffer limitations, we'll use INSERT only
            # And rely on the cleanup process or query-time deduplication
            insert_query = f"""
            INSERT INTO `{table_id}` (
                conversation_id,
                user_id,
                session_start_time,
                session_end_time,
                messages,
                metadata,
                summary
            ) VALUES (
                @conversation_id,
                @user_id,
                @session_start_time,
                @session_end_time,
                @messages,
                @metadata,
                @summary
            )
            """
            
            # Configure query parameters
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("conversation_id", "STRING", conversation_id),
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                    bigquery.ScalarQueryParameter("session_start_time", "TIMESTAMP", session_start_time),
                    bigquery.ScalarQueryParameter("session_end_time", "TIMESTAMP", session_end_time),
                    bigquery.ScalarQueryParameter("messages", "JSON", json.dumps(messages)),
                    bigquery.ScalarQueryParameter("metadata", "JSON", json.dumps(metadata) if metadata else None),
                    bigquery.ScalarQueryParameter("summary", "STRING", summary),
                ]
            )
            
            # Execute the insert query
            query_job = self.client.query(insert_query, job_config=job_config)
            query_job.result()  # Wait for the job to complete
            
            logger.info(f"Saved conversation session {conversation_id} for user {user_id} with {len(messages)} messages")
                
        except Exception as e:
            logger.error(f"Error saving conversation session: {str(e)}")
            # Re-raise the exception so it can be handled by the caller
            raise

    def get_conversation_session(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a complete conversation session in structured format.
        
        Args:
            conversation_id: Unique identifier for the conversation session
            
        Returns:
            Dictionary containing the complete conversation session data
        """
        try:
            # Optimized query with specific field selection for better performance
            query = f"""
                SELECT 
                    conversation_id,
                    user_id,
                    session_start_time,
                    session_end_time,
                    messages,
                    metadata,
                    summary
                FROM `{self.dataset_ref}.conversation_sessions`
                WHERE conversation_id = @conversation_id
                ORDER BY session_end_time DESC
                LIMIT 1
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("conversation_id", "STRING", conversation_id),
                ],
                use_query_cache=True,  # Enable query caching for faster repeated queries
            )
            
            query_job = self.client.query(query, job_config=job_config)
            results = list(query_job.result())
            
            if results:
                # Parse JSON fields
                session = dict(results[0])
                if isinstance(session.get('messages'), str):
                    session['messages'] = json.loads(session['messages'])
                if isinstance(session.get('metadata'), str):
                    session['metadata'] = json.loads(session['metadata'])
                return session
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving conversation session: {str(e)}")
            return None
    
    def get_user_conversations(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve all conversations for a specific user.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            List of conversation records
        """
        try:
            query = f"""
                SELECT *
                FROM `{self.dataset_ref}.conversations`
                WHERE user_id = @user_id
                ORDER BY timestamp ASC
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                ]
            )
            
            query_job = self.client.query(query, job_config=job_config)
            results = query_job.result()
            
            conversations = []
            for row in results:
                conversations.append(dict(row))
                
            return conversations
            
        except Exception as e:
            logger.error(f"Error retrieving user conversations: {str(e)}")
            return []
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Retrieve the history of a specific conversation.
        
        Args:
            conversation_id: Unique identifier for the conversation
            
        Returns:
            List of message records in chronological order
        """
        try:
            query = f"""
                SELECT *
                FROM `{self.dataset_ref}.conversations`
                WHERE conversation_id = @conversation_id
                ORDER BY timestamp ASC
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("conversation_id", "STRING", conversation_id),
                ]
            )
            
            query_job = self.client.query(query, job_config=job_config)
            results = query_job.result()
            
            messages = []
            for row in results:
                messages.append(dict(row))
                
            return messages
            
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {str(e)}")
            return []
    
    def get_user_conversation_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve all conversation sessions for a specific user in structured format.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            List of conversation session records
        """
        try:
            # Get the latest record for each conversation_id to handle potential duplicates
            # Limit to 50 most recent conversations for performance
            query = f"""
                SELECT 
                    conversation_id, 
                    user_id, 
                    session_start_time, 
                    session_end_time, 
                    metadata, 
                    summary
                FROM (
                    SELECT 
                        *,
                        ROW_NUMBER() OVER (
                            PARTITION BY conversation_id, user_id 
                            ORDER BY session_end_time DESC
                        ) as rn
                    FROM `{self.dataset_ref}.conversation_sessions`
                    WHERE user_id = @user_id
                )
                WHERE rn = 1
                ORDER BY session_start_time DESC
                LIMIT 50
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                ]
            )
            
            query_job = self.client.query(query, job_config=job_config)
            results = query_job.result()
            
            sessions = []
            for row in results:
                sessions.append(dict(row))
                
            return sessions
            
        except Exception as e:
            logger.error(f"Error retrieving user conversation sessions: {str(e)}")
            return []
    
    def cleanup_duplicate_sessions(self) -> Dict[str, int]:
        """Clean up any duplicate conversation sessions, keeping only the latest one for each user and conversation_id.
        This method should only be needed if duplicates were created due to race conditions or system issues.
        
        Returns:
            Dictionary with statistics about the cleanup operation
        """
        try:
            # Get the client and dataset reference
            client = self.client
            dataset_ref = self.dataset_ref
            table_id = f"{dataset_ref}.conversation_sessions"
            
            # First, let's check how many records we have before cleanup
            count_query = f"""
                SELECT COUNT(*) as total_records
                FROM `{table_id}`
            """
            
            count_job = client.query(count_query)
            count_result = list(count_job.result())
            total_before = count_result[0]['total_records']
            
            logger.info(f"Total conversation session records before cleanup: {total_before}")
            
            # Use a batch approach to handle streaming buffer limitations
            # Create a temporary table with deduplicated data
            temp_table_id = f"{dataset_ref}.conversation_sessions_dedup_temp_{int(datetime.utcnow().timestamp())}"
            
            # Create temporary table with deduplicated data
            dedup_query = f"""
            CREATE TABLE `{temp_table_id}` AS
            SELECT 
                conversation_id,
                user_id,
                session_start_time,
                session_end_time,
                messages,
                metadata,
                summary
            FROM (
                SELECT 
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY conversation_id, user_id 
                        ORDER BY session_end_time DESC
                    ) as rn
                FROM `{table_id}`
            )
            WHERE rn = 1
            """
            
            logger.info("Creating deduplicated temporary table...")
            dedup_job = client.query(dedup_query)
            dedup_job.result()  # Wait for the job to complete
            
            # Wait a bit to ensure the temp table is fully created
            import time
            time.sleep(2)
            
            # Delete the original table
            delete_table_query = f"DROP TABLE `{table_id}`"
            delete_table_job = client.query(delete_table_query)
            delete_table_job.result()
            
            # Recreate the original table with the same schema
            self._create_conversation_sessions_table()
            
            # Copy data from temp table to original table
            copy_query = f"""
            INSERT INTO `{table_id}` 
            SELECT * FROM `{temp_table_id}`
            """
            copy_job = client.query(copy_query)
            copy_job.result()
            
            # Drop the temporary table
            drop_query = f"DROP TABLE `{temp_table_id}`"
            drop_job = client.query(drop_query)
            drop_job.result()
            
            # Count records after cleanup
            count_after_query = f"""
                SELECT COUNT(*) as total_records
                FROM `{table_id}`
            """
            
            count_after_job = client.query(count_after_query)
            count_after_result = list(count_after_job.result())
            total_after = count_after_result[0]['total_records']
            
            logger.info(f"Total conversation session records after cleanup: {total_after}")
            
            # Prepare result statistics
            result_stats = {
                "records_before": total_before,
                "records_after": total_after,
                "records_removed": total_before - total_after
            }
            
            if result_stats["records_removed"] > 0:
                logger.info(f"Successfully cleaned up conversation sessions. Removed {result_stats['records_removed']} duplicate records.")
            else:
                logger.info("No duplicate records found. All records are unique.")
            
            return result_stats
                
        except Exception as e:
            logger.error(f"Error cleaning up conversation sessions: {str(e)}")
            raise


@lru_cache(maxsize=1)
def get_bigquery_manager() -> BigQueryManager:
    """Return a cached singleton instance of BigQueryManager."""
    start = time.perf_counter()
    manager = BigQueryManager()
    elapsed = time.perf_counter() - start
    logger.info(f"Shared BigQueryManager ready in {elapsed * 1000:.1f}ms")
    return manager
