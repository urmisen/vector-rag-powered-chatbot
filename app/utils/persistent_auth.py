"""Persistent authentication storage using file system"""
import json
import time
import hashlib
from typing import Optional, Dict
import streamlit as st
from pathlib import Path
import platform


# Use a persistent file-based storage for auth state
AUTH_CACHE_DIR = Path.home() / ".mcp_chatbot_auth"
AUTH_CACHE_DIR.mkdir(exist_ok=True)


def get_machine_id() -> str:
    """Get a unique identifier for this machine"""
    # Create a hash based on machine-specific information
    machine_info = f"{platform.node()}_{platform.machine()}_{platform.system()}"
    return hashlib.sha256(machine_info.encode()).hexdigest()[:16]


def get_browser_id() -> str:
    """Alias for get_machine_id() for backwards compatibility"""
    return get_machine_id()


def _get_auth_file_path() -> Path:
    """Get the path to the auth cache file for this machine"""
    machine_id = get_machine_id()
    return AUTH_CACHE_DIR / f"auth_{machine_id}.json"


def _serialize_token_data(token_data: Dict) -> Dict:
    """Convert token data to JSON-serializable format"""
    if not token_data:
        return {}
    
    serializable = {}
    for key, value in token_data.items():
        # Convert datetime objects to ISO format string
        if hasattr(value, 'isoformat'):
            serializable[key] = value.isoformat()
        # Skip non-serializable objects, keep simple types
        elif isinstance(value, (str, int, float, bool, type(None))):
            serializable[key] = value
        # Try to convert to string as fallback
        elif value is not None:
            try:
                # Test if it's JSON serializable
                json.dumps(value)
                serializable[key] = value
            except (TypeError, ValueError):
                # Skip non-serializable values
                print(f"Skipping non-serializable field: {key}")
                pass
    
    return serializable


def store_persistent_auth(email: str, token_data: Dict) -> bool:
    """Store authentication data in local file system"""
    try:
        machine_id = get_machine_id()
        auth_file = _get_auth_file_path()
        
        # Serialize token data to handle datetime and other objects
        serializable_token_data = _serialize_token_data(token_data)
        
        auth_data = {
            'email': email,
            'machine_id': machine_id,
            'timestamp': int(time.time()),
            'token_data': serializable_token_data,
            'expires_in_days': 30  # 30 days validity
        }
        
        # Write to file with proper JSON serialization
        with open(auth_file, 'w') as f:
            json.dump(auth_data, f, indent=2)
        
        # Store in session state as well
        st.session_state['stored_auth_email'] = email
        st.session_state['stored_auth_timestamp'] = int(time.time())
        st.session_state['persistent_auth_active'] = True
        
        print(f"Stored persistent auth for {email} in {auth_file}")
        return True
        
    except Exception as e:
        print(f"Error storing persistent auth: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_persistent_auth() -> Optional[Dict]:
    """Retrieve authentication data from local file system"""
    try:
        # First check session state cache (for current session)
        if st.session_state.get('persistent_auth_active'):
            if 'stored_auth_email' in st.session_state and 'stored_auth_timestamp' in st.session_state:
                email = st.session_state['stored_auth_email']
                timestamp = st.session_state['stored_auth_timestamp']
                machine_id = get_machine_id()
                
                # Check if not expired (30 days)
                current_time = int(time.time())
                days_elapsed = (current_time - timestamp) / (24 * 60 * 60)
                
                if days_elapsed < 30:
                    print(f"Retrieved persistent auth from session state for {email}")
                    return {
                        'email': email,
                        'machine_id': machine_id,
                        'timestamp': timestamp,
                        'token_data': {'email': email}
                    }
        
        # Try to load from file (for new sessions/browser reloads)
        auth_file = _get_auth_file_path()
        print(f"Checking for persistent auth file: {auth_file}")
        
        if auth_file.exists():
            with open(auth_file, 'r') as f:
                auth_data = json.load(f)
            
            print(f"Found auth file with email: {auth_data.get('email')}")
            
            # Validate and restore to session state
            if is_persistent_auth_valid(auth_data):
                st.session_state['stored_auth_email'] = auth_data['email']
                st.session_state['stored_auth_timestamp'] = auth_data['timestamp']
                st.session_state['persistent_auth_active'] = True
                print(f"Restored persistent auth for {auth_data['email']}")
                return auth_data
            else:
                # Clean up expired auth file
                print(f"Auth file expired, cleaning up")
                auth_file.unlink()
        else:
            print(f"No auth file found at {auth_file}")
        
        return None
        
    except Exception as e:
        print(f"Error retrieving persistent auth: {e}")
        import traceback
        traceback.print_exc()
        return None


def clear_persistent_auth() -> bool:
    """Clear authentication data from file system and session"""
    try:
        # Clear file
        auth_file = _get_auth_file_path()
        print(f"Clearing auth file: {auth_file}")
        if auth_file.exists():
            auth_file.unlink()
            print(f"Deleted auth file: {auth_file}")
        
        # Clear from session state
        for key in ['persistent_auth_checked', 'stored_auth_email', 
                   'stored_auth_timestamp', 'persistent_auth_active']:
            if key in st.session_state:
                del st.session_state[key]
        
        # Try to clean up all auth files older than 30 days
        try:
            current_time = time.time()
            for auth_file in AUTH_CACHE_DIR.glob("auth_*.json"):
                if auth_file.exists():
                    file_age = current_time - auth_file.stat().st_mtime
                    if file_age > (30 * 24 * 60 * 60):  # 30 days
                        auth_file.unlink()
        except Exception:
            pass  # Ignore cleanup errors
        
        print("Persistent auth cleared successfully")
        return True
        
    except Exception as e:
        print(f"Error clearing persistent auth: {e}")
        import traceback
        traceback.print_exc()
        return False


def is_persistent_auth_valid(auth_data: Optional[Dict]) -> bool:
    """Check if persistent auth data is valid and not expired"""
    if not auth_data:
        return False
    
    required_fields = ['email', 'timestamp']
    if not all(field in auth_data for field in required_fields):
        return False
    
    # Check expiry (30 days)
    timestamp = auth_data.get('timestamp', 0)
    current_time = int(time.time())
    days_elapsed = (current_time - timestamp) / (24 * 60 * 60)
    
    if days_elapsed >= 30:
        return False
    
    return True

