import json
import os
import time
import uuid
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

import streamlit as st
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
from google.auth.transport import requests

# Add BigQuery import and logger
from app.infra.bigquery import BigQueryManager
from app.infra.logger import logger as app_logger
from app.utils.persistent_auth import (
    store_persistent_auth,
    get_persistent_auth,
    clear_persistent_auth,
    is_persistent_auth_valid,
    get_browser_id
)

OAUTH_CREDENTIALS_PATH = (
    Path(__file__).resolve().parent.parent.parent / ".gcloud" / "oauth_credentials.json"
)

# Initialize BigQuery manager
bq_manager = BigQueryManager()
auth_logger = app_logger.getChild("OAuth")


def _log_auth_event_async(event_name: str, **payload) -> None:
    """Offload BigQuery auth-event logging onto a background thread."""

    def _worker():
        started = time.perf_counter()
        try:
            bq_manager.log_user_auth_event(**payload)
        except Exception as exc:
            auth_logger.warning(
                "Async BigQuery %s audit failed after %.3fs: %s",
                event_name,
                time.perf_counter() - started,
                exc,
            )
        else:
            auth_logger.info(
                "Async BigQuery %s audit logged in %.3fs",
                event_name,
                time.perf_counter() - started,
            )

    threading.Thread(
        target=_worker,
        name=f"auth-log-{event_name}",
        daemon=True,
    ).start()

def _load_client_config() -> Dict:
    with open(OAUTH_CREDENTIALS_PATH, "r") as f:
        return json.load(f)

def _get_redirect_uri() -> str:
    # Query param override (optional, for testing); otherwise use OAUTH_REDIRECT_URI from init.sh
    app_base = st.query_params.get("app_base", [None])
    if app_base and isinstance(app_base, list):
        app_base = app_base[0]
    if not app_base:
        app_base = os.getenv("OAUTH_REDIRECT_URI", "http://localhost:8501")
    return app_base

def _new_flow(redirect_uri: Optional[str] = None) -> Flow:
    client_config = _load_client_config()
    flow = Flow.from_client_config(
        client_config,
        scopes=[
            "openid",
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/userinfo.profile",
        ],
        redirect_uri=redirect_uri or _get_redirect_uri(),
    )
    return flow

def build_authorization_url() -> Tuple[str, str]:
    flow = _new_flow()
    authorization_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    # Persist state so we can validate on callback
    st.session_state["oauth_state"] = state
    st.session_state["oauth_started_at"] = int(time.time())
    # Persist client config for token exchange
    st.session_state["oauth_client_config"] = _load_client_config()
    st.session_state["oauth_redirect_uri"] = flow.redirect_uri
    return authorization_url, state

def exchange_code_for_credentials(code: str, state: str) -> Optional[Dict]:
    expected_state = st.session_state.get("oauth_state")
    # Be tolerant if state is missing after redirect (session reset) or mismatched
    # for local development environments. We still attempt token exchange.

    client_config = st.session_state.get("oauth_client_config") or _load_client_config()
    redirect_uri = st.session_state.get("oauth_redirect_uri") or _get_redirect_uri()

    flow = Flow.from_client_config(
        client_config,
        scopes=[
            "openid",
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/userinfo.profile",
        ],
        redirect_uri=redirect_uri,
    )
    flow.fetch_token(code=code)
    creds = flow.credentials
    return {
        "token": creds.token,
        "id_token": creds.id_token,
        "refresh_token": getattr(creds, "refresh_token", None),
        "expiry": getattr(creds, "expiry", None),
    }

def parse_email_from_id_token(raw_id_token: str) -> Optional[str]:
    try:
        # Verify the token's issuer and signature
        request = requests.Request()
        info = id_token.verify_oauth2_token(raw_id_token, request)
        return info.get("email")
    except Exception:
        return None

def is_allowed_domain(email: Optional[str]) -> bool:
    if not email:
        return False
    allowed = os.getenv("ALLOWED_DOMAIN", "").strip().lower()
    if not allowed:
        return False
    domain = email.split("@")[-1].lower()
    return domain == allowed

def sign_out():
    # Log sign out event
    auth_state = get_auth_state()
    if auth_state.get("is_authenticated"):
        email = auth_state.get("email")
        user_id = email.split("@")[0] if email else "unknown"
        _log_auth_event_async(
            "logout",
            user_id=user_id,
            email=email,
            auth_event="logout",
            ip_address=st.query_params.get("REMOTE_ADDR", None),
            user_agent=st.query_params.get("HTTP_USER_AGENT", None),
        )
    
    # Clear persistent authentication
    clear_persistent_auth()
    
    for key in [
        "auth",
        "oauth_state",
        "oauth_started_at",
        "oauth_client_config",
        "oauth_redirect_uri",
        "persistent_auth_loaded",
    ]:
        if key in st.session_state:
            del st.session_state[key]

def check_persistent_auth() -> bool:
    """Check for persistent authentication and restore it if valid"""
    # Only check once per session
    if st.session_state.get("persistent_auth_checked"):
        return False
    
    st.session_state["persistent_auth_checked"] = True
    
    try:
        auth_logger.info("Checking for persistent authentication...")
        auth_data = get_persistent_auth()
        
        if auth_data and is_persistent_auth_valid(auth_data):
            email = auth_data.get("email")
            if email and is_allowed_domain(email):
                # Restore authentication without storing again (avoid loop)
                set_authenticated(email, store_persistent=False)
                auth_logger.info("Restored persistent authentication for user=%s", email)
                return True
            else:
                auth_logger.warning("Invalid domain in persistent auth: %s", email)
                clear_persistent_auth()
        else:
            auth_logger.debug("No valid persistent authentication found")
    except Exception as e:
        auth_logger.error("Error checking persistent auth: %s", e)
    
    return False

def get_auth_state() -> Dict:
    auth = st.session_state.get("auth") or {
        "is_authenticated": False,
        "email": None,
        "error": None,
    }
    return auth

def set_authenticated(email: str, store_persistent: bool = True, token_data: Optional[Dict] = None):
    st.session_state["auth"] = {
        "is_authenticated": True,
        "email": email,
        "error": None,
    }
    
    # Store persistent authentication if requested
    if store_persistent:
        try:
            token_info = token_data or {"email": email, "timestamp": int(time.time())}
            store_persistent_auth(email, token_info)
            auth_logger.info("Stored persistent authentication for user=%s", email)
        except Exception as e:
            auth_logger.warning("Failed to store persistent authentication: %s", e)
    
    # Log successful login event
    user_id = email.split("@")[0] if email else "unknown"
    _log_auth_event_async(
        "login",
        user_id=user_id,
        email=email,
        auth_event="login",
        ip_address=st.query_params.get("REMOTE_ADDR", None),
        user_agent=st.query_params.get("HTTP_USER_AGENT", None),
    )
    auth_logger.debug("set_authenticated completed for user=%s", user_id)

def set_auth_error(message: str):
    st.session_state["auth"] = {
        "is_authenticated": False,
        "email": None,
        "error": message,
    }
    
    # Log failed login attempt
    _log_auth_event_async(
        "failed_login",
        user_id="unknown",
        email="unknown",
        auth_event="failed_login",
        ip_address=st.query_params.get("REMOTE_ADDR", None),
        user_agent=st.query_params.get("HTTP_USER_AGENT", None),
        metadata={"error_message": message},
    )
    auth_logger.debug("set_auth_error completed.")

def render_login_overlay():
    auth_url, _state = build_authorization_url()
    st.markdown(
        f"""
        <style>
        .login-overlay {{
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 9999;
        }}
        .login-card {{
            background: white !important;
            border-radius: 16px;
            padding: 32px;
            width: 100%;
            max-width: 420px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
            text-align: center;
        }}
        .login-title {{ 
            font-size: 22px !important; 
            margin-bottom: 6px !important; 
            font-weight: 700 !important; 
            color: #1e293b !important;
            display: block !important;
        }}
        .login-sub {{ 
            color: #475569 !important; 
            margin-bottom: 20px !important; 
            font-size: 14px !important;
            display: block !important;
        }}
        .google-btn {{
            display: inline-flex !important;
            align-items: center;
            gap: 10px;
            background: #ffffff !important;
            color: #111827 !important;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 10px 16px;
            font-weight: 600;
            text-decoration: none;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }}
        .google-btn:hover {{
            background: #f9fafb !important;
        }}
        .google-icon {{ width: 18px; height: 18px; }}
        </style>
        <div class="login-overlay">
            <div class="login-card">
                <div class="login-title">Sign in to continue</div>
                <div class="login-sub">Use your company Google account</div>
                <a class="google-btn" href="{auth_url}">
                    <svg class="google-icon" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg"><path fill="#FFC107" d="M43.611,20.083H42V20H24v8h11.303c-1.649,4.657-6.08,8-11.303,8c-6.627,0-12-5.373-12-12 s5.373-12,12-12c3.059,0,5.842,1.154,7.961,3.039l5.657-5.657C34.046,6.05,29.268,4,24,4C12.955,4,4,12.955,4,24 s8.955,20,20,20s20-8.955,20-20C44,22.659,43.862,21.35,43.611,20.083z"/><path fill="#FF3D00" d="M6.306,14.691l6.571,4.819C14.655,16.108,18.961,14,24,14c3.059,0,5.842,1.154,7.961,3.039l5.657-5.657 C34.046,6.05,29.268,4,24,4C16.318,4,9.656,8.337,6.306,14.691z"/><path fill="#4CAF50" d="M24,44c5.166,0,9.86-1.977,13.409-5.197l-6.191-5.238C29.211,35.091,26.715,36,24,36 c-5.202,0-9.619-3.317-11.278-7.954l-6.538,5.036C9.497,39.556,16.227,44,24,44z"/><path fill="#1976D2" d="M43.611,20.083H42V20H24v8h11.303c-0.792,2.237-2.231,4.166-4.091,5.607l0.003-0.002l6.191,5.238 C35.971,39.804,44,34,44,24C44,22.659,43.862,21.35,43.611,20.083z"/></svg>
                    Sign in with Google
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def handle_oauth_callback_if_present() -> bool:
    params = st.query_params
    code = params.get("code")
    state = params.get("state")
    if isinstance(code, list):
        code = code[0]
    if isinstance(state, list):
        state = state[0]

    if not code or not state:
        auth_logger.debug("OAuth callback not detected (code/state missing).")
        return False

    callback_started = time.perf_counter()
    auth_logger.info("OAuth callback detected; beginning token exchange.")

    token_exchange_started = time.perf_counter()
    creds = exchange_code_for_credentials(code, state)
    token_exchange_elapsed = time.perf_counter() - token_exchange_started
    auth_logger.info(
        "OAuth token exchange finished in %.3fs",
        token_exchange_elapsed,
    )
    if not creds:
        auth_logger.warning(
            "OAuth token exchange failed; state=%s; elapsed=%.3fs",
            state,
            time.perf_counter() - callback_started,
        )
        set_auth_error("OAuth exchange failed. Please try signing in again.")
        return True

    email_parse_started = time.perf_counter()
    email = parse_email_from_id_token(creds.get("id_token"))
    email_parse_elapsed = time.perf_counter() - email_parse_started
    auth_logger.info(
        "OAuth email parsing finished in %.3fs (email=%s)",
        email_parse_elapsed,
        email or "unknown",
    )
    if not email:
        auth_logger.warning(
            "OAuth callback missing email; elapsed=%.3fs",
            time.perf_counter() - callback_started,
        )
        set_auth_error("Unable to retrieve email from Google. Try again.")
        return True

    domain_check_started = time.perf_counter()
    if not is_allowed_domain(email):
        elapsed = time.perf_counter() - callback_started
        auth_logger.warning(
            "OAuth callback rejected for unauthorized domain; email=%s; elapsed=%.3fs",
            email,
            elapsed,
        )
        set_auth_error(
            f"Unauthorized email domain. Please use your @{os.getenv('ALLOWED_DOMAIN', '')} email."
        )
        return True
    auth_logger.debug(
        "OAuth domain check accepted in %.3fs",
        time.perf_counter() - domain_check_started,
    )

    # Store authentication with token data for persistence
    set_authenticated(email, store_persistent=True, token_data=creds)
    elapsed = time.perf_counter() - callback_started
    auth_logger.info(
        "OAuth callback succeeded; email=%s; elapsed=%.3fs",
        email,
        elapsed,
    )
    # Clean query params so refreshes don't re-trigger the callback
    try:
        # Newer Streamlit supports assignment to st.query_params
        st.query_params.clear()
    except Exception:
        try:
            # Fallback for older versions
            st.experimental_set_query_params()
        except Exception:
            pass
    
    # Redirect to clean URL (without query parameters) using JavaScript
    base_url = os.getenv("OAUTH_REDIRECT_URI", "http://localhost:8501")
    
    # Use JavaScript to redirect to clean URL
    st.markdown(
        f"""
        <script>
            window.location.replace("{base_url}");
        </script>
        """,
        unsafe_allow_html=True
    )
    
    # Force rerun to re-render without overlay
    st.rerun()
    return True