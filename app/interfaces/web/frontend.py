import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load environment variables from config/.env (project root)
_env_path = Path(__file__).resolve().parent.parent.parent.parent / "config" / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

from app.infra.background_client import get_background_client_manager
from app.infra.logger import logger
from app.auth import (
    get_auth_state,
    handle_oauth_callback_if_present,
    render_login_overlay,
    sign_out,
    check_persistent_auth,
)

# Telemetry logger for frontend render lifecycle
frontend_telemetry = logger.getChild("FrontendTelemetry")

# Proactively spin up a warm MCP client so post-login load is instant.
get_background_client_manager().ensure_prewarmed()

# Define avatar styles as constants to ensure consistency
AVATAR_STYLES = {
    "user": "👤",        
    "assistant": "⚙️"     
}


# Constants at the top of your file
APP_CONFIG = {
    "TITLE": "Pay: RegulationsBot",
    "ICON": "🔄",
    "LAYOUT": "wide",
    "HEADER_ICON": "🔄"
}

# Session cleanup flag
if 'cleanup_registered' not in st.session_state:
    st.session_state.cleanup_registered = False

def setup_page():
    """Configure standardized page layout and metadata"""
    st.set_page_config(
        page_title=APP_CONFIG["TITLE"],
        page_icon=APP_CONFIG["ICON"],
        layout=APP_CONFIG["LAYOUT"],
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': None, 
            'Report a bug': None, 
            'About': None 
        }
    )
    
    # CSS for dark mode interface with sidebar visibility
    st.markdown("""
    <style>
    /* ===== DARK MODE THEME ===== */
    .stApp {
        background-color: #0e1117 !important;
        color: #fafafa !important;
    }
    
    /* Main content area dark mode */
    .main .block-container {
        background-color: #0e1117 !important;
        color: #fafafa !important;
    }
    
    /* Sidebar dark mode */
    [data-testid="stSidebar"] {
        background-color: #1a1d24 !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #fafafa !important;
    }

    /* Sidebar buttons: remove blue border/outline */
    [data-testid="stSidebar"] .stButton > button {
        border: 1px solid transparent !important;
        box-shadow: none !important;
        outline: none !important;
    }
    [data-testid="stSidebar"] .stButton > button:focus,
    [data-testid="stSidebar"] .stButton > button:focus-visible,
    [data-testid="stSidebar"] .stButton > button:active {
        border: 1px solid transparent !important;
        box-shadow: none !important;
        outline: none !important;
    }
    
    /* Headers dark mode */
    h1, h2, h3, h4, h5, h6 {
        color: #fafafa !important;
    }
    
    /* Text and labels dark mode */
    p, span, label, div {
        color: #fafafa !important;
    }
    
    /* Chat messages dark mode */
    .stChatMessage {
        background-color: #1a1d24 !important;
        border-color: #404552 !important;
    }
    
    /* Chat input dark mode */
    .stChatInput textarea {
        background-color: #262a33 !important;
        color: #fafafa !important;
        border-color: #404552 !important;
        border-radius: 999px !important;
        padding-left: 1.25rem !important;
        padding-right: 2.5rem !important;
    }
    
    .stChatInput textarea:focus {
        border-color: #4da6ff !important;
    }
    
    /* Dividers dark mode */
    hr {
        border-color: #404552 !important;
    }
    
    /* Hide main menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Hide the default header but keep our custom one */
    .stApp > header {visibility: hidden;}

    # /* Hide Streamlit toolbar (Deploy button) */
    # [data-testid="stToolbar"] {
    #     display: none !important;
    # }

    # /* Extra guard: hide any remaining Deploy buttons */
    # button[title="Deploy"],
    # [data-testid="stToolbar"] button,
    # button[data-testid="toolbarDeployButton"],
    # .stDeployButton {
    #     display: none !important;
    #     visibility: hidden !important;
    # }
    
    /* Ensure sidebar is always visible */
    .stSidebar {visibility: visible !important; display: block !important;}
    .stApp > div[data-testid="stSidebar"] {visibility: visible !important; display: block !important;}
    
    /* Ensure sidebar toggle button is ALWAYS visible */
    .stApp > div[data-testid="stSidebar"] > div:first-child {visibility: visible !important;}
    
    /* Keep toggle button visible and positioned correctly */
    [data-testid="collapsedControl"] {
        visibility: visible !important;
        display: flex !important;
        position: absolute !important;
        left: 0.5rem !important;
        top: 0.5rem !important;
        z-index: 999999 !important;
        background-color: #262a33 !important;
        border-radius: 4px !important;
        padding: 0.5rem !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.5) !important;
        border: 1px solid #404552 !important;
    }
    
    /* Toggle button when sidebar is expanded */
    [data-testid="stSidebarCollapsedControl"] {
        visibility: visible !important;
        display: flex !important;
        position: relative !important;
        z-index: 999999 !important;
        background-color: #262a33 !important;
    }
    
    /* Sidebar toggle button styling */
    button[kind="header"] {
        visibility: visible !important;
        display: flex !important;
        background-color: #262a33 !important;
        color: #fafafa !important;
    }
    
    button[kind="header"]:hover {
        background-color: #333a4a !important;
    }
    
    /* Fix sidebar collapse behavior - don't collapse completely */
    [data-testid="stSidebar"][aria-expanded="false"] {
        margin-left: -19rem !important;
        transition: margin-left 0.3s ease !important;
    }
    
    [data-testid="stSidebar"][aria-expanded="true"] {
        margin-left: 0 !important;
        transition: margin-left 0.3s ease !important;
    }
    
    /* Ensure the sidebar has proper width */
    [data-testid="stSidebar"] {
        width: 21rem !important;
        min-width: 21rem !important;
    }
    
    /* Keep collapsed sidebar visible enough for toggle button */
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        visibility: visible !important;
        position: relative !important;
    }
    
    /* Ensure main content takes full width when sidebar is collapsed */
    .main .block-container {
        max-width: 100% !important;
        padding-left: 5rem !important;
        padding-right: 5rem !important;
    }
    
    /* Main content area optimization */
    .stApp > div[data-testid="stAppViewContainer"] {
        width: 100% !important;
    }
    
    section.main > div {
        max-width: 100% !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
    }
    
    /* Chat input takes full width */
    .stChatInput {
        max-width: 100% !important;
        padding: 0 !important;
    }

    .stChatInput > div {
        padding: 0 !important;
        background-color: transparent !important;
    }

    .stChatInput > div div[data-baseweb="form-control-container"] {
        padding: 0 !important;
        background-color: transparent !important;
    }

    .stChatInput > div div[data-baseweb="base-input"] {
        background-color: #262a33 !important;
        border: 1px solid #404552 !important;
        border-radius: 999px !important;
        overflow: hidden;
    }

    .stChatInput > div div[data-baseweb="base-input"]::before,
    .stChatInput > div div[data-baseweb="base-input"]::after {
        display: none !important;
    }
    
    /* Chat messages take full width */
    .stChatMessage {
        max-width: 100% !important;
    }
    
    /* Custom styling for dark mode visibility */
    .stSidebar .stMarkdown {color: #fafafa;}
    .stSidebar .stTextInput > div > div > input {
        color: #fafafa !important;
        background-color: #262a33 !important;
        border-color: #404552 !important;
    }
    
    .stSidebar .stTextInput > div > div > input:focus {
        border-color: #4da6ff !important;
    }
    
    /* Responsive sidebar width */
    @media (max-width: 768px) {
        .stSidebar {
            min-width: 100% !important;
        }
        
        .stSidebar [data-testid="stSidebarNav"] {
            min-width: 100% !important;
        }
    }
    
    @media (min-width: 769px) and (max-width: 1024px) {
        .stSidebar {
            min-width: 280px !important;
            max-width: 320px !important;
        }
    }
    
    @media (min-width: 1025px) {
        .stSidebar {
            min-width: 300px !important;
            max-width: 350px !important;
        }
    }
    
    /* CRITICAL: Remove all gaps in sidebar */
    .stSidebar [data-testid="stVerticalBlock"] {
        gap: 0 !important;
    }
    
    .stSidebar > div:first-child {
        gap: 0 !important;
    }
    
    .stSidebar .element-container {
        margin-bottom: 0 !important;
        margin-top: 0 !important;
        padding-bottom: 0 !important;
        padding-top: 0 !important;
    }
    
    .stSidebar [data-testid="column"] {
        gap: 0 !important;
        padding: 0 !important;
    }
    
    /* Remove default Streamlit spacing */
    .stSidebar .stMarkdown {
        margin-bottom: 0 !important;
        margin-top: 0 !important;
    }
    
    /* Custom styling for history section - DARK MODE */
    .history-container {
        width: 100%;
        margin: 0;
        padding: 0;
    }
    .history-item {
        width: 100%;
        border: none;
        border-radius: 0;
        padding: 8px 16px;
        margin: 0;
        background-color: transparent;
        cursor: pointer;
        transition: all 0.2s ease;
        font-size: 12px;
        line-height: 1.2;
        text-align: left;
        color: #fafafa;
        border-bottom: 1px solid #404552;
        display: block;
        text-decoration: none;
        word-wrap: break-word;
        min-height: 44px;
    }
    .history-item:hover {
        background-color: #2a3040;
        color: #4da6ff;
        transform: translateX(2px);
    }
    .history-item:active {
        background-color: #333a4a;
        transform: translateX(0);
    }
    .history-item:focus {
        outline: 2px solid #4da6ff;
        outline-offset: -2px;
    }
    .history-item:last-child {
        border-bottom: none;
    }
    .history-timestamp {
        font-weight: 600;
        font-size: 11px;
        color: #c5c9d0;
        margin-bottom: 2px;
        display: block;
        word-wrap: break-word;
    }
    .history-content {
        font-size: 11px;
        color: #9aa0aa;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        word-wrap: break-word;
        line-height: 1.3;
    }
    
    /* Responsive history items */
    @media (max-width: 768px) {
        .history-item {
            padding: 10px 16px;
            font-size: 13px;
            min-height: 48px;
        }
        
        .history-timestamp {
            font-size: 12px;
        }
        
        .history-content {
            font-size: 12px;
        }
    }
    
    /* Button styling - DARK MODE */
    .stButton button {
        width: 100%;
        border: none;
        border-radius: 0;
        padding: 8px 16px;
        margin: 0;
        background-color: transparent;
        cursor: pointer;
        transition: all 0.2s ease;
        font-size: 13px;
        font-weight: 400;
        line-height: 1.2;
        text-align: left;
        color: #fafafa;
        border-bottom: 1px solid #404552;
        justify-content: flex-start;
        word-wrap: break-word;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        min-height: 44px;
        display: flex;
        align-items: center;
    }
    .stButton button:hover {
        background-color: #2a3040;
        color: #4da6ff;
        border-color: #4da6ff;
        transform: translateX(2px);
    }
    .stButton button:focus {
        box-shadow: none;
        outline: 2px solid #4da6ff;
        outline-offset: -2px;
    }
    .stButton:last-child button {
        border-bottom: none;
    }
    
    /* Selected conversation styling - DARK MODE */
    [data-testid="stButton"][data-selected="true"] button {
        background-color: #2a3040 !important;
        color: #4da6ff !important;
        border-left: 3px solid #4da6ff !important;
        font-weight: 400 !important;
        padding-left: 13px !important;
    }
    
    [data-testid="stButton"][data-selected="true"] button:hover {
        background-color: #333a4a !important;
    }
    
    /* Responsive button sizing */
    @media (max-width: 768px) {
        .stButton button {
            padding: 10px 16px;
            font-size: 14px;
            min-height: 48px;
        }
    }
    
    /* Remove gaps between buttons */
    .stButton {
        margin: 0;
        padding: 0;
        width: 100%;
        text-align: left !important;
    }
    
    /* Ensure full-width buttons align content to the left */
    .stButton[data-testid="stButton"] {
        text-align: left !important;
    }
    
    /* Target the button's internal wrapper */
    .stButton > div[data-testid="baseButton"],
    .stButton > div[data-baseweb="button"] {
        text-align: left !important;
        justify-content: flex-start !important;
    }
    
    /* Style for session info */
    .session-info {
        padding-bottom: 0.4rem;
        margin-bottom: 0 !important;
        margin-top: 0 !important;
        text-align: left !important;
    }
    
    /* Force left alignment for all sidebar content to match Session Info */
    .stSidebar [data-testid="stVerticalBlock"] {
        text-align: left !important;
    }
    
    /* Ensure button containers don't center content */
    .stSidebar [data-testid="stButton"] {
        text-align: left !important;
        display: flex !important;
        justify-content: flex-start !important;
    }
    
    .stSidebar [data-testid="stButton"] > div {
        text-align: left !important;
        justify-content: flex-start !important;
        width: 100% !important;
    }
    
    /* User info divider - DARK MODE */
    .user-info-divider {
        border-top: 1px solid #404552;
        margin-top: 0.5rem !important;
        margin-bottom: 0 !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        height: 0;
        width: 100%;
        line-height: 0;
    }
    
    .session-info .stTextInput {
        margin-bottom: 0 !important;
        margin-top: 0 !important;
    }
    
    .session-info .stTextInput > div {
        margin-bottom: 0 !important;
        margin-top: 0 !important;
    }
    
    .session-info .stTextInput input {
        font-weight: 500 !important;
        color: #262730 !important;
        word-wrap: break-word !important;
        font-size: 13px !important;
    }
    
    /* Responsive text input */
    @media (max-width: 768px) {
        .session-info .stTextInput input {
            font-size: 14px !important;
        }
    }
    
    /* Compact subheader */
    .session-info .stSubheader {
        margin-bottom: 0.4rem !important;
        margin-top: 0 !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        color: #fafafa !important;
        padding-bottom: 0 !important;
        word-wrap: break-word !important;
        line-height: 1.2 !important;
    }
    
    .stSubheader {
        margin-bottom: 0.4rem !important;
        margin-top: 0 !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        word-wrap: break-word !important;
        line-height: 1.2 !important;
        color: #fafafa !important;
    }
    
    /* Responsive subheader */
    @media (max-width: 768px) {
        .session-info .stSubheader {
            font-size: 16px !important;
        }
        
        .stSubheader {
            font-size: 16px !important;
        }
    }
    
    /* Session stats styling - DARK MODE */
    .session-stats {
        display: flex;
        justify-content: space-between;
        padding: 6px 0 4px 0;
        margin: 0;
        font-size: 12px;
        color: #9aa0aa;
    }
    
    .session-stat-item {
        display: flex;
        align-items: center;
        gap: 4px;
        margin: 0;
    }
    
    /* Style for action buttons - full width, no gaps */
    .action-buttons {
        margin: 0 !important;
        padding: 0 !important;
        width: 100%;
    }
    
    .action-buttons .stButton {
        margin: 0 !important;
        padding: 0 !important;
        width: 100%;
    }
    
    .action-buttons .stButton button {
        width: 100%;
        border: none;
        border-radius: 0;
        padding: 8px 16px !important;
        padding-left: 16px !important;
        margin: 0;
        background-color: transparent;
        cursor: pointer;
        transition: all 0.2s ease;
        font-size: 13px;
        font-weight: 500;
        line-height: 1.2;
        text-align: left !important;
        color: #fafafa;
        border-bottom: 1px solid #404552;
        justify-content: flex-start !important;
        align-items: center !important;
        word-wrap: break-word;
        white-space: normal;
        min-height: 44px;
        display: flex !important;
    }
    
    /* Ensure button content (including icons and all nested elements) is left-aligned */
    .action-buttons .stButton button > *,
    .action-buttons .stButton button span,
    .action-buttons .stButton button div,
    .action-buttons .stButton button p {
        text-align: left !important;
        justify-content: flex-start !important;
        margin-left: 0 !important;
        margin-right: auto !important;
    }
    
    /* Override any Streamlit default centering */
    .action-buttons .stButton {
        text-align: left !important;
    }
    
    .action-buttons .stButton > div {
        text-align: left !important;
        justify-content: flex-start !important;
    }
    
    .action-buttons .stButton button:hover {
        background-color: #2a3040;
        color: #4da6ff;
        border-color: #4da6ff;
        transform: translateX(2px);
    }
    
    .action-buttons .stButton button:active {
        background-color: #333a4a;
        transform: translateX(0);
    }
    
    .action-buttons .stButton button:focus {
        outline: 2px solid #4da6ff;
        outline-offset: -2px;
    }
    
    /* Chat history toggle button - left align */
    /* Target secondary buttons in sidebar (used for chat history toggle) */
    .stSidebar .stButton button[data-baseweb="button"][kind="secondary"],
    .stSidebar button[kind="secondary"],
    .stSidebar [data-testid="baseButton-secondary"],
    .stSidebar button[type="button"][kind="secondary"] {
        text-align: left !important;
        justify-content: flex-start !important;
        align-items: center !important;
        display: flex !important;
        padding-left: 16px !important;
    }
    
    /* Ensure all sidebar buttons are left-aligned by default */
    .stSidebar .stButton button {
        text-align: left !important;
        justify-content: flex-start !important;
        padding-left: 16px !important;
    }
    
    /* Target button content in sidebar */
    .stSidebar .stButton button > *,
    .stSidebar .stButton button span,
    .stSidebar .stButton button div {
        text-align: left !important;
        justify-content: flex-start !important;
        margin-left: 0 !important;
        margin-right: auto !important;
    }
    
    /* Override Streamlit's default button container centering */
    .stSidebar .stButton {
        text-align: left !important;
    }
    
    .stSidebar .stButton > div {
        text-align: left !important;
        justify-content: flex-start !important;
    }
    
    /* Sign out button special styling - DARK MODE */
    .sign-out-button button {
        color: #ff6b6b !important;
        font-weight: 500 !important;
    }
    
    .sign-out-button button:hover {
        background-color: #3d1f1f !important;
        color: #ff8585 !important;
        border-color: #ff6b6b !important;
    }
    
    /* Responsive action button sizing */
    @media (max-width: 768px) {
        .action-buttons .stButton button {
            padding: 10px 16px;
            font-size: 14px;
            min-height: 48px;
        }
    }
    
    /* Divider between action buttons and history - DARK MODE */
    .action-buttons-divider {
        border-bottom: 2px solid #404552;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        padding: 0;
    }
    
    /* Divider below sign out button - DARK MODE */
    .sign-out-divider {
        border-bottom: 1px solid #404552;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        padding: 0;
        height: 1px;
        width: 100%;
    }
    
    /* Divider above history section - DARK MODE */
    .history-divider {
        border-top: 1px solid #404552;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        height: 0;
        width: 100%;
        line-height: 0;
    }
    
    /* Loading indicator - DARK MODE */
    .loading-indicator {
        text-align: center;
        padding: 8px;
        font-size: 12px;
        color: #9aa0aa;
        font-style: italic;
    }
    
    /* Empty state styling - DARK MODE */
    .empty-state {
        text-align: center;
        padding: 20px 16px;
        color: #9aa0aa;
        font-size: 12px;
        word-wrap: break-word;
        line-height: 1.3;
    }
    
    .empty-state-icon {
        font-size: 32px;
        margin-bottom: 6px;
        opacity: 0.4;
    }
    
    /* Responsive empty state */
    @media (max-width: 768px) {
        .empty-state {
            padding: 24px 16px;
            font-size: 13px;
        }
        
        .empty-state-icon {
            font-size: 36px;
            margin-bottom: 8px;
        }
    }
    
    /* Smooth scrolling for sidebar */
    .stSidebar {
        scroll-behavior: smooth;
    }
    
    /* Touch-friendly spacing on mobile */
    @media (max-width: 768px) {
        .user-info-divider,
        .sign-out-divider {
            margin-top: 0.5rem !important;
            margin-bottom: 0.5rem !important;
        }
    }
    
    /* Ensure content doesn't overflow on small screens */
    .stSidebar * {
        max-width: 100%;
        overflow-wrap: break-word;
    }
    
    /* Dropdown styling - DARK MODE */
    .dropdown-content {
        max-height: 300px;
        overflow-y: auto;
        border: 1px solid #404552;
        border-radius: 4px;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    
    /* Dropdown header styling - DARK MODE */
    .dropdown-header {
        background-color: #262a33;
        border: 1px solid #404552;
        border-radius: 4px;
        padding: 6px 12px;
        margin: 0 !important;
        cursor: pointer;
        transition: background-color 0.2s ease;
    }
    
    .dropdown-header:hover {
        background-color: #333a4a;
    }
    
    /* Custom expander styling - DARK MODE */
    .streamlit-expanderHeader {
        background-color: transparent !important;
        border: none !important;
        border-top: none !important;
        border-bottom: none !important;
        border-left: none !important;
        border-right: none !important;
        border-radius: 0 !important;
        padding: 8px 16px !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        color: #fafafa !important;
        transition: all 0.2s ease !important;
        min-height: 44px !important;
        word-wrap: break-word !important;
        line-height: 1.2 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        text-align: left !important;
        box-shadow: none !important;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #2a3040 !important;
        color: #4da6ff !important;
        transform: translateX(2px) !important;
        border: none !important;
    }
    
    .streamlit-expanderHeader:focus {
        outline: none !important;
        border: none !important;
    }
    
    .streamlit-expanderContent {
        border: none !important;
        border-top: none !important;
        border-bottom: none !important;
        border-left: none !important;
        border-right: none !important;
        border-radius: 0 !important;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        padding: 0 !important;
        background-color: transparent !important;
    }
    
    /* Remove all borders from details element */
    .stSidebar details {
        border: none !important;
    }
    
    .stSidebar summary {
        border: none !important;
    }
    
    /* Responsive expander */
    @media (max-width: 768px) {
        .streamlit-expanderHeader {
            padding: 10px 16px !important;
            font-size: 14px !important;
            min-height: 48px !important;
        }
    }
    
    /* Remove expander container margins */
    details[open] {
        margin-bottom: 0 !important;
        margin-top: 0 !important;
    }
    
    details {
        margin-bottom: 0 !important;
        margin-top: 0 !important;
    }
    
    /* Target Streamlit's expander wrapper */
    .stExpander {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        padding: 0 !important;
        gap: 0 !important;
    }
    
    /* Compact sidebar spacing */
    .stSidebar .element-container {
        margin-bottom: 0 !important;
    }
    
    /* Reduce spacing between elements - CRITICAL */
    .stSidebar [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="element-container"] {
        gap: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Remove spacing from button containers */
    .stSidebar [data-testid="stVerticalBlock"] > div {
        gap: 0 !important;
    }
    
    /* Force zero spacing on all sidebar containers */
    .stSidebar section {
        padding-top: 0 !important;
    }
    
    .stSidebar section > div {
        gap: 0 !important;
    }
    
    /* Remove any default padding/margin from text input wrappers */
    .stSidebar .row-widget {
        margin: 0 !important;
        padding: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.header(f"{APP_CONFIG['HEADER_ICON']} {APP_CONFIG['TITLE']}")

def setup_sidebar():
    """Standard sidebar for session management"""
    with st.sidebar:
        auth = get_auth_state()
        client_bridge = st.session_state.get("client_bridge")
        bridge_ready = bool(client_bridge and client_bridge.is_ready())

        # Session Info
        st.markdown('<div class="session-info">', unsafe_allow_html=True)
        st.subheader("Session Info")
        
        # Show authenticated user email if available
        # First check if user is authenticated via OAuth and has email
        if auth.get("is_authenticated") and auth.get("email"):
            display_name = auth.get("email")
            label = "👤 Authenticated User"
            st.text_input(label, 
                        value=display_name, 
                        disabled=True,
                        label_visibility="visible")
        # Then check if client is initialized and has authenticated user info
        elif bridge_ready and client_bridge.get_authenticated_user():
            display_name = client_bridge.get_authenticated_user()
            label = "👤 Authenticated User"
            st.text_input(label, 
                        value=display_name, 
                        disabled=True,
                        label_visibility="visible")
        # Fallback to GCP username if available
        elif bridge_ready and client_bridge.get_gcp_username():
            # Extract username and format it
            username = client_bridge.get_gcp_username()
            
            # Remove last two words (split by underscore and take all except last two)
            parts = username.split('_')
            if len(parts) > 2:
                name_parts = parts[:-2]  # Take all parts except last two
            else:
                name_parts = parts  # Fallback if not enough parts
            
            # Capitalize each part and join with space
            display_name = ' '.join([part.capitalize() for part in name_parts])
            label = "👤 GCP User"
            
            st.text_input(label, 
                        value=display_name, 
                        disabled=True,
                        label_visibility="visible")
        else:
            # Show a message if no user info is available yet
            st.text_input("👤 User Status", 
                        value="Initializing...", 
                        disabled=True,
                        label_visibility="visible")
        
        # Divider below authenticated user
        st.divider()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Action buttons under Session Info - vertical column, full width, no gaps
        st.markdown('<div class="action-buttons">', unsafe_allow_html=True)
        
        # Clear button with icon
        if st.button("🗑️ Clear Chat", use_container_width=True,
                    # help="Clear current chat history without starting a new session",
                    key="clear_btn",
                    disabled=not bridge_ready):
            if bridge_ready:
                with st.spinner("Clearing chat..."):
                    client_bridge.clear_chat_history()
                st.success("Chat cleared successfully!")
                st.rerun()
        
        # New Chat button with icon
        if st.button("➕ New Chat", use_container_width=True,
                    help="",
                    key="new_chat_btn",
                    disabled=not bridge_ready):
            # Save the current conversation before clearing
            if bridge_ready:
                try:
                    # Save the conversation session before clearing
                    history = client_bridge.get_chat_history()
                    if history:
                        client_bridge.save_conversation_session()
                        
                        # Clear the history cache to force refresh
                        st.session_state.conversation_history_cache_timestamp = None
                        if 'conversation_history_cache' in st.session_state:
                            st.session_state.conversation_history_cache.clear()
                except Exception as e:
                    # Log error but continue with cleanup
                    st.warning(f"Could not save conversation: {str(e)}")
                    print(f"Error saving conversation session: {e}")
                
                # Clear only the chat history, not the entire session
                client_bridge.clear_chat_history()
                
                # Generate a new conversation ID
                import uuid
                client_bridge.set_conversation_id(str(uuid.uuid4()))
                client_bridge.set_conversation_saved(False)
                
                # Clear selected conversation
                if 'selected_conversation_id' in st.session_state:
                    st.session_state.selected_conversation_id = None
                
                # Clear folder filter selections
                if 'selected_folders' in st.session_state:
                    st.session_state.selected_folders = []
                
                # Reset folder filter expanded state for new chat
                if 'folder_filter_expanded' in st.session_state:
                    st.session_state.folder_filter_expanded = False
                
                # Force folder checkbox refresh
                if 'folder_filter_refresh' in st.session_state:
                    st.session_state.folder_filter_refresh += 1
                
                st.success("New chat started! History refreshed.")
                st.rerun()
        
        # Sign out button with icon and special styling
        if auth.get("is_authenticated"):
            st.markdown('<div class="sign-out-button">', unsafe_allow_html=True)
            if st.button("⏪ Sign Out", use_container_width=True,
                        # help="Sign out and return to login screen",
                        key="sign_out_btn"):
                # Save the current conversation before signing out
                if bridge_ready:
                    with st.spinner("Signing out..."):
                        try:
                            # Save the conversation session before signing out
                            history = client_bridge.get_chat_history()
                            if history:
                                client_bridge.save_conversation_session()
                                st.success("Conversation saved!")
                        except Exception as e:
                            # Log error but continue with cleanup
                            st.warning(f"Could not save conversation: {str(e)}")
                            print(f"Error saving conversation session: {e}")
                
                sign_out()
                st.session_state.clear()
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Divider above history section
        st.divider()
        
        # Add History section as a dropdown
        display_conversation_history()

def display_conversation_history():
    """Display the conversation history section in the sidebar as a dropdown"""
    client_bridge = st.session_state.get("client_bridge")
    client_ready = False
    user_id = None

    if client_bridge is not None:
        try:
            client_ready = client_bridge.is_ready()
            if client_ready:
                user_id = client_bridge.get_user_id()
        except RuntimeError:
            # Background client may still be initializing; treat as not ready.
            client_ready = False
            user_id = None

    # Initialize selected conversation ID if not exists
    if 'selected_conversation_id' not in st.session_state:
        st.session_state.selected_conversation_id = None
    
    # Initialize conversation session cache (for loaded conversation content)
    if 'conversation_session_cache' not in st.session_state:
        st.session_state.conversation_session_cache = {}
    
    current_conv_id = None
    if client_ready:
        try:
            current_conv_id = client_bridge.get_conversation_id()
        except RuntimeError:
            # Worker not ready after all; treat as no active conversation yet.
            client_ready = False
            user_id = None
    if current_conv_id:
        st.session_state.selected_conversation_id = current_conv_id
    
    if 'conversation_history_cache' not in st.session_state:
        st.session_state.conversation_history_cache = {}
    if 'conversation_history_cache_timestamp' not in st.session_state:
        st.session_state.conversation_history_cache_timestamp = None
    if 'chat_history_expanded' not in st.session_state:
        st.session_state.chat_history_expanded = False

    import time
    current_time = time.time()
    cache_timeout = 300  # 5 minutes

    need_refresh = (
        st.session_state.conversation_history_cache_timestamp is None
        or (current_time - st.session_state.conversation_history_cache_timestamp) > cache_timeout
        or user_id not in st.session_state.conversation_history_cache
    )

    toggle_label = "▼ Chat" if st.session_state.chat_history_expanded else "▶ Chat"
    if st.button(toggle_label, key="toggle_chat_history_btn", use_container_width=True, type="secondary"):
        st.session_state.chat_history_expanded = not st.session_state.chat_history_expanded
        st.rerun()

    if not st.session_state.chat_history_expanded:
        # st.caption("Expand Chat to load conversation history.")
        return

    if not client_ready:
        st.info("Assistant backend is still starting up. Chat history will be available once ready.")
        return

    if not user_id:
        st.info("Sign in to view your chat history.")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption("Conversation History")
    with col2:
        if st.button("🔄", key="refresh_history_btn", help="Refresh conversation history"):
            st.session_state.conversation_history_cache_timestamp = None
            st.rerun()

        sessions = []
        is_loading = False
        error_message = None

        try:
            if need_refresh:
                is_loading = True
                sessions = client_bridge.get_user_conversation_sessions(user_id)
                st.session_state.conversation_history_cache[user_id] = sessions
                st.session_state.conversation_history_cache_timestamp = current_time
            else:
                sessions = st.session_state.conversation_history_cache.get(user_id, [])
        except RuntimeError as e:
            # Background client may have gone offline; mark as not ready to show info.
            st.session_state.chat_history_expanded = False
            st.warning("Chat service is still loading, please expand again in a moment.")
            logger.warning("Conversation history fetch skipped because client not ready: %s", e)
            return
        except Exception as e:
            sessions = st.session_state.conversation_history_cache.get(user_id, [])
            error_message = str(e)

        if is_loading and not sessions:
            st.markdown('<div class="loading-indicator">⌛ Loading history...</div>', unsafe_allow_html=True)

        if error_message:
            st.warning(f"⚠️ Could not refresh history: {error_message}")

    if sessions:
        # Display sessions in reverse chronological order (newest first)
        for i, session in enumerate(sessions):
            session_start = session.get('session_start_time', '')
            if session_start:
                try:
                    from datetime import datetime
                    if isinstance(session_start, str):
                        dt = datetime.fromisoformat(session_start.replace('Z', '+00:00'))
                    else:
                        dt = session_start
                    formatted_time = dt.strftime("%b %d, %Y %H:%M")
                except Exception:
                    formatted_time = str(session_start)
            else:
                formatted_time = "Unknown time"

            summary = session.get('summary')
            if not summary:
                summary = f"Conversation {session['conversation_id'][:8]}"

            message_count = session.get('message_count', 0)
            count_text = f" • {message_count} msgs" if message_count > 0 else ""

            is_selected = st.session_state.selected_conversation_id == session['conversation_id']

            max_summary_length = 50
            truncated_summary = summary[:max_summary_length] + '...' if len(summary) > max_summary_length else summary
            button_label = f"{'✓ ' if is_selected else ''}{formatted_time}{count_text}\n{truncated_summary}"
            # hover_text = f"Load conversation from {formatted_time}: {summary}"

            button_type = "primary" if is_selected else "secondary"

            if st.button(
                button_label,
                key=f"session_btn_{i}_{session['conversation_id']}",
                # help=hover_text,
                use_container_width=True,
                type=button_type
            ):
                st.session_state.selected_conversation_id = session['conversation_id']
                load_conversation_session(session['conversation_id'])
    else:
        st.markdown('''
        <div class="empty-state">
            <div class="empty-state-icon">💭</div>
            <div>No conversation history yet</div>
            <div style="font-size: 11px; margin-top: 4px;">Start chatting to build your history</div>
        </div>
        ''', unsafe_allow_html=True)

def load_conversation_session(conversation_id: str):
    """Load a conversation session and display it in the main chat area"""
    client_bridge = st.session_state.get("client_bridge")
    if not client_bridge or not client_bridge.is_ready():
        st.warning("Assistant backend is not ready yet. Please try again shortly.")
        return

    try:
        # Check if conversation is already cached
        if conversation_id in st.session_state.conversation_session_cache:
            session_data = st.session_state.conversation_session_cache[conversation_id]
            client_bridge.clear_chat_history()

            messages = session_data['messages']
            for msg in messages:
                client_bridge.add_message(
                    role=msg['role'],
                    content=msg['content'],
                    timestamp=msg.get('timestamp', '')
                )

            client_bridge.set_conversation_id(conversation_id)
            client_bridge.set_conversation_saved(False)
            st.session_state.selected_conversation_id = conversation_id

            session_start = session_data.get('session_start_time', 'unknown time')
            try:
                from datetime import datetime
                if isinstance(session_start, str):
                    dt = datetime.fromisoformat(session_start.replace('Z', '+00:00'))
                    session_start = dt.strftime("%b %d, %Y at %H:%M")
            except Exception:
                pass

            st.success(f"✅ Loaded from cache: {session_start} ({len(messages)} messages)")
            st.rerun()
            return

        with st.spinner("..."):
            session_data = client_bridge.get_conversation_session(conversation_id)

            if session_data and 'messages' in session_data:
                st.session_state.conversation_session_cache[conversation_id] = session_data
                client_bridge.clear_chat_history()

                messages = session_data['messages']
                for msg in messages:
                    client_bridge.add_message(
                        role=msg['role'],
                        content=msg['content'],
                        timestamp=msg.get('timestamp', '')
                    )

                client_bridge.set_conversation_id(conversation_id)
                client_bridge.set_conversation_saved(False)
                st.session_state.selected_conversation_id = conversation_id

                session_start = session_data.get('session_start_time', 'unknown time')
                try:
                    from datetime import datetime
                    if isinstance(session_start, str):
                        dt = datetime.fromisoformat(session_start.replace('Z', '+00:00'))
                        session_start = dt.strftime("%b %d, %Y at %H:%M")
                except Exception:
                    pass

                st.rerun()
            else:
                st.error("❌ Could not load conversation session - no data found")
    except Exception as e:
        st.error(f"❌ Error loading conversation: {str(e)}")
        print(f"Error loading conversation session {conversation_id}: {e}")

def display_chat_history():
    """Display the conversation history"""
    client_bridge = st.session_state.get("client_bridge")
    if not client_bridge or not client_bridge.is_ready():
        # st.info("🔄 Connecting to assistant backend. You can start typing as soon as the connection is ready.")
        return
    for msg in client_bridge.get_chat_history():
        with st.chat_message(msg['role'], avatar=AVATAR_STYLES.get(msg['role'], "⚙️")):
            st.markdown(msg['content'], unsafe_allow_html=True)

def render_folder_filter():
    """Render folder filter UI as a collapsible dropdown with multi-select support"""
    client_bridge = st.session_state.get("client_bridge")
    bridge_ready = client_bridge and client_bridge.is_ready()
    
    # Check for cached folder list first (available immediately)
    cache_key = 'available_folders_cache'
    bridge_email_key = 'client_bridge_email'
    current_bridge_email = st.session_state.get(bridge_email_key)
    folders = None
    
    # Try to get folders from cache immediately (no bridge call needed)
    if cache_key in st.session_state and st.session_state.get('available_folders_cache_bridge_email') == current_bridge_email:
        folders = st.session_state[cache_key]
    
    # If not cached and bridge is ready, try to fetch (but don't block UI)
    if folders is None and bridge_ready:
        try:
            folders = client_bridge.get_available_folders()
            # Cache the result
            st.session_state[cache_key] = folders
            st.session_state['available_folders_cache_bridge_email'] = current_bridge_email
        except Exception:
            # Silently fail - will show loading state
            folders = None
    
    # Always render the folder filter UI structure immediately
    # Show loading state if folders aren't available yet, otherwise show full UI
    try:
        if folders is None:
            # Show placeholder expander when folders are loading
            # UI appears immediately, will populate on next render when cache is available
            with st.expander("📁 **Filter:** Loading...", expanded=False):
                st.info("🔄 Loading folders...")
            # Show tip immediately even when loading
            st.caption("💡 Tip: Select specific folders to get more accurate and targeted responses from relevant documents")
            return
        
        if not folders:
            # Still show tip even if no folders available
            st.caption("💡 Tip: Select specific folders to get more accurate and targeted responses from relevant documents")
            return  # Don't show filter if no folders available
        
        # Get current folder filters from backend (source of truth)
        current_filters = client_bridge.get_current_folder_filters()
        
        # Sync frontend state with backend state (ensure they match)
        if 'selected_folders' not in st.session_state:
            st.session_state.selected_folders = current_filters.copy() if current_filters else []
        else:
            # Always sync with backend to reflect "New Chat" clears
            st.session_state.selected_folders = current_filters.copy() if current_filters else []
        
        # Track folder filter expanded state (persistent across reruns)
        # User has full control - state only changes when user interacts or we explicitly set it
        if 'folder_filter_expanded' not in st.session_state:
            st.session_state.folder_filter_expanded = False
        
        # Track a refresh counter to force checkbox re-render when needed
        if 'folder_filter_refresh' not in st.session_state:
            st.session_state.folder_filter_refresh = 0
        
        # Track if we're setting expanded state programmatically (to avoid overriding user's manual close)
        if 'folder_filter_setting_state' not in st.session_state:
            st.session_state.folder_filter_setting_state = False
        
        # Show current selection status
        if not current_filters:
            current_display = "All Folders"
        elif len(current_filters) == 1:
            current_display = current_filters[0]
        else:
            current_display = f"{len(current_filters)} folders selected"
        
        # Collapsible dropdown for folder selection
        # User has full control - if they open it, it stays open; if they close it, it stays closed
        # We only set it to True when user interacts with buttons/checkboxes inside
        with st.expander(f"📁 **Filter:** {current_display}", expanded=st.session_state.folder_filter_expanded):
            # Don't automatically set expanded to True here - let user control it
            # Only set to True when user actually interacts with controls (see button/checkbox handlers below)
            
            # Compact control buttons
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                if st.button("Clear", key="clear_all_folders", use_container_width=True, type="secondary"):
                    st.session_state.selected_folders = []
                    client_bridge.clear_folder_filters()
                    st.session_state.folder_filter_expanded = True  # Keep expanded
                    st.session_state.folder_filter_refresh += 1  # Force checkbox refresh
                    st.rerun()
            with col2:
                if st.button("All", key="select_all_folders", use_container_width=True, type="secondary"):
                    st.session_state.selected_folders = folders.copy()
                    client_bridge.set_folder_filters(folders)
                    st.session_state.folder_filter_expanded = True  # Keep expanded
                    st.session_state.folder_filter_refresh += 1  # Force checkbox refresh
                    st.rerun()
            
            # Create columns for folder checkboxes (show 5 per row for compact layout)
            num_cols = 5
            
            # Display folders in rows with checkboxes
            for i in range(0, len(folders), num_cols):
                cols = st.columns(num_cols)
                for j, folder in enumerate(folders[i:i+num_cols]):
                    with cols[j]:
                        # Check if this folder is currently selected
                        is_selected = folder in current_filters
                        
                        # Checkbox for each folder (include refresh counter in key to force re-render)
                        checkbox_value = st.checkbox(
                            folder,
                            value=is_selected,
                            key=f"folder_checkbox_{folder}_{st.session_state.folder_filter_refresh}"
                        )
                        
                        # Update selection if changed
                        if checkbox_value and not is_selected:
                            # Add folder to filters
                            client_bridge.add_folder_filter(folder)
                            if folder not in st.session_state.selected_folders:
                                st.session_state.selected_folders.append(folder)
                            st.session_state.folder_filter_expanded = True  # Keep expanded after interaction
                        elif not checkbox_value and is_selected:
                            # Remove folder from filters
                            client_bridge.remove_folder_filter(folder)
                            if folder in st.session_state.selected_folders:
                                st.session_state.selected_folders.remove(folder)
                            st.session_state.folder_filter_expanded = True  # Keep expanded after interaction
        
    except Exception as e:
        # Silently fail - don't break the UI if folder filter fails
        pass
    
    # Always show tip right under the folder filter (renders at same time as other components)
    st.caption("💡 Tip: Select specific folders to get more accurate and targeted responses from relevant documents")

def handle_user_input(bridge_ready: bool):
    """Process user input and display responses with graceful queuing while backend warms up."""
    waiting_placeholder = "Assistant is connecting… type now and we'll send it once ready."
    prompt_placeholder = "Ask about pay regulations"
    prompt = st.chat_input(
        waiting_placeholder if not bridge_ready else prompt_placeholder,
        key="chat_input",
        disabled=False,
    )
    pending_prompt = st.session_state.get("pending_prompt")

    if not bridge_ready:
        if prompt:
            st.session_state.pending_prompt = prompt
            st.info("⚙️ Assistant is still starting up. We'll send your question automatically when it's ready.")
        elif pending_prompt:
            st.info(f"⚙️ Assistant connecting… queued question: “{pending_prompt}”")
        return

    # Backend ready; send any queued prompt first if no new input this render
    if not prompt and pending_prompt:
        prompt = pending_prompt
        st.session_state.pop("pending_prompt", None)

    if not prompt:
        return

    # Display user message immediately with person avatar
    with st.chat_message("user", avatar=AVATAR_STYLES["user"]):
        st.markdown(prompt)

    client_bridge = st.session_state.get("client_bridge")
    if not client_bridge or not client_bridge.is_ready():
        st.warning("Assistant backend became unavailable. Please wait a moment.")
        st.session_state.pending_prompt = prompt
        time.sleep(0.2)
        st.experimental_rerun()

    # Check if this is a reference request
    is_reference_request = False
    reference_patterns = [
        "references", "show references", "give me references", 
        "what are the references", "show me the references",
        "show me the references for this session"
    ]
    
    if prompt.lower().strip() in reference_patterns:
        is_reference_request = True

    assistant_message = st.chat_message("assistant", avatar=AVATAR_STYLES["assistant"])
    response_placeholder = assistant_message.empty()
    response_placeholder.markdown("🔎 Searching regulations...")

    if is_reference_request:
        response = client_bridge.format_references()
        from datetime import datetime
        client_bridge.add_message("assistant", response, datetime.utcnow().isoformat() + 'Z')
    else:
        response = client_bridge.process_query(prompt)

    response_placeholder.markdown(response, unsafe_allow_html=True)

def show_footer():
    """Display footer information"""
    st.markdown("---")
    # st.caption("💡 Tip: Start a fresh session by clicking 'New Chat' button")

def render_connection_wait_state(bridge):
    """Render a lightweight placeholder while the background client connects."""
    elapsed = time.time() - st.session_state.get('client_connect_started', time.time())
    status_box = st.empty()
    # status_box.info(f"Connecting to assistant… {elapsed:.1f}s elapsed")
    # Yield control briefly before re-running to poll readiness
    time.sleep(0.2)

    # Streamlit renamed experimental_rerun -> rerun in newer releases.
    rerun = getattr(st, "rerun", None)
    if rerun is None:
        rerun = getattr(st, "experimental_rerun", None)
    if rerun:
        rerun()
    else:
        raise RuntimeError("Streamlit rerun function unavailable")

def render_connection_ready_banner():
    """Display a subtle ready indicator once the backend session is established."""
    if 'connection_ready_elapsed' not in st.session_state:
        started = st.session_state.get('client_connect_started')
        if started:
            elapsed = time.time() - started
            st.session_state.connection_ready_elapsed = elapsed
            logger.getChild("FrontendInit").info(
                "Assistant connection established in %.0fms", elapsed * 1000
            )
    elapsed = st.session_state.get('connection_ready_elapsed')
    # if elapsed is not None:
    #     # st.caption(f"✅ Assistant ready in {elapsed:.1f}s")

def ensure_client_bridge():
    """Ensure a background MCP client exists for the authenticated user."""
    auth = get_auth_state()
    user_email = auth.get('email') if auth.get('is_authenticated') else None
    if 'client_bridge' in st.session_state:
        current_email = st.session_state.get('client_bridge_email')
        if current_email == user_email:
            return st.session_state.client_bridge

    frontend_logger = logger.getChild("FrontendInit")
    manager = get_background_client_manager()
    bridge = manager.get_or_create(user_email)
    st.session_state.client_bridge = bridge
    previous_email = st.session_state.get('client_bridge_email')
    st.session_state.client_bridge_email = user_email
    
    # Reset folder fetch flag if bridge email changed (new user)
    if previous_email != user_email:
        st.session_state.pop('folders_fetch_triggered', None)
        st.session_state.pop('available_folders_cache', None)
        st.session_state.pop('available_folders_cache_bridge_email', None)
    
    if 'client_connect_started' not in st.session_state:
        st.session_state.client_connect_started = time.time()
    status = "ready" if bridge.is_ready() else "pending"
    frontend_logger.info("Background client status=%s for user=%s", status, user_email or "anonymous")
    return bridge

def cleanup_client():
    """Cleanup function to properly close resources and save conversation sessions"""
    if 'client_bridge' in st.session_state:
        bridge = st.session_state.client_bridge
        try:
            if bridge.is_ready():
                try:
                    history = bridge.get_chat_history()
                    if history:
                        bridge.save_conversation_session()
                except Exception as e:
                    print(f"Error saving conversation session during cleanup: {e}")
                try:
                    bridge.close()
                except Exception as e:
                    print(f"Error during client cleanup: {e}")
        finally:
            st.session_state.pop('client_bridge', None)
            st.session_state.pop('client_bridge_email', None)
            st.session_state.pop('client_connect_started', None)

def run_frontend():
    """Main function to run the Streamlit frontend"""
    render_start = time.perf_counter()
    # Track render start for later elapsed calculations
    st.session_state["frontend_render_start"] = render_start
    render_token = f"{time.time():.6f}"
    st.session_state["frontend_render_token"] = render_token
    frontend_telemetry.info("Frontend render invoked; token=%s", render_token)

    setup_page()
    # Handle OAuth callback if present
    oauth_start = time.perf_counter()
    oauth_handled = handle_oauth_callback_if_present()
    if oauth_handled:
        frontend_telemetry.info(
            "OAuth callback processed in %.3fs",
            time.perf_counter() - oauth_start,
        )

    # Check for persistent authentication first
    auth = get_auth_state()
    if not auth.get("is_authenticated"):
        # Try to restore from persistent authentication
        if check_persistent_auth():
            frontend_telemetry.info(
                "Restored authentication from persistent storage (elapsed=%.3fs)",
                time.perf_counter() - render_start,
            )
            # Rerun to show the authenticated UI
            st.rerun()
            return
    
    auth = get_auth_state()

    if not auth.get("is_authenticated"):
        # Kick off backend prewarm so the assistant is ready immediately post-login
        get_background_client_manager().ensure_prewarmed()
        frontend_telemetry.debug(
            "User not authenticated; rendering login overlay (elapsed=%.3fs)",
            time.perf_counter() - render_start,
        )
        render_login_overlay()
        return

    # Authenticated user flow
    bridge = ensure_client_bridge()
    bridge_ready = bridge.is_ready()

    if not bridge_ready:
        frontend_telemetry.info(
            "Client bridge pending; ready=%s; elapsed=%.3fs",
            bridge_ready,
            time.perf_counter() - render_start,
        )

    client_error = bridge.get_error()
    if client_error:
        frontend_telemetry.error(
            "Client bridge error after %.3fs: %s",
            time.perf_counter() - render_start,
            client_error,
        )
        st.error(f"Unable to connect to assistant backend: {client_error}")
        st.stop()
    if bridge_ready:
        ready_elapsed = time.perf_counter() - render_start
        st.session_state["connection_ready_elapsed"] = ready_elapsed
        frontend_telemetry.info(
            "Chatbot ready; elapsed=%.3fs; conversation_id=%s",
            ready_elapsed,
            bridge.get_conversation_id(),
        )
        render_connection_ready_banner()
        
        # Pre-fetch folder list immediately when bridge becomes ready
        # This ensures it's cached and triggers UI update
        cache_key = 'available_folders_cache'
        bridge_email_key = 'client_bridge_email'
        current_bridge_email = st.session_state.get(bridge_email_key)
        folders_fetch_key = 'folders_fetch_triggered'
        
        # Only fetch if not already cached or if bridge email changed
        if not (cache_key in st.session_state and st.session_state.get('available_folders_cache_bridge_email') == current_bridge_email):
            # Check if we've already triggered a fetch (to avoid multiple reruns)
            if not st.session_state.get(folders_fetch_key, False):
                try:
                    folders = bridge.get_available_folders()
                    st.session_state[cache_key] = folders
                    st.session_state['available_folders_cache_bridge_email'] = current_bridge_email
                    st.session_state[folders_fetch_key] = True
                    # Trigger rerun to update UI immediately with folders
                    st.rerun()
                except Exception:
                    # Silently fail - render_folder_filter() will handle it
                    pass
        else:
            # Cache exists, mark as fetched
            st.session_state[folders_fetch_key] = True
    
        # Handle session cleanup when the page is reloaded or session ends
        # Track the current conversation ID to detect when it changes
        current_conversation_id = bridge.get_conversation_id()
        
        if 'last_conversation_id' not in st.session_state:
            # First run of this session
            st.session_state.last_conversation_id = current_conversation_id
        else:
            # Check if we're continuing the same conversation or starting a new one
            if st.session_state.last_conversation_id != current_conversation_id:
                # The conversation ID has changed, which means we started a new conversation
                # Save the previous one if it has messages
                try:
                    # We need to get the previous client to save its conversation
                    # This is a bit tricky in Streamlit, so we'll just log this situation
                    print(f"Conversation changed from {st.session_state.last_conversation_id} to {current_conversation_id}")
                except Exception as e:
                    print(f"Error handling conversation change: {e}")
            
            # Update the last conversation ID
            st.session_state.last_conversation_id = current_conversation_id
        
        # Handle page reload - save conversation if there are messages
        if 'page_reload_check' not in st.session_state:
            st.session_state.page_reload_check = True
        else:
            # This is not the first run, so we might be dealing with a page reload
            # Check if we have messages that need to be saved
            history = bridge.get_chat_history()
            if history and not bridge.is_conversation_saved():
                # Try to save the conversation session
                try:
                    bridge.save_conversation_session()
                except Exception as e:
                    print(f"Error saving conversation session on page reload: {e}")
    
    # Render folder filter right under the title (fixed position)
    render_folder_filter()
    
    setup_sidebar()
    display_chat_history()
    handle_user_input(bridge_ready=bridge_ready)
    show_footer()

    if not bridge_ready:
        render_connection_wait_state(bridge)

# Register cleanup function to run when the script exits
import atexit
import sys

def cleanup():
    """Cleanup function to properly close resources"""
    cleanup_client()

# Register the cleanup function
atexit.register(cleanup)

if __name__ == "__main__":
    run_frontend()