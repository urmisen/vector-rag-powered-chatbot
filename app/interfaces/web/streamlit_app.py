import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import streamlit as st
import asyncio
import atexit
import os
from dotenv import load_dotenv

# Load environment variables from config/.env (project root)
_env_path = Path(__file__).resolve().parent.parent.parent.parent / "config" / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

from app.interfaces.web.frontend import run_frontend, cleanup
from app.infra.logger import logger

app_logger = logger.getChild("StreamlitApp")
app_logger.info("Starting RegulationsBot Streamlit application")

# Set event loop policy for Unix systems
if sys.platform != "win32":
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

if __name__ == "__main__":
    # Register cleanup function to run on exit
    atexit.register(cleanup)
    
    # Run the frontend
    run_frontend()
    app_logger.info("Frontend execution completed")