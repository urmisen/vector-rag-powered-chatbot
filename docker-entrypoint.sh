#!/bin/bash
# Docker entrypoint script
# Handles init.sh and startup.sh, making data sync optional

echo "[Docker] Starting Regulon container..."

# Step 1: Run init.sh if .env doesn't exist
if [ ! -f "config/.env" ]; then
    echo "[Docker] Running init.sh prod..."
    ./init.sh prod || {
        echo "[Docker] WARNING: init.sh had issues, checking if .env was created..."
        if [ ! -f "config/.env" ]; then
            echo "[Docker] ERROR: config/.env not found after init.sh"
            exit 1
        fi
    }
fi

# Step 2: Load environment variables
if [ -f "config/.env" ]; then
    export $(grep -v '^#' config/.env | grep -v '^$' | xargs)
    echo "[Docker] Environment variables loaded"
else
    echo "[Docker] ERROR: config/.env not found"
    exit 1
fi

# Step 3: Try to run startup.sh
# Since startup.sh uses 'set -e', we need to run it in a way that doesn't exit the container
echo "[Docker] Running startup.sh..."
set +e  # Disable exit on error temporarily
./startup.sh
STARTUP_EXIT=$?
set -e  # Re-enable exit on error

if [ $STARTUP_EXIT -eq 0 ]; then
    # startup.sh succeeded
    echo "[Docker] Startup completed successfully"
    # startup.sh should keep running, but if it exits, we'll be here
    exit 0
else
    # startup.sh failed - likely due to data sync
    echo "[Docker] startup.sh exited with code $STARTUP_EXIT"
    echo "[Docker] This is likely due to data sync failure (network/GCS issue)"
    echo "[Docker] Starting services manually without data sync..."
    
    # Get Python executable
    if [ -f ".venv/bin/python" ]; then
        PYTHON_CMD=".venv/bin/python"
    else
        PYTHON_CMD="python3"
    fi
    
    # Activate venv if it exists
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    fi
    
    # Start FastAPI in background
    echo "[Docker] Starting FastAPI on port 8000..."
    PYTHONPATH=. $PYTHON_CMD -m uvicorn app.interfaces.api.api:app --host 0.0.0.0 --port 8000 &
    API_PID=$!
    echo $API_PID > .api_pid
    echo "[Docker] FastAPI started with PID $API_PID"
    
    # Start Streamlit in foreground (so Docker sees it as main process)
    echo "[Docker] Starting Streamlit on port 8501..."
    exec PYTHONPATH=. $PYTHON_CMD -m streamlit run app/interfaces/web/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
fi
