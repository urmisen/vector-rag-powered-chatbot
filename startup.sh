#!/bin/bash

# Pay Regulations Chatbot Startup Script
# Starts the application. Dev and prod run the same steps; only env vars differ (see init.sh).
#   - Ensures config/.env has correct deployment vars (runs init.sh)
#   - Activates venv, loads env, syncs GCS data, warms up RAG, starts API + Streamlit
#
# Usage:
#   ./startup.sh          # production (default)
#   ./startup.sh prod     # production
#   ./startup.sh dev      # development (same steps, dev env vars from init.sh)

set -e  # Exit on any error

# Mode: dev or prod (default prod)
MODE="${1:-prod}"
if [ "$MODE" != "dev" ] && [ "$MODE" != "prod" ]; then
    echo "Usage: $0 [dev|prod]"
    exit 1
fi

# Configuration
API_PORT=8000
STREAMLIT_PORT=8501

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output with timestamp
print_status() {
    echo -e "$(date +"%Y-%m-%d T%H:%M:%S%z") ${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "$(date +"%Y-%m-%d T%H:%M:%S%z") ${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "$(date +"%Y-%m-%d T%H:%M:%S%z") ${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "$(date +"%Y-%m-%d T%H:%M:%S%z") ${RED}[ERROR]${NC} $1"
    exit 1
}

# Function to check if virtual environment exists
check_venv() {
    if [ ! -d ".venv" ]; then
        print_error "Virtual environment not found. Please run init.sh first to set up the environment."
        exit 1
    fi
    
    print_status "Activating virtual environment"
    source .venv/bin/activate
    
    print_success "Virtual environment activated"
}

# Function to check environment variables
check_env() {
    print_status "Checking environment variables..."
    
    # Check for config/.env file
    if [ ! -f "config/.env" ]; then
        print_warning ".env file not found."
        print_warning "Please run init.sh to generate the environment file or create config/.env manually."
        exit 1
    fi
    
    # Load environment variables from config/.env file
    export $(grep -v '^#' config/.env | xargs)
    
    # Check for required files (prod may rely on GOOGLE_APPLICATION_CREDENTIALS)
    if [ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ] && [ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
        print_warning "GOOGLE_APPLICATION_CREDENTIALS file not found: $GOOGLE_APPLICATION_CREDENTIALS"
        print_warning "Please ensure the credentials file exists or update the path in config/.env"
    fi
    
    print_success "Environment variables loaded"
}

# Function to get Python executable from virtual environment
get_python_executable() {
    if [ -f ".venv/bin/python" ]; then
        echo ".venv/bin/python"
    elif [ -f ".venv/Scripts/python.exe" ]; then
        echo ".venv/Scripts/python.exe"
    else
        print_error "Python executable not found in virtual environment"
        exit 1
    fi
}

# Function to pre-warm backend services
run_warmup() {
    print_status "Pre-warming backend services (RAG)... this may take up to 2 minutes on a cold start."

    PYTHON_CMD=$(get_python_executable)

    # Warm up RAG resources and bootstrap the background client
    if PYTHONPATH=. $PYTHON_CMD scripts/warmup_services.py --repeat 1 --bootstrap-client --client-timeout 150; then
        print_success "Warmup cycle completed."
    else
        print_warning "Warmup script encountered an issue. Continuing startup without pre-warmed services."
    fi
}

sync_data() {
    print_status "Syncing latest sentence and FAISS data from GCS..."
    
    PYTHON_CMD=$(get_python_executable)
    if PYTHONPATH=. $PYTHON_CMD scripts/sync_index_data.py --force; then
        print_success "Data synchronization completed."
    else
        print_error "Failed to synchronize data from GCS. Aborting startup."
    fi
}

# Function to start the application
start_application() {
    print_status "Starting the application..."
    
    # Check if API server is already running
    if lsof -Pi :$API_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        print_warning "Port $API_PORT is already in use. Stopping existing process..."
        pkill -f "uvicorn.*app.api" || true
        sleep 2
    fi
    
    # Get Python executable from virtual environment
    PYTHON_CMD=$(get_python_executable)
    
    # Start the FastAPI server with proper environment
    print_status "Starting FastAPI server on http://localhost:$API_PORT"
    source .venv/bin/activate && PYTHONPATH=. $PYTHON_CMD -m uvicorn app.interfaces.api.api:app --host 0.0.0.0 --port $API_PORT --reload &
    API_PID=$!
    
    print_success "API started successfully!"
    print_status "API server PID: $API_PID"
    print_status "API Documentation: http://localhost:$API_PORT/docs"
    print_status "Health Check: http://localhost:$API_PORT/health"

    # Save PID to file for later cleanup
    echo $API_PID > .api_pid
}

# Function to start the Streamlit frontend
start_streamlit() {
    print_status "Starting the Streamlit frontend..."

    # Check if Streamlit is already running
    if lsof -Pi :$STREAMLIT_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        print_warning "Port $STREAMLIT_PORT is already in use. Stopping existing Streamlit process..."
        pkill -f "streamlit run app/frontend.py" || true
        sleep 2
    fi

    # Get Python executable from virtual environment
    PYTHON_CMD=$(get_python_executable)

    # Start Streamlit with proper environment
    print_status "Starting Streamlit app on http://localhost:$STREAMLIT_PORT"
    source .venv/bin/activate && PYTHONPATH=. $PYTHON_CMD -m streamlit run app/interfaces/web/frontend.py --server.address 0.0.0.0 --server.port $STREAMLIT_PORT &
    STREAMLIT_PID=$!

    print_success "Streamlit started successfully!"
    print_status "Streamlit PID: $STREAMLIT_PID"

    # Save PID to file for later cleanup
    echo $STREAMLIT_PID > .streamlit_pid
}

# Function to cleanup on exit
cleanup() {
    print_status "Stopping application..."
    
    if [ -f ".api_pid" ]; then
        API_PID=$(cat .api_pid)
        if kill -0 $API_PID 2>/dev/null; then
            kill $API_PID
            print_success "Application stopped"
        fi
        rm -f .api_pid
    fi
    
    if [ -f ".streamlit_pid" ]; then
        STREAMLIT_PID=$(cat .streamlit_pid)
        if kill -0 $STREAMLIT_PID 2>/dev/null; then
            kill $STREAMLIT_PID
            print_success "Streamlit stopped"
        fi
        rm -f .streamlit_pid
    fi
    
    exit 0
}

# Main execution
main() {
    print_status "Starting Pay Regulations Chatbot ($MODE mode)..."
    
    # Check if we're in the right directory
    if [ ! -f "config/requirements.txt" ] && [ ! -f "pyproject.toml" ]; then
        print_error "Not in project root directory. Please run this script from the project root."
        exit 1
    fi
    
    # Check virtual environment
    check_venv
    
    # Ensure config/.env has correct deployment vars (env-only, no venv/deps)
    print_status "Ensuring config/.env for $MODE deployment..."
    ./init.sh "$MODE" --env-only
    
    # Check environment variables
    check_env
    
    # Sync index data from GCS
    sync_data
    
    # Pre-warm heavy services before serving traffic
    run_warmup
    
    # Start API and Frontend
    start_application
    start_streamlit

    # Wait for user to stop
    print_status "Press Ctrl+C to stop the services"
    trap 'cleanup' INT

    # Keep script running; wait until either process exits
    if [ -f .api_pid ]; then API_PID=$(cat .api_pid); fi
    if [ -f .streamlit_pid ]; then STREAMLIT_PID=$(cat .streamlit_pid); fi
    
    # Wait for either process to exit (compatible with older bash versions)
    while kill -0 $API_PID 2>/dev/null && kill -0 $STREAMLIT_PID 2>/dev/null; do
        sleep 1
    done
    
    # Clean up on exit
    cleanup
}

# Run main function
main "$@"