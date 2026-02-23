#!/bin/bash
# Pay Regulations Chatbot Clean Script
# Clears generated files while preserving source code and configurations

function log() {
    echo -e "$(date +"%Y-%m-%d %H:%M:%S") INFO $@"
}

function warn() {
    echo -e "$(date +"%Y-%m-%d %H:%M:%S") WARNING $@"
}

function error() {
    echo -e "$(date +"%Y-%m-%d %H:%M:%S") ERROR $@"
}

# Get the project root (directory containing this script)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_DIR"

log "Cleaning up generated files in $SCRIPT_DIR..."

# Remove config/.env file if it exists
if [ -f "config/.env" ]; then
    log "Removing config/.env"
    rm -f "config/.env"
fi

# Remove backup .env file if it exists
if [ -f "config/.env.backup" ]; then
    log "Removing config/.env.backup"
    rm -f "config/.env.backup"
fi

# Remove virtual environment
if [ -d ".venv" ]; then
    log "Removing .venv directory"
    rm -rf ".venv"
fi

# Remove python directory
if [ -d "python" ]; then
    log "Removing python directory"
    rm -rf "python"
fi

# Remove uv-cache directory
if [ -d "uv-cache" ]; then
    log "Removing uv-cache directory"
    rm -rf "uv-cache"
fi

# Remove logs directory
if [ -d "logs" ]; then
    log "Removing logs directory"
    rm -rf "logs"
fi

# Remove __pycache__ directories
log "Removing __pycache__ directories"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Remove Python compiled files
log "Removing Python compiled files"
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
find . -type f -name "*.pyd" -delete 2>/dev/null || true

# Remove other cache and temporary files
log "Removing cache and temporary files"
rm -rf ".mypy_cache" 2>/dev/null || true
rm -rf ".pytest_cache" 2>/dev/null || true
rm -rf ".coverage" 2>/dev/null || true
rm -rf "htmlcov" 2>/dev/null || true
rm -f ".api_pid" 2>/dev/null || true
rm -f ".streamlit_pid" 2>/dev/null || true
rm -f "vector_data.csv" 2>/dev/null || true

log "Cleanup complete!"
log "Note: If a virtual environment was active in your current shell, you may need to restart your shell or manually deactivate it."
