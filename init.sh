#!/bin/bash
DEPLOYMENT=${1:-"dev"}
ENV_ONLY="${2:-}"

VENV_PYTHON_VERSION=3.11

# Set up logging directory
LOG_DIR="$PWD/logs"
mkdir -p "$LOG_DIR"

function log() {
    local timestamp=$(date +"%Y-%m-%d T%H:%M:%S%z")
    echo -e "$timestamp INFO $@" | tee -a "$LOG_DIR/init_$(date +%F).log"
}

function warn() {
    local timestamp=$(date +"%Y-%m-%d T%H:%M:%S%z")
    echo -e "$timestamp WARNING $@" | tee -a "$LOG_DIR/init_$(date +%F).log"
}

function error() {
    local timestamp=$(date +"%Y-%m-%d T%H:%M:%S%z")
    echo -e "$timestamp ERROR $@" | tee -a "$LOG_DIR/init_$(date +%F).log"
    exit 1
}

EXECUTION_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$EXECUTION_DIRECTORY"

log "Starting initialization for $DEPLOYMENT environment"

# Read project ID
if [ -f ".gcloud/keyfile.json" ]; then
    ACTUAL_PROJECT_ID=$(grep -o '"project_id": *"[^"]*"' .gcloud/keyfile.json | cut -d'"' -f4)
    ACTUAL_PROJECT_ID=${ACTUAL_PROJECT_ID:-fintech-dep-staging}
else
    ACTUAL_PROJECT_ID="fintech-dep-staging"
fi

# Deployment config
case $DEPLOYMENT in
    dev)
        ROOT_PROJECT_DIR=$EXECUTION_DIRECTORY
        DOT_ENV_FILE=$EXECUTION_DIRECTORY/config/.env
        DATA_BUCKET_NAME="dev-data-bucket-regulon"
        EMBEDDING_BUCKET_NAME="dev-embedding-bucket-regulon"
        FAISS_BUCKET_NAME="dev-faiss-bucket-regulon"
        INDEX_NAME="regulon_index_dev"
        ENDPOINT_NAME="regulon_index_dev_endpoint"
        OAUTH_REDIRECT_URI="http://localhost:8501"
        ;;
    prod)
        ROOT_PROJECT_DIR=$EXECUTION_DIRECTORY
        DOT_ENV_FILE=$EXECUTION_DIRECTORY/config/.env
        DATA_BUCKET_NAME="prod-data-bucket-regulon"
        EMBEDDING_BUCKET_NAME="prod-embedding-bucket-regulon"
        FAISS_BUCKET_NAME="prod-faiss-bucket-regulon"
        INDEX_NAME="regulon_index_prod"
        ENDPOINT_NAME="regulon_index_prod_endpoint"
        OAUTH_REDIRECT_URI="http://regulator-bot-sandbox.pathaopay.com:8501"
        ;;
    *)
        error "Invalid deployment $DEPLOYMENT. Use dev or prod"
        ;;
esac

UV_CACHE_DIR="$ROOT_PROJECT_DIR/uv-cache"

# ------------------ ENV FILE ------------------
function create_dot_env() {
    [ -f "$DOT_ENV_FILE" ] && warn "Overwriting existing $DOT_ENV_FILE"

    cat > "$DOT_ENV_FILE" <<EOF
# Auto-generated .env for $DEPLOYMENT
GOOGLE_APPLICATION_CREDENTIALS=.gcloud/keyfile.json
OAUTH_CREDENTIALS_FILE=.gcloud/oauth_credentials.json
ALLOWED_DOMAIN=pathao.com
GCP_PROJECT_ID=$ACTUAL_PROJECT_ID
GCP_LOCATION=us-central1

DATA_BUCKET_NAME=$DATA_BUCKET_NAME
EMBEDDING_BUCKET_NAME=$EMBEDDING_BUCKET_NAME
FAISS_BUCKET_NAME=$FAISS_BUCKET_NAME

INDEX_NAME=$INDEX_NAME
ENDPOINT_NAME=$ENDPOINT_NAME
OAUTH_REDIRECT_URI=$OAUTH_REDIRECT_URI
MODEL_NAME=gemini-2.5-flash

DATA_DIR=data
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
VECTOR_STORE_PATH=vector_store
CSV_PATH=vector_data.csv

HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
DEPLOYMENT=$DEPLOYMENT
ROOT_PROJECT_DIR=$ROOT_PROJECT_DIR
EOF

    log "$DOT_ENV_FILE file created"
}

# ------------------ PYTHON & VENV ------------------
function create_python_venv() {
    log "Checking uv installation"
    command -v uv >/dev/null 2>&1 || error "uv is not installed"

    log "Checking system Python 3.11"
    command -v python3.11 >/dev/null 2>&1 || \
        error "Python 3.11 not found. Install with: sudo apt install python3.11 python3.11-venv python3.11-dev"

    PYTHON_BIN=$(command -v python3.11)
    log "Using system Python: $($PYTHON_BIN --version)"

    if [ ! -d ".venv" ]; then
        log "Creating virtual environment"
        uv venv .venv --python "$PYTHON_BIN" --cache-dir "$UV_CACHE_DIR" || \
            error "Failed to create virtual environment"
    else
        log ".venv already exists"
    fi

    log "Activating virtual environment"
    source .venv/bin/activate

    if [ -f "config/requirements.txt" ]; then
        log "Installing requirements"
        uv pip install -r config/requirements.txt || error "Dependency installation failed"
    else
        error "config/requirements.txt not found"
    fi

    log "Python in venv: $(which python)"
}

# ------------------ RUN ------------------
create_dot_env
export $(grep -v '^#' "$DOT_ENV_FILE" | grep -v '^$' | xargs)
log "Environment variables exported"

if [ "$ENV_ONLY" != "--env-only" ]; then
    create_python_venv
    log "Setup completed successfully"
    log "Activate venv: source .venv/bin/activate"
else
    log "Env file updated (--env-only)"
fi
log "Load env: export \$(grep -v '^#' $DOT_ENV_FILE | grep -v '^$' | xargs)"
