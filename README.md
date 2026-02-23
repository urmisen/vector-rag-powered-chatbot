# Vector RAG Bot

RAG chatbot for banking and payment regulations. Uses vector similarity search and an LLM to answer questions from regulatory documents with inline citations.

## Problem

Regulatory and compliance staff at banks and financial institutions need quick, accurate answers from lengthy regulatory documents (e.g., Bangladesh Bank PSD circulars, MFS regulations). Manual search is slow and error-prone. Generic chatbots hallucinate or lack domain grounding.

This project solves that by **retrieving** relevant passages from indexed documents and **generating** answers grounded in those sources, with citations for auditability.

## Features

- **Vector similarity search** — Vertex AI embeddings + FAISS fallback for fast retrieval
- **Grounded answers** — Gemini LLM generates responses with inline citations `[1]`, `[2]`
- **Conversation support** — Multi-turn chat with configurable context window
- **Folder filtering** — Restrict search to specific document folders
- **OAuth authentication** — Google sign-in with domain allowlist
- **BigQuery logging** — Conversation history and auth events for analytics

## Requirements

- **Python 3.11+**
- **uv** — Python package manager ([install](https://docs.astral.sh/uv/getting-started/installation/))
- **Google Cloud** — GCP project with:
  - Vertex AI enabled
  - Cloud Storage buckets for document indices
  - BigQuery dataset (optional, for conversation logging)
- **Credentials** (in `.gcloud/`):
  - `keyfile.json` — GCP service account with Vertex AI, Storage, BigQuery access
  - `oauth_credentials.json` — OAuth 2.0 client for Google sign-in

## Dependencies

Core dependencies (from `config/requirements.txt`):

| Category | Packages |
|----------|----------|
| Web | `streamlit`, `fastapi`, `uvicorn` |
| RAG / ML | `langchain`, `langchain-community`, `faiss-cpu` |
| Google Cloud | `google-cloud-aiplatform`, `google-cloud-storage`, `google-auth`, `google-auth-oauthlib` |
| Utilities | `python-dotenv`, `numpy`, `pandas`, `watchdog` |

See `pyproject.toml` and `config/requirements.txt` for the full list.

## Setup

### 1. Clone and enter the project

```bash
git clone https://github.com/urmisen/vector-rag-powered-chatbot.git
cd vector-rag-powered-chatbot
```

### 2. Add credentials

Create a `.gcloud/` directory and add (do not commit these):

- `.gcloud/keyfile.json` — GCP service account JSON
- `.gcloud/oauth_credentials.json` — OAuth client config from Google Cloud Console

### 3. Initialize environment

```bash
./init.sh dev
```

This will:

- Create `config/.env` with deployment variables
- Create a Python 3.11 virtual environment in `.venv`
- Install dependencies from `config/requirements.txt`

For production:

```bash
./init.sh prod
```

### 4. Environment variables

`init.sh` generates `config/.env`. Key variables:

| Variable | Description |
|----------|-------------|
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to service account JSON |
| `OAUTH_CREDENTIALS_FILE` | Path to OAuth credentials |
| `ALLOWED_DOMAIN` | Email domain for sign-in (e.g. `pathao.com`) |
| `GCP_PROJECT_ID` | GCP project ID |
| `DATA_BUCKET_NAME` | GCS bucket for sentence/index data |
| `FAISS_BUCKET_NAME` | GCS bucket for FAISS index |
| `INDEX_NAME` | Index name (e.g. `regulon_index_dev`) |
| `OAUTH_REDIRECT_URI` | OAuth redirect URL (must match Console config) |

## How to Run

### Local (development)

```bash
./startup.sh dev
```

This will:

1. Sync sentence and FAISS data from GCS to `data/`
2. Warm up RAG services
3. Start FastAPI on port 8000 and Streamlit on port 8501

Open the chat UI at **http://localhost:8501**.

### Local (production env vars)

```bash
./startup.sh prod
```

### Docker

```bash
docker compose up -d
```

Ensure `.gcloud/` credentials are present and `data/` is mounted. The container runs both FastAPI (8000) and Streamlit (8501).

Access the app at **http://localhost:8501**.

### Manual run (without startup script)

```bash
source .venv/bin/activate
export $(grep -v '^#' config/.env | xargs)

# Optional: sync data from GCS
PYTHONPATH=. python scripts/sync_index_data.py --force

# Start API
PYTHONPATH=. python -m uvicorn app.interfaces.api.api:app --host 0.0.0.0 --port 8000 &

# Start Streamlit
PYTHONPATH=. python -m streamlit run app.interfaces.web/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

## Project Structure

```
├── app/
│   ├── auth/           # OAuth and authentication
│   ├── client.py       # MCP client, conversation handling
│   ├── core/           # RAG manager, models
│   ├── infra/          # BigQuery, logging, background client
│   ├── interfaces/     # FastAPI, Streamlit
│   └── utils/          # RAG profiles, query helpers
├── config/             # requirements.txt, .env (generated)
├── data/               # FAISS index, sentence pickles (synced from GCS)
├── scripts/
│   ├── sync_index_data.py   # Sync indices from GCS
│   └── warmup_services.py   # Pre-warm RAG
├── init.sh             # Environment setup
├── startup.sh          # Run API + Streamlit
└── docker-compose.yml
```

## License

See [LICENSE](LICENSE) if present.
