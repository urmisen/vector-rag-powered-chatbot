# Vector RAG Bot

A chatbot that answers questions about banking and payment regulations by searching your documents and generating grounded answers with citations.

---

## Quick Start

```bash
git clone https://github.com/urmisen/vector-rag-powered-chatbot.git
cd vector-rag-powered-chatbot
```

1. **Add credentials** — Put `keyfile.json` and `oauth_credentials.json` in `.gcloud/`
2. **Setup** — Run `./init.sh dev`
3. **Run** — Run `./startup.sh dev`
4. **Open** — Go to http://localhost:8501

---

## What It Does

Regulatory staff need fast, accurate answers from long documents (PSD circulars, MFS regulations, etc.). Manual search is slow; generic chatbots hallucinate.

This bot **retrieves** relevant passages from your indexed documents and **generates** answers grounded in those sources, with inline citations like `[1]`, `[2]` for auditability.

---

## Features

| Feature | Description |
|--------|-------------|
| Vector search | Vertex AI embeddings + FAISS for fast retrieval |
| Grounded answers | Gemini LLM with inline citations |
| Multi-turn chat | Conversation history with configurable context |
| Folder filters | Restrict search to specific document folders |
| OAuth | Google sign-in with domain allowlist |
| Logging | BigQuery for conversations and auth events |

---

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (package manager)
- GCP project with Vertex AI, Cloud Storage, and (optionally) BigQuery
- Credentials in `.gcloud/`:
  - `keyfile.json` — GCP service account
  - `oauth_credentials.json` — OAuth client from Google Cloud Console

---

## Setup

### 1. Clone

```bash
git clone https://github.com/urmisen/vector-rag-powered-chatbot.git
cd vector-rag-powered-chatbot
```

### 2. Add credentials

Create `.gcloud/` and add (do not commit):

- `keyfile.json` — Service account with Vertex AI, Storage, BigQuery access
- `oauth_credentials.json` — OAuth client config

### 3. Initialize

```bash
./init.sh dev
```

Creates `config/.env` and a Python venv with dependencies. Use `./init.sh prod` for production.

---

## Run

### Local

```bash
./startup.sh dev
```

Syncs data from GCS, warms up RAG, and starts the app. Open **http://localhost:8501**.

### Docker

```bash
docker compose up -d
```

Ensure `.gcloud/` and `data/` are available. App runs on port 8501.

---

## Configuration

`init.sh` generates `config/.env`. Important variables:

- `GOOGLE_APPLICATION_CREDENTIALS` — Path to service account JSON
- `ALLOWED_DOMAIN` — Email domain for sign-in (e.g. `pathao.com`)
- `GCP_PROJECT_ID` — GCP project ID
- `DATA_BUCKET_NAME`, `FAISS_BUCKET_NAME` — GCS buckets for indices
- `OAUTH_REDIRECT_URI` — Must match your OAuth config in Google Cloud Console

---

## Project Layout

```
app/
├── auth/          OAuth
├── client.py      Conversation handling
├── core/          RAG manager
├── infra/         BigQuery, logging
└── interfaces/    FastAPI, Streamlit

scripts/
├── sync_index_data.py   Sync indices from GCS
└── warmup_services.py   Pre-warm RAG

init.sh      Setup env and venv
startup.sh   Run API + Streamlit
```
