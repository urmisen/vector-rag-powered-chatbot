#!/usr/bin/env python3
"""
Sync the latest sentence and FAISS index data from GCS into the local data/ folder.
This ensures startup has warmed caches before the application or warmup scripts run.
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Optional, Tuple

try:
    from dotenv import load_dotenv
except ImportError:
    pass  # dotenv is optional

try:
    from google.cloud import storage
    from google.oauth2 import service_account
except ImportError as err:
    print(f"[sync] ERROR: Google Cloud dependencies are missing: {err}")
    sys.exit(1)

def _get_project_root() -> Path:
    """Return the project root directory (where pyproject.toml lives)."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: two levels up from scripts/ subpackages
    return current.parents[2]


PROJECT_ROOT = _get_project_root()

# Load environment variables from config/.env if it exists
env_path = PROJECT_ROOT / "config" / ".env"
if env_path.exists():
    try:
        load_dotenv(env_path)
    except NameError:
        pass  # dotenv not available, skip
DATA_DIR = PROJECT_ROOT / "data"
FAISS_DIR = DATA_DIR / "faiss_index"


def log(msg: str) -> None:
    print(f"[sync] {msg}")


def ensure_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise RuntimeError(f"Environment variable '{var_name}' is required but not set.")
    return value


def build_storage_client(credentials_path: Path, project_id: str) -> storage.Client:
    credentials = service_account.Credentials.from_service_account_file(
        str(credentials_path),
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return storage.Client(credentials=credentials, project=project_id)


def maybe_remove(path: Path) -> None:
    if path.is_file():
        path.unlink()


def sync_sentences(
    client: storage.Client,
    bucket_name: str,
    index_name: str,
    dest_pickle: Path,
    force: bool,
) -> None:
    bucket = client.bucket(bucket_name)
    DATA_DIR.mkdir(exist_ok=True)

    if force:
        maybe_remove(dest_pickle)

    if dest_pickle.exists():
        log(f"Sentences pickle already present: {dest_pickle}")
        return

    pkl_blob_name = f"{index_name}_sentences.pkl"
    json_blob_name = f"{index_name}_sentences.json"
    pkl_blob = bucket.blob(pkl_blob_name)

    if pkl_blob.exists():
        log(f"Downloading {pkl_blob_name} -> {dest_pickle}")
        pkl_blob.download_to_filename(dest_pickle)
        log("Sentence pickle download complete")
        return

    json_blob = bucket.blob(json_blob_name)
    if not json_blob.exists():
        raise RuntimeError(
            f"Neither {pkl_blob_name} nor {json_blob_name} found in bucket {bucket_name}"
        )

    log(f"{pkl_blob_name} missing. Building pickle from {json_blob_name}")
    content = json_blob.download_as_text()
    sentences = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        sentences.append(json.loads(line))

    with dest_pickle.open("wb") as fh:
        pickle.dump(sentences, fh)

    log(f"Built pickle with {len(sentences)} sentences at {dest_pickle}")


def sync_faiss(
    client: storage.Client,
    bucket_name: str,
    index_name: str,
    dest_pickle: Path,
    dest_folder: Path,
    force: bool,
) -> None:
    if not bucket_name:
        log("FAISS bucket not configured; skipping FAISS sync.")
        return

    bucket = client.bucket(bucket_name)
    dest_folder.mkdir(parents=True, exist_ok=True)

    if force:
        maybe_remove(dest_pickle)
        for name in ("index.faiss", "index.pkl"):
            maybe_remove(dest_folder / name)

    if dest_pickle.exists():
        log(f"FAISS pickle already present: {dest_pickle}")
        return

    pkl_blob_name = f"{index_name}_faiss_sentences.pkl"
    pkl_blob = bucket.blob(pkl_blob_name)
    if pkl_blob.exists():
        log(f"Downloading {pkl_blob_name} -> {dest_pickle}")
        pkl_blob.download_to_filename(dest_pickle)
        log("FAISS pickle download complete")
        return

    log(f"{pkl_blob_name} not found. Downloading raw FAISS index files...")
    required_files = ("index.faiss", "index.pkl")
    for filename in required_files:
        blob = bucket.blob(filename)
        if not blob.exists():
            raise RuntimeError(
                f"FAISS file '{filename}' not found in bucket {bucket_name}"
            )
        target_file = dest_folder / filename
        log(f"Downloading {filename} -> {target_file}")
        blob.download_to_filename(target_file)
    log("FAISS raw files downloaded successfully")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync sentence and FAISS data from GCS into data/."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force refresh by deleting any existing local cache before downloading.",
    )
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    try:
        bucket_name = os.getenv("BUCKET_NAME") or os.getenv("DATA_BUCKET_NAME")
        if not bucket_name:
            raise RuntimeError("Environment variable 'BUCKET_NAME' (or DATA_BUCKET_NAME) is required but not set.")
        sentences_bucket = (
            os.getenv("SENTENCE_BUCKET_NAME")
            or os.getenv("EMBEDDING_BUCKET_NAME")
            or bucket_name
        )
        index_name = ensure_env("INDEX_NAME")
        project_id = ensure_env("GCP_PROJECT_ID")
        credentials_path = Path(ensure_env("GOOGLE_APPLICATION_CREDENTIALS")).expanduser()
    except RuntimeError as err:
        log(f"ERROR: {err}")
        return 1

    faiss_bucket = os.getenv("FAISS_BUCKET_NAME", "")

    if not credentials_path.exists():
        log(f"ERROR: Credentials file not found: {credentials_path}")
        return 1

    try:
        client = build_storage_client(credentials_path, project_id)
    except Exception as err:
        log(f"ERROR: Failed to build storage client: {err}")
        return 1

    try:
        sentences_path = DATA_DIR / f"{index_name}_sentences.pkl"
        sync_sentences(client, sentences_bucket, index_name, sentences_path, args.force)

        faiss_pickle_path = DATA_DIR / f"{index_name}_faiss_sentences.pkl"
        sync_faiss(client, faiss_bucket, index_name, faiss_pickle_path, FAISS_DIR, args.force)

        log("Data sync completed successfully.")
        return 0
    except Exception as err:
        log(f"ERROR: {err}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
