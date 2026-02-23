import os
import json
import re
import numpy as np
import pickle
import time
import logging
import threading
import concurrent.futures
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from typing import List, Dict, Optional, Tuple, Set
try:
    from google.cloud import aiplatform
    from google.cloud import storage
    from google.oauth2 import service_account
    from vertexai.language_models import TextEmbeddingModel
    from vertexai.generative_models import GenerativeModel
    import vertexai
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False
    print("Vertex AI dependencies not available, falling back to local mode")

    # FAISS imports
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import FakeEmbeddings
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS dependencies not available")

from langchain_core.prompts import PromptTemplate
from app.infra.logger import logger
from app.utils.rag_query_helpers import build_query_variants, is_consequence_question, extract_keywords_from_query, is_definitional_question
from app.utils.rag_profiles import (
    DEFAULT_PROFILE_NAME,
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_SKIP_PATTERNS,
    RAGProfile,
    get_rag_profile,
)

class RAGManager:
    _shared_lock = threading.Lock()
    _shared_state_by_profile: Dict[str, Dict[str, object]] = {}
    _warmup_lock = threading.Lock()
    _warmup_threads: Dict[str, threading.Thread] = {}

    def __init__(self, profile_name: Optional[str] = None, profile_path: Optional[str] = None):
        self.logger = logger.getChild("RAG")
        
        env_profile_name = os.getenv("RAG_PROFILE_NAME", DEFAULT_PROFILE_NAME)
        resolved_profile_name = profile_name or env_profile_name
        self.profile_name = (resolved_profile_name or DEFAULT_PROFILE_NAME).strip().lower() or DEFAULT_PROFILE_NAME
        self.profile_path = profile_path or os.getenv("RAG_PROFILE_PATH")
        self.profile: RAGProfile = get_rag_profile(self.profile_name, self.profile_path, self.logger)
        self.profile_metadata = dict(self.profile.metadata or {})
        self.logger.info(f"Using RAG profile '{self.profile.name}'")
        
        # Configuration from environment variables
        self.project_id = os.getenv("GCP_PROJECT_ID")
        self.location = os.getenv("GCP_LOCATION", "us-central1")
        self.bucket_name = os.getenv("BUCKET_NAME")
        self.index_name = os.getenv("INDEX_NAME")
        self.service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        # FAISS bucket configuration
        self.faiss_bucket_name = os.getenv("FAISS_BUCKET_NAME")
        # Latency alert threshold (seconds) for instrumentation logs
        self.latency_warning_threshold = float(os.getenv("RAG_LATENCY_WARN_THRESHOLD", "2.0"))
        # Vertex AI model configuration
        base_model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash")
        self.vertex_ai_model = base_model_name
        self.vertex_ai_fast_model = base_model_name
        self.fast_model_char_limit = int(os.getenv("RAG_FAST_MODEL_PROMPT_LIMIT", "6000"))
        self.use_vertex_streaming = os.getenv("VERTEX_AI_ENABLE_STREAMING", "true").lower() in ("true", "1", "yes", "y")  # Enabled by default for faster responses
        self.streaming_first_token_log = os.getenv("RAG_LOG_STREAMING_FIRST_TOKEN", "false").lower() in ("true", "1", "yes", "y")
        self.fast_response_timeout = float(os.getenv("RAG_FAST_RESPONSE_TIMEOUT", "0.4"))  # seconds to wait for fast model before escalating
        self.fast_model_only = os.getenv("RAG_FAST_MODEL_ONLY", "true").lower() in ("true", "1", "yes", "y")  # Skip default model entirely for speed - default to True
        self.default_generation_config = {
            "temperature": float(os.getenv("RAG_DEFAULT_TEMPERATURE", "0.35")),  # Optimized: 0.3-0.4 range
            "max_output_tokens": int(os.getenv("RAG_DEFAULT_MAX_TOKENS", "8192")),  # Maximum supported by Vertex AI to prevent truncation
            "top_p": float(os.getenv("RAG_DEFAULT_TOP_P", "0.8")),
            "top_k": int(os.getenv("RAG_DEFAULT_TOP_K", "40")),
        }
        self.fast_generation_config = {
            "temperature": float(os.getenv("RAG_FAST_TEMPERATURE", "0.3")),  # Optimized: 0.3-0.4 range
            "max_output_tokens": int(os.getenv("RAG_FAST_MAX_TOKENS", "6000")),  # Increased to prevent truncation in fast model
            "top_p": float(os.getenv("RAG_FAST_TOP_P", "0.9")),
            "top_k": int(os.getenv("RAG_FAST_TOP_K", "32")),
        }
        
        # File paths - directly in bucket root, not in data directory
        self.sentence_file_path = f"{self.index_name}_sentences.json"
        self.pkl_cache_path = f"data/{self.index_name}_sentences.pkl"
        # FAISS local path
        self.faiss_local_path = f"data/faiss_index"
        # FAISS pickle file path
        self.faiss_pkl_cache_path = f"data/{self.index_name}_faiss_sentences.pkl"
        
        # Initialize services
        self.storage_client = None
        self.embedding_model = None
        self.generative_model = None
        self.fast_generative_model = None
        self.sentences_data = []
        self.embeddings_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.embedding_cache_max_size = int(os.getenv("RAG_EMBEDDING_CACHE_SIZE", "2048"))  # Increased from 1024 for better cache hit rate
        # Cache for keyword extraction results
        self.keyword_extraction_cache: OrderedDict[str, List[str]] = OrderedDict()
        self.keyword_cache_max_size = int(os.getenv("RAG_KEYWORD_CACHE_SIZE", "1024"))  # Increased from 512 for better cache hit rate
        # Persistent cache of per-sentence embeddings used by the slow fallback path
        self.sentence_embedding_cache: List[Optional[np.ndarray]] = []
        self.embedding_cache_lock = threading.Lock()
        # Vertex AI embedding API limit is 20,000 tokens per request; ~750 tokens/sentence
        # => ~20 sentences max. Use 16 to stay safely under the limit.
        self.embedding_prefetch_batch_size = int(os.getenv("RAG_EMBEDDING_PREFETCH_BATCH", "16"))
        self.embedding_fallback_limit = int(os.getenv("RAG_EMBEDDING_FALLBACK_LIMIT", "200"))
        self.prompt_chars_per_token = float(os.getenv("RAG_PROMPT_CHARS_PER_TOKEN", "4.0"))
        self.dynamic_token_factor = float(os.getenv("RAG_DYNAMIC_TOKEN_FACTOR", "1.5"))
        self._embedding_warmup_started = False
        # FAISS vector store
        self.faiss_vector_store = None
        
        # Performance optimization: Precompute document indices for faster lookup
        self.document_indices = {}
        # Keyword indices for fast keyword-based filtering
        self.keyword_indices: Dict[str, Set[int]] = {}
        # Reverse ID to index mapping for O(1) lookups (critical for performance)
        self.id_to_index: Dict[str, int] = {}
        
        # Store references for the current session
        self.session_references = {}
        
        # Folder filtering for targeted document search (supports multiple folders)
        self.active_folder_filters = []  # List of active folder filters
        self.available_folders = []  # Cached list of available folder names (computed during initialization)
        
        # Flag to indicate if we should use FAISS as primary method (now False - Vertex AI is primary)
        self.use_faiss_primary = False
        self.keyword_filter_limit = int(os.getenv("RAG_KEYWORD_FILTER_LIMIT", "100"))  # Reduced from 200 for faster responses
        self.context_sentence_limit = int(os.getenv("RAG_CONTEXT_SENTENCE_LIMIT", "4"))  # Reduced from 8 to 4-5 for faster responses
        self.neighbor_sentence_window = int(os.getenv("RAG_NEIGHBOR_WINDOW", "0"))  # Reduced from 1, disable by default for speed
        # Query variant configuration for performance
        self.max_query_variants = int(os.getenv("RAG_MAX_QUERY_VARIANTS", "1"))  # Limit to 1 variant max for speed
        self.enable_query_variants = os.getenv("RAG_ENABLE_QUERY_VARIANTS", "false").lower() in ("true", "1", "yes", "y")  # Disabled by default for speed
        # Keyword filtering configuration
        self.keyword_filter_enabled = os.getenv("RAG_KEYWORD_FILTER_ENABLED", "true").lower() in ("true", "1", "yes", "y")
        self.keyword_match_strategy = os.getenv("RAG_KEYWORD_MATCH_STRATEGY", "OR").upper()  # "AND" or "OR"
        self.keyword_filter_threshold = int(os.getenv("RAG_KEYWORD_FILTER_THRESHOLD", "1"))  # Minimum matches required
        # Response caching for exact and fuzzy query matches
        self.response_cache: OrderedDict[str, str] = OrderedDict()
        self.response_cache_max_size = int(os.getenv("RAG_RESPONSE_CACHE_SIZE", "10000"))  # Increased to 10000 for better hit rate
        # Prompt assets placeholders (shared later)
        self.prompt_template = None
        self.skip_patterns: List[str] = []

        self._apply_profile_overrides()
        
        profiler = None
        with self.__class__._shared_lock:
            shared_state = self.__class__._shared_state_by_profile.get(self.profile_name)
            if shared_state is None:
                profiler = self._start_latency_session("rag_bootstrap")
                try:
                    with self._stage(profiler, "initialize_services"):
                        self.initialize_services()
                    with self._stage(profiler, "load_sentences"):
                        self.load_sentences()
                    with self._stage(profiler, "build_folder_list"):
                        self._build_folder_list()
                    with self._stage(profiler, "build_document_indices"):
                        self._build_document_indices()
                    with self._stage(profiler, "build_keyword_indices"):
                        self._build_keyword_indices()
                    with self._stage(profiler, "build_id_index"):
                        self._build_id_to_index_map()
                    with self._stage(profiler, "initialize_prompts"):
                        self._initialize_prompt_assets()
                    self._ensure_sentence_embedding_cache_initialized()

                    self.__class__._shared_state_by_profile[self.profile_name] = {
                        "storage_client": self.storage_client,
                        "embedding_model": self.embedding_model,
                        "generative_model": self.generative_model,
                        "fast_generative_model": self.fast_generative_model,
                        "faiss_vector_store": self.faiss_vector_store,
                        "sentences_data": self.sentences_data,
                        "document_indices": self.document_indices,
                        "keyword_indices": self.keyword_indices,
                        "id_to_index": self.id_to_index,
                        "prompt_template": self.prompt_template,
                        "skip_patterns": tuple(self.skip_patterns),
                        "use_faiss_primary": self.use_faiss_primary,
                        "vertex_ai_model": self.vertex_ai_model,
                        "vertex_ai_fast_model": self.vertex_ai_fast_model,
                        "profile_name": self.profile_name,
                        "profile_metadata": dict(self.profile_metadata),
                        "sentence_embedding_cache": self.sentence_embedding_cache,
                        "available_folders": self.available_folders.copy() if self.available_folders else [],
                    }
                finally:
                    self._end_latency_session(profiler)
            else:
                # Reuse cached heavy resources
                self.logger.debug("Reusing cached RAG resources")

        # Apply shared state to this instance (for first and subsequent creations)
        self._adopt_shared_state()
        self._ensure_sentence_embedding_cache_initialized()
        self._maybe_start_embedding_warmup()
        # Warmup runs asynchronously so we don't block login flows
        self._schedule_async_warmup()
    
    def _apply_profile_overrides(self):
        """Apply per-profile overrides to the RAG manager configuration."""
        profile = getattr(self, "profile", None)
        if profile is None:
            self.skip_patterns = list(DEFAULT_SKIP_PATTERNS)
            return
        
        if profile.keyword_filter_limit is not None:
            self.keyword_filter_limit = profile.keyword_filter_limit
        if profile.context_sentence_limit is not None:
            self.context_sentence_limit = profile.context_sentence_limit
        if profile.neighbor_sentence_window is not None:
            self.neighbor_sentence_window = profile.neighbor_sentence_window
        
        # Apply keyword filtering settings from profile
        if profile.keyword_filter_enabled is not None:
            self.keyword_filter_enabled = profile.keyword_filter_enabled
        if profile.keyword_match_strategy is not None:
            self.keyword_match_strategy = profile.keyword_match_strategy.upper()
        if profile.keyword_filter_threshold is not None:
            self.keyword_filter_threshold = profile.keyword_filter_threshold
        
        self.skip_patterns = list(profile.skip_patterns or DEFAULT_SKIP_PATTERNS)
        merged_default, merged_fast = profile.merge_generation_configs(
            self.default_generation_config,
            self.fast_generation_config,
        )
        self.default_generation_config = merged_default
        self.fast_generation_config = merged_fast
        self.prompt_template = profile.prompt_template
        self.profile_metadata = dict(profile.metadata or {})

    def _ensure_sentence_embedding_cache_initialized(self):
        """Ensure the per-sentence embedding cache matches the loaded corpus size."""
        sentence_count = len(self.sentences_data or [])
        if sentence_count <= 0:
            self.sentence_embedding_cache = []
            return
        if not self.sentence_embedding_cache or len(self.sentence_embedding_cache) != sentence_count:
            self.sentence_embedding_cache = [None] * sentence_count

    def _prefetch_sentence_embeddings(self, indices: List[int]):
        """Preload sentence embeddings for the fallback retrieval path."""
        if (
            not indices
            or not (VERTEX_AI_AVAILABLE and self.embedding_model)
            or not self.sentences_data
        ):
            return

        self._ensure_sentence_embedding_cache_initialized()
        missing_indices = [
            idx for idx in indices
            if 0 <= idx < len(self.sentence_embedding_cache)
            and self.sentence_embedding_cache[idx] is None
        ]
        if not missing_indices:
            return

        batch_size = max(1, self.embedding_prefetch_batch_size)
        for start in range(0, len(missing_indices), batch_size):
            batch_indices = missing_indices[start:start + batch_size]
            texts = []
            valid_indices = []
            for idx in batch_indices:
                sentence_item = self.sentences_data[idx]
                sentence_text = sentence_item.get('sentence')
                if sentence_text:
                    texts.append(sentence_text)
                    valid_indices.append(idx)

            if not texts:
                continue

            embeddings = self.embedding_model.get_embeddings(texts)
            with self.embedding_cache_lock:
                for idx, embedding_obj in zip(valid_indices, embeddings):
                    vector = self._convert_embedding_to_array(embedding_obj)
                    self.sentence_embedding_cache[idx] = vector
    
    def _maybe_start_embedding_warmup(self):
        """Start background sentence embedding warmup so first query stays fast."""
        if self._embedding_warmup_started:
            return
        if not (VERTEX_AI_AVAILABLE and self.embedding_model):
            return
        if not self.sentences_data:
            return
        self._embedding_warmup_started = True

        def _warmup_worker():
            try:
                all_indices = list(range(len(self.sentences_data)))
                self.logger.info(f"Starting sentence embedding warmup for {len(all_indices)} sentences...")
                self._prefetch_sentence_embeddings(all_indices)
                self.logger.info("Sentence embedding warmup completed.")
            except Exception as exc:
                self.logger.warning(f"Sentence embedding warmup failed: {exc}")

        threading.Thread(target=_warmup_worker, name="rag-embedding-warmup", daemon=True).start()
    
    def _adopt_shared_state(self):
        shared = self.__class__._shared_state_by_profile.get(self.profile_name) or {}
        if not shared:
            return
        
        self.storage_client = shared.get("storage_client")
        self.embedding_model = shared.get("embedding_model")
        self.generative_model = shared.get("generative_model")
        self.fast_generative_model = shared.get("fast_generative_model")
        self.faiss_vector_store = shared.get("faiss_vector_store")
        self.sentences_data = shared.get("sentences_data", [])
        self.document_indices = shared.get("document_indices", {})
        self.keyword_indices = shared.get("keyword_indices", {})
        self.id_to_index = shared.get("id_to_index", {})
        self.prompt_template = shared.get("prompt_template")
        shared_skip_patterns = shared.get("skip_patterns")
        if shared_skip_patterns:
            # Copy to avoid accidental mutation of shared tuple
            self.skip_patterns = list(shared_skip_patterns)
        self.use_faiss_primary = shared.get("use_faiss_primary", self.use_faiss_primary)
        self.vertex_ai_model = shared.get("vertex_ai_model", self.vertex_ai_model)
        self.vertex_ai_fast_model = shared.get("vertex_ai_fast_model", self.vertex_ai_fast_model)
        self.profile_metadata = shared.get("profile_metadata", self.profile_metadata)
        shared_embedding_cache = shared.get("sentence_embedding_cache")
        if shared_embedding_cache is not None:
            self.sentence_embedding_cache = shared_embedding_cache
        shared_available_folders = shared.get("available_folders")
        if shared_available_folders is not None:
            # Copy to avoid accidental mutation of shared list
            self.available_folders = list(shared_available_folders)
    
    def _schedule_async_warmup(self):
        """Kick off asynchronous warmup so login flow is non-blocking."""
        # If warmup has already run or there is nothing to warm, skip
        if self.profile_name not in self.__class__._shared_state_by_profile:
            return
        
        with self.__class__._warmup_lock:
            thread = self.__class__._warmup_threads.get(self.profile_name)
            if thread and thread.is_alive():
                return
            
            if not any([self.faiss_vector_store, self.embedding_model, self.generative_model]):
                return
            
            def _run_warmup():
                try:
                    self.logger.info("Starting asynchronous RAG warmup cycle")
                    self._warmup_resources()
                    shared_state = self.__class__._shared_state_by_profile.get(self.profile_name)
                    if shared_state is not None:
                        shared_state["warmup_complete"] = True
                    self.logger.info("Asynchronous RAG warmup complete")
                except Exception as exc:
                    self.logger.warning(f"Asynchronous RAG warmup failed: {exc}")
            
            warmup_thread = threading.Thread(
                target=_run_warmup,
                name="rag-warmup",
                daemon=True,
            )
            self.__class__._warmup_threads[self.profile_name] = warmup_thread
            warmup_thread.start()
    
    def _start_latency_session(self, label: str) -> Dict:
        """Start a latency profiling session."""
        start = time.perf_counter()
        profiling_context = {
            "label": label,
            "start": start,
            "stages": [],
        }
        return profiling_context
    
    @contextmanager
    def _profile_stage(self, profiler: Optional[Dict], stage_name: str):
        """Context manager to record timing for a specific stage."""
        if profiler is None:
            yield
            return
        
        stage_start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - stage_start
            profiler["stages"].append((stage_name, elapsed))
    
    def _stage(self, profiler: Optional[Dict], stage_name: str):
        """Return a context manager for an optional profiling stage."""
        if profiler is None:
            return nullcontext()
        return self._profile_stage(profiler, stage_name)
    
    def _end_latency_session(self, profiler: Optional[Dict], **extra_metadata) -> float:
        """Finalize a latency profiling session and log the breakdown."""
        if profiler is None:
            return 0.0
        
        total_elapsed = time.perf_counter() - profiler["start"]
        breakdown_parts = [f"{name}={duration * 1000:.1f}ms" for name, duration in profiler["stages"]]
        if extra_metadata:
            breakdown_parts.extend([f"{key}={value}" for key, value in extra_metadata.items()])
        
        breakdown_str = ", ".join(breakdown_parts)
        
        # Enhanced performance monitoring: warn on slow stages (>500ms)
        slow_stages = [(name, duration) for name, duration in profiler["stages"] if duration > 0.5]
        if slow_stages:
            slow_stage_names = ", ".join([f"{name}={duration*1000:.0f}ms" for name, duration in slow_stages])
            self.logger.warning(f"Slow stages detected in {profiler['label']}: {slow_stage_names}")
        
        # Critical threshold for 1s target: warn if total exceeds 1.5s
        critical_threshold = 1.5
        log_level = logging.WARNING if total_elapsed > critical_threshold else (
            logging.WARNING if total_elapsed > self.latency_warning_threshold else logging.INFO
        )
        
        self.logger.log(
            log_level,
            f"Latency profile ({profiler['label']}): total={total_elapsed * 1000:.1f}ms | {breakdown_str}"
        )
        
        # Log if we're exceeding 1s target
        if total_elapsed > 1.0:
            self.logger.warning(f"Response time {total_elapsed:.2f}s exceeds 1s target. Breakdown: {breakdown_str}")
        
        return total_elapsed
    
    def _initialize_prompt_assets(self):
        """Pre-build prompt templates and reusable assets to avoid per-request overhead."""
        self.logger.info(f"Initializing prompt assets for RAG profile '{self.profile_name}'")
        template_string = (
            self.profile.prompt_template
            if getattr(self, "profile", None) and self.profile.prompt_template
            else DEFAULT_PROMPT_TEMPLATE
        )
        self.prompt_template = PromptTemplate(
            template=template_string,
            input_variables=["context", "document_context", "document_mapping", "question"]
        )
        
        # Precompute repeated string patterns to avoid recreating them per query
        if getattr(self, "profile", None) and self.profile.skip_patterns:
            self.skip_patterns = list(self.profile.skip_patterns)
        elif not self.skip_patterns:
            self.skip_patterns = list(DEFAULT_SKIP_PATTERNS)
    
    def _warmup_resources(self):
        """Warm up Vertex AI embedding model (primary) and FAISS (fallback) to avoid cold starts."""
        self.logger.info("Warming up Vertex AI and model resources")
        
        # Warm embedding model first (primary method)
        if VERTEX_AI_AVAILABLE and self.embedding_model:
            try:
                self.embedding_model.get_embeddings(["warmup text"])
                self.logger.debug("Embedding model warmup completed")
            except Exception as embedding_error:
                self.logger.warning(f"Embedding model warmup skipped: {embedding_error}")
        
        # Warm FAISS by executing a lightweight similarity call (fallback method)
        if self.faiss_vector_store:
            try:
                self.faiss_vector_store.similarity_search_with_score("__warmup__", k=1)
                self.logger.debug("FAISS warmup completed")
            except Exception as faiss_error:
                self.logger.warning(f"FAISS warmup skipped due to error: {faiss_error}")
        
        # Warm generative model with a minimal request
        if VERTEX_AI_AVAILABLE and self.generative_model:
            try:
                self.generative_model.generate_content(
                    "Respond with 'Ready'.",
                    generation_config={
                        "temperature": 0.0,
                        "max_output_tokens": 1,
                        "top_p": 0.1,
                        "top_k": 1
                    }
                )
                self.logger.debug("Generative model warmup completed")
            except Exception as gen_error:
                self.logger.warning(f"Generative model warmup skipped: {gen_error}")
        
        # Warm fast generative model if it's distinct from primary
        if VERTEX_AI_AVAILABLE and self.fast_generative_model and self.fast_generative_model is not self.generative_model:
            try:
                self.fast_generative_model.generate_content(
                    "Respond with 'Ready'.",
                    generation_config={
                        "temperature": 0.0,
                        "max_output_tokens": 1,
                        "top_p": 0.1,
                        "top_k": 1
                    }
                )
                self.logger.debug("Fast generative model warmup completed")
            except Exception as fast_error:
                self.logger.warning(f"Fast generative model warmup skipped: {fast_error}")
    
    def _build_folder_list(self):
        """
        Build and cache the list of available folder names from sentences_data.
        Called during initialization to pre-compute folder names for fast access.
        """
        if not self.sentences_data:
            self.logger.warning("No sentences data available, cannot extract folders")
            self.available_folders = []
            return
        
        folders = set()
        for sentence_item in self.sentences_data:
            folder_name = sentence_item.get('folder_name')
            if folder_name and isinstance(folder_name, str) and folder_name.strip():
                folders.add(folder_name.strip())
        
        # Store sorted list for consistent ordering
        self.available_folders = sorted(list(folders))
        self.logger.info(f"Built folder list cache: {len(self.available_folders)} unique folders: {self.available_folders}")
    
    def get_available_folders(self) -> List[str]:
        """
        Get the cached list of available folder names.
        Returns sorted list of unique folder names found in the document metadata.
        """
        # Return cached folder list (computed during initialization)
        return self.available_folders.copy() if self.available_folders else []
    
    def set_folder_filters(self, folder_names: List[str]):
        """
        Set the active folder filters for document search.
        All subsequent searches will be restricted to documents in these folders.
        
        Args:
            folder_names: List of folder names to filter by
        """
        if folder_names and isinstance(folder_names, list):
            self.active_folder_filters = [f.strip() for f in folder_names if f and isinstance(f, str)]
            self.logger.info(f"Folder filters set to: {self.active_folder_filters}")
        else:
            self.logger.warning(f"Invalid folder names provided: {folder_names}")
    
    def add_folder_filter(self, folder_name: str):
        """
        Add a folder to the active filters.
        
        Args:
            folder_name: The folder name to add
        """
        if folder_name and isinstance(folder_name, str):
            folder_name = folder_name.strip()
            if folder_name not in self.active_folder_filters:
                self.active_folder_filters.append(folder_name)
                self.logger.info(f"Added folder filter: {folder_name}. Active filters: {self.active_folder_filters}")
    
    def remove_folder_filter(self, folder_name: str):
        """
        Remove a folder from the active filters.
        
        Args:
            folder_name: The folder name to remove
        """
        if folder_name and isinstance(folder_name, str):
            folder_name = folder_name.strip()
            if folder_name in self.active_folder_filters:
                self.active_folder_filters.remove(folder_name)
                self.logger.info(f"Removed folder filter: {folder_name}. Active filters: {self.active_folder_filters}")
    
    def get_current_folder_filters(self) -> List[str]:
        """
        Get the currently active folder filters.
        
        Returns:
            List of active folder filter names
        """
        return self.active_folder_filters.copy()
    
    def clear_folder_filters(self):
        """Clear all active folder filters, allowing searches across all folders."""
        if self.active_folder_filters:
            self.logger.info(f"Clearing folder filters: {self.active_folder_filters}")
        self.active_folder_filters = []
    
    def _initialize_fast_generation_model(self):
        """Set up the fast-path generative model for low-latency responses."""
        if not VERTEX_AI_AVAILABLE:
            return
        
        # If fast model name matches the default, reuse initialized model
        if self.vertex_ai_fast_model == self.vertex_ai_model and self.generative_model:
            self.fast_generative_model = self.generative_model
            self.logger.info("Fast generative model reusing default model instance")
            return
        
        try:
            self.fast_generative_model = GenerativeModel(self.vertex_ai_fast_model)
            self.logger.info(f"Fast generative model initialized: {self.vertex_ai_fast_model}")
        except Exception as fast_init_error:
            self.logger.warning(f"Failed to initialize fast generative model {self.vertex_ai_fast_model}: {fast_init_error}")
            # Fall back to the default model to ensure availability
            self.fast_generative_model = self.generative_model
    
    def initialize_services(self):
        """Initialize Google Cloud services."""
        self.logger.info("Initializing RAG services...")
        
        # Initialize Vertex AI services first (primary method)
        if not VERTEX_AI_AVAILABLE:
            self.logger.warning("Vertex AI not available, running in local mode")
            return
        
        try:
            # Initialize service account credentials
            credentials = service_account.Credentials.from_service_account_file(
                self.service_account_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            
            # Initialize Cloud Storage client
            self.storage_client = storage.Client(credentials=credentials, project=self.project_id)
            self.logger.info("Cloud Storage client initialized")
            
            # Initialize Vertex AI
            aiplatform.init(project=self.project_id, location=self.location, credentials=credentials)
            vertexai.init(project=self.project_id, location=self.location, credentials=credentials)
            self.logger.info("Vertex AI initialized")
            
            # Initialize embedding model
            try:
                self.embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
                self.logger.info("Embedding model initialized: text-embedding-004")
            except Exception as e1:
                try:
                    self.embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
                    self.logger.info("Embedding model initialized: textembedding-gecko@003")
                except Exception as e2:
                    self.logger.warning(f"Failed to initialize embedding model: {e1}")
                    self.logger.warning("Running in local mode without Vertex AI embedding model")
                    self.embedding_model = None
            
            # Initialize text generation model using the VERTEX_AI_MODEL from environment
            try:
                self.generative_model = GenerativeModel(self.vertex_ai_model)  # Use the environment variable
                self.logger.info(f"Generative model initialized: {self.vertex_ai_model}")
            except Exception as e:
                # Fallback models in order of preference
                fallback_models = ["gemini-2.5-pro", "gemini-1.5-pro", "gemini-1.0-pro", "gemini-pro"]
                initialized = False
                
                for model_name in fallback_models:
                    try:
                        self.generative_model = GenerativeModel(model_name)
                        self.logger.info(f"Generative model initialized: {model_name}")
                        initialized = True
                        break
                    except Exception as e2:
                        self.logger.warning(f"Failed to initialize generative model {model_name}: {e2}")
                        continue
                
                if not initialized:
                    self.logger.warning(f"Generative model not available: {e}")
                    self.generative_model = None
            
            # Initialize fast-path generative model if configured
            self._initialize_fast_generation_model()
            
            # Try to initialize FAISS as fallback method (after Vertex AI is initialized)
            if self._initialize_faiss():
                self.logger.info("FAISS initialized as fallback search method")
            else:
                self.logger.warning("FAISS initialization failed, will use Vertex AI only")
            
        except Exception as e:
            self.logger.warning(f"Error initializing Vertex AI services: {e}")
            self.logger.warning("Running in local mode without Vertex AI services")

    def _initialize_faiss(self) -> bool:
        """Initialize FAISS as the fallback search method."""
        self.logger.info("Initializing FAISS as fallback search method...")
        
        # Only use FAISS if dependencies are available
        if not FAISS_AVAILABLE:
            self.logger.error("FAISS dependencies not available")
            return False
        
        try:
            # Try to load FAISS index
            if self._load_faiss_with_pickle_processing():
                self.logger.info("FAISS index loaded successfully")
                return True
            else:
                self.logger.error("Failed to load FAISS index")
                return False
                
        except Exception as e:
            self.logger.error(f"Error initializing FAISS: {e}")
            return False

    def load_sentences(self):
        """Load sentences from GCS bucket containing embeddings with pickle caching."""
        self.logger.info("Loading document sentences...")
        
        # Load sentence data first (for Vertex AI primary method)
        # Try to load from pickle cache first
        if self._load_from_pickle_cache():
            self.logger.info(f"Loaded {len(self.sentences_data)} sentences from pickle cache")
            self._prepare_sentence_metadata_bulk()
            success = True
        else:
            # If pickle cache doesn't exist or failed, load from GCS
            success = self._load_from_gcs()
            
            # If successfully loaded from GCS, save to pickle cache
            if success:
                self._prepare_sentence_metadata_bulk()
                self._save_to_pickle_cache()
        
        # Try to load FAISS as fallback method (after Vertex AI data is loaded)
        faiss_loaded = False
        if self.faiss_vector_store is None:
            if self._load_faiss_with_pickle_processing():
                self.logger.info("Loaded FAISS as fallback search method")
                faiss_loaded = True
            else:
                self.logger.warning("Failed to load FAISS, will use Vertex AI only")
        
        # Return success based on whether we loaded sentences (primary requirement)
        return success
    
    def _load_from_pickle_cache(self) -> bool:
        """Load sentences data from pickle cache file."""
        try:
            if os.path.exists(self.pkl_cache_path):
                self.logger.info(f"Loading sentences from pickle cache: {self.pkl_cache_path}")
                with open(self.pkl_cache_path, 'rb') as f:
                    self.sentences_data = pickle.load(f)
                self.logger.info(f"Successfully loaded {len(self.sentences_data)} sentences from local pickle cache")
                return True
            else:
                self.logger.info("Pickle cache file not found, will attempt to load from GCS")
                return False
        except Exception as e:
            self.logger.warning(f"Failed to load from pickle cache: {e}")
            return False
    
    def _save_to_pickle_cache(self):
        """Save sentences data to pickle cache file."""
        try:
            # Create data directory if it doesn't exist
            os.makedirs(os.path.dirname(self.pkl_cache_path), exist_ok=True)
            
            self.logger.info(f"Saving sentences to pickle cache: {self.pkl_cache_path}")
            with open(self.pkl_cache_path, 'wb') as f:
                pickle.dump(self.sentences_data, f)
            self.logger.info("Successfully saved to pickle cache")
        except Exception as e:
            self.logger.warning(f"Failed to save to pickle cache: {e}")
    
    def _load_from_gcs(self) -> bool:
        """Load sentences from GCS bucket containing embeddings."""
        self.logger.info("Loading document sentences from GCS bucket...")
        
        # Only use GCS (only if Vertex AI is available)
        if VERTEX_AI_AVAILABLE and self.storage_client and self.bucket_name:
            try:
                bucket = self.storage_client.bucket(self.bucket_name)
                
                # First, try to load pickle file from GCS
                pkl_blob_name = f"{self.index_name}_sentences.pkl"
                pkl_blob = bucket.blob(pkl_blob_name)
                
                if pkl_blob.exists():
                    self.logger.info(f"Loading pickle file from GCS: {pkl_blob_name}")
                    # Download pickle file to local cache
                    os.makedirs(os.path.dirname(self.pkl_cache_path), exist_ok=True)
                    pkl_blob.download_to_filename(self.pkl_cache_path)
                    
                    # Load from the downloaded pickle file
                    with open(self.pkl_cache_path, 'rb') as f:
                        self.sentences_data = pickle.load(f)
                    
                    self.logger.info(f"Loaded {len(self.sentences_data)} sentences from GCS pickle file")
                    return True
                
                # If pickle file doesn't exist, fall back to JSON
                blob = bucket.blob(self.sentence_file_path)
                
                if blob.exists():
                    content = blob.download_as_text()
                    self.sentences_data = []
                    
                    for line in content.strip().split('\n'):
                        if line.strip():
                            sentence_item = json.loads(line)
                            self.sentences_data.append(sentence_item)
                    
                    self.logger.info(f"Loaded {len(self.sentences_data)} sentences from GCS JSON file")
                    return True
                else:
                    self.logger.error(f"File {self.sentence_file_path} not found in bucket {self.bucket_name}")
            except Exception as e:
                self.logger.error(f"Failed to load sentences from GCS bucket: {e}")
        else:
            self.logger.error("Vertex AI services not available or bucket configuration missing")
        
        self.logger.error("Could not load sentences from GCS bucket")
        return False
    
    def _load_faiss_with_pickle_processing(self) -> bool:
        """Load FAISS index with pickle file processing similar to Vertex AI."""
        self.logger.info("Loading FAISS with pickle file processing...")
        
        # Only use FAISS if dependencies are available
        if not FAISS_AVAILABLE:
            self.logger.error("FAISS dependencies not available")
            return False
        
        try:
            # First, try to load from FAISS pickle cache (similar to Vertex AI pickle cache)
            if self._load_faiss_from_pickle_cache():
                self.logger.info("Loaded FAISS from pickle cache")
                return True
            
            # If pickle cache doesn't exist or failed, try to load local FAISS index files directly
            if self._load_local_faiss_index():
                self.logger.info("Loaded FAISS from local index files")
                # Save to pickle cache for faster loading next time
                self._save_faiss_to_pickle_cache()
                return True
            
            # If local files don't exist or failed, load from GCS with pickle processing
            success = self._load_faiss_from_gcs_with_pickle()
            
            # If successfully loaded from GCS, save to FAISS pickle cache
            if success:
                self._save_faiss_to_pickle_cache()
                
            return success
            
        except Exception as e:
            self.logger.error(f"Error loading FAISS with pickle processing: {e}")
            return False
    
    def _load_local_faiss_index(self) -> bool:
        """Load FAISS vector store from local index files."""
        try:
            # Check if local FAISS index files exist
            index_file = os.path.join(self.faiss_local_path, "index.faiss")
            pkl_file = os.path.join(self.faiss_local_path, "index.pkl")
            
            if os.path.exists(index_file) and os.path.exists(pkl_file):
                self.logger.info(f"Loading FAISS from local index files: {self.faiss_local_path}")
                
                # Load FAISS vector store with dummy embeddings
                dummy_embeddings = FakeEmbeddings(size=768)
                self.faiss_vector_store = FAISS.load_local(
                    folder_path=self.faiss_local_path,
                    embeddings=dummy_embeddings,
                    allow_dangerous_deserialization=True
                )
                
                self.logger.info("Successfully loaded FAISS from local index files")
                return True
            else:
                self.logger.info("Local FAISS index files not found, will attempt other loading methods")
                return False
        except Exception as e:
            self.logger.warning(f"Failed to load FAISS from local index files: {e}")
            return False
    
    def _load_faiss_from_pickle_cache(self) -> bool:
        """Load FAISS vector store from pickle cache file (similar to Vertex AI approach)."""
        try:
            if os.path.exists(self.faiss_pkl_cache_path):
                self.logger.info(f"Loading FAISS from pickle cache: {self.faiss_pkl_cache_path}")
                with open(self.faiss_pkl_cache_path, 'rb') as f:
                    self.faiss_vector_store = pickle.load(f)
                self.logger.info("Successfully loaded FAISS from local pickle cache")
                return True
            else:
                self.logger.info("FAISS pickle cache file not found, will attempt to load from GCS")
                return False
        except Exception as e:
            self.logger.warning(f"Failed to load FAISS from pickle cache: {e}")
            return False
    
    def _save_faiss_to_pickle_cache(self):
        """Save FAISS vector store to pickle cache file (similar to Vertex AI approach)."""
        try:
            # Create data directory if it doesn't exist
            os.makedirs(os.path.dirname(self.faiss_pkl_cache_path), exist_ok=True)
            
            self.logger.info(f"Saving FAISS to pickle cache: {self.faiss_pkl_cache_path}")
            with open(self.faiss_pkl_cache_path, 'wb') as f:
                pickle.dump(self.faiss_vector_store, f)
            self.logger.info("Successfully saved FAISS to pickle cache")
        except Exception as e:
            self.logger.warning(f"Failed to save FAISS to pickle cache: {e}")
            # Try to save FAISS index files as alternative
            try:
                self.logger.info("Attempting to save FAISS index files as alternative...")
                self.faiss_vector_store.save_local(self.faiss_local_path)
                self.logger.info(f"Successfully saved FAISS index files to {self.faiss_local_path}")
            except Exception as e2:
                self.logger.warning(f"Failed to save FAISS index files: {e2}")
    
    def _load_faiss_from_gcs_with_pickle(self) -> bool:
        """Load FAISS index from GCS bucket with pickle file processing."""
        self.logger.info("Loading FAISS index from GCS bucket with pickle processing...")
        
        # Only use FAISS if dependencies are available and FAISS bucket is configured
        if not self.storage_client or not self.faiss_bucket_name:
            self.logger.error("Storage client or FAISS bucket not configured")
            return False
        
        try:
            bucket = self.storage_client.bucket(self.faiss_bucket_name)
            
            # First, try to load pickle file from GCS (similar to Vertex AI approach)
            pkl_blob_name = f"{self.index_name}_faiss_sentences.pkl"
            pkl_blob = bucket.blob(pkl_blob_name)
            
            if pkl_blob.exists():
                self.logger.info(f"Loading FAISS pickle file from GCS: {pkl_blob_name}")
                # Download pickle file to local cache
                os.makedirs(os.path.dirname(self.faiss_pkl_cache_path), exist_ok=True)
                pkl_blob.download_to_filename(self.faiss_pkl_cache_path)
                
                # Load from the downloaded pickle file
                with open(self.faiss_pkl_cache_path, 'rb') as f:
                    self.faiss_vector_store = pickle.load(f)
                
                self.logger.info("Loaded FAISS from GCS pickle file")
                return True
            
            # If pickle file doesn't exist, fall back to loading FAISS index files
            # Create data directory if it doesn't exist
            os.makedirs(self.faiss_local_path, exist_ok=True)
            
            # Download FAISS index files
            required_files = ["index.faiss", "index.pkl"]
            downloaded_files = []
            
            for filename in required_files:
                try:
                    local_file_path = os.path.join(self.faiss_local_path, filename)
                    blob = bucket.blob(filename)
                    if blob.exists():
                        blob.download_to_filename(local_file_path)
                        downloaded_files.append(filename)
                        self.logger.info(f"Downloaded FAISS file: {filename}")
                    else:
                        self.logger.error(f"FAISS file not found in bucket: {filename}")
                except Exception as e:
                    self.logger.error(f"Error downloading FAISS file {filename}: {e}")
            
            # Check if all required files were downloaded
            if len(downloaded_files) == len(required_files):
                self.logger.info(f"Downloaded all required FAISS files: {downloaded_files}")
                
                # Load FAISS vector store with dummy embeddings
                dummy_embeddings = FakeEmbeddings(size=768)
                self.faiss_vector_store = FAISS.load_local(
                    folder_path=self.faiss_local_path,
                    embeddings=dummy_embeddings,
                    allow_dangerous_deserialization=True
                )
                
                self.logger.info("FAISS index loaded successfully from GCS")
                return True
            else:
                self.logger.error(f"Failed to download all required FAISS files. Downloaded: {downloaded_files}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to load FAISS index from GCS bucket: {e}")
            return False
    
    def _build_document_indices(self):
        """Build indices for faster document lookup - performance optimization."""
        self.logger.info("Building document indices for faster lookup...")
        self.document_indices = {}
        
        for i, sentence_item in enumerate(self.sentences_data):
            self._prepare_sentence_metadata(sentence_item)
            doc_id = sentence_item.get('document_id')
            if doc_id:
                if doc_id not in self.document_indices:
                    self.document_indices[doc_id] = []
                self.document_indices[doc_id].append(i)
        
        self.logger.info(f"Built indices for {len(self.document_indices)} documents")
    
    def _build_keyword_indices(self):
        """Build keyword indices for faster keyword-based filtering."""
        self.logger.info("Building keyword indices for faster filtering...")
        self.keyword_indices = {}
        
        for i, sentence_item in enumerate(self.sentences_data):
            self._prepare_sentence_metadata(sentence_item)
            
            # Extract keywords from sentence item
            # Support multiple formats: list, comma-separated string, or extract from sentence
            keywords = []
            
            # Check if 'keywords' field exists in the sentence item
            if 'keywords' in sentence_item:
                keyword_field = sentence_item['keywords']
                if isinstance(keyword_field, list):
                    keywords = [str(k).strip().lower() for k in keyword_field if k]
                elif isinstance(keyword_field, str):
                    # Handle comma-separated or space-separated keywords
                    keywords = [k.strip().lower() for k in re.split(r'[,\s]+', keyword_field) if k.strip()]
            
            # If no keywords field, extract from sentence text as fallback
            if not keywords:
                sentence_text = sentence_item.get('sentence', '')
                if sentence_text:
                    # Extract keywords from sentence (simple approach: important words)
                    keywords = extract_keywords_from_query(sentence_text, min_length=4, max_keywords=5)
            
            # Add to inverted index
            for keyword in keywords:
                keyword = keyword.strip().lower()
                if keyword and len(keyword) >= 2:  # Minimum 2 characters
                    if keyword not in self.keyword_indices:
                        self.keyword_indices[keyword] = set()
                    self.keyword_indices[keyword].add(i)
        
        self.logger.info(f"Built keyword indices for {len(self.keyword_indices)} unique keywords across {len(self.sentences_data)} sentences")
    
    def _build_id_to_index_map(self):
        """Build reverse mapping from sentence ID to index for O(1) lookups."""
        self.logger.info("Building ID to index mapping for fast lookups...")
        self.id_to_index = {}
        
        for i, sentence_item in enumerate(self.sentences_data):
            sentence_id = sentence_item.get('id')
            if sentence_id:
                self.id_to_index[str(sentence_id)] = i
        
        self.logger.info(f"Built ID to index mapping for {len(self.id_to_index)} sentences")
    
    def _filter_by_keywords(self, query_keywords: List[str], all_indices: Optional[List[int]] = None) -> List[int]:
        """
        Filter sentence indices by matching query keywords against document keywords.
        
        Args:
            query_keywords: List of normalized keywords extracted from query
            all_indices: Optional list of indices to filter from (defaults to all sentences)
        
        Returns:
            List of sentence indices that match the keyword criteria
        """
        if not self.keyword_filter_enabled or not query_keywords:
            # If filtering disabled or no keywords, return all indices
            if all_indices is not None:
                return all_indices
            return list(range(len(self.sentences_data)))
        
        if not self.keyword_indices:
            # No keyword indices built, return all
            if all_indices is not None:
                return all_indices
            return list(range(len(self.sentences_data)))
        
        # Normalize query keywords
        normalized_query_keywords = [kw.strip().lower() for kw in query_keywords if kw.strip()]
        if not normalized_query_keywords:
            if all_indices is not None:
                return all_indices
            return list(range(len(self.sentences_data)))
        
        # Find matching sentence indices for each keyword
        matching_indices_sets = []
        for keyword in normalized_query_keywords:
            if keyword in self.keyword_indices:
                matching_indices_sets.append(self.keyword_indices[keyword])
        
        if not matching_indices_sets:
            # No matches found, return empty list or fallback to all if threshold allows
            if self.keyword_filter_threshold == 0:
                if all_indices is not None:
                    return all_indices
                return list(range(len(self.sentences_data)))
            return []
        
        # Apply matching strategy (AND or OR)
        if self.keyword_match_strategy == "AND":
            # All keywords must match (set intersection)
            result_indices = matching_indices_sets[0]
            for indices_set in matching_indices_sets[1:]:
                result_indices = result_indices & indices_set
        else:
            # OR: Any keyword matches (set union)
            result_indices = set()
            for indices_set in matching_indices_sets:
                result_indices = result_indices | indices_set
        
        # Apply threshold: minimum number of keywords that must match
        if len(matching_indices_sets) < self.keyword_filter_threshold:
            # Not enough keyword matches, return all as fallback
            if all_indices is not None:
                return all_indices
            return list(range(len(self.sentences_data)))
        
        # Convert to sorted list
        filtered_indices = sorted(result_indices)
        
        # If we have a specific set of indices to filter from, intersect with those
        if all_indices is not None:
            all_indices_set = set(all_indices)
            filtered_indices = [idx for idx in filtered_indices if idx in all_indices_set]
        
        self.logger.debug(f"Keyword filtering: {len(normalized_query_keywords)} query keywords, "
                         f"{len(matching_indices_sets)} matched, {len(filtered_indices)} sentences after filtering")
        
        return filtered_indices
    
    def _prepare_sentence_metadata_bulk(self):
        """Add reusable metadata (tokens, lowercase text) to all sentences once at load time."""
        if not self.sentences_data:
            return
        
        for sentence_item in self.sentences_data:
            self._prepare_sentence_metadata(sentence_item)
    
    def _prepare_sentence_metadata(self, sentence_item: Dict) -> Dict:
        """Ensure a sentence item contains cached lowercase + tokenized values."""
        if not sentence_item:
            return sentence_item
        
        sentence_text = sentence_item.get('sentence', '')
        if not sentence_text:
            return sentence_item
        
        if 'sentence_lower' not in sentence_item:
            sentence_item['sentence_lower'] = sentence_text.lower()
        if 'sentence_tokens' not in sentence_item:
            sentence_item['sentence_tokens'] = frozenset(sentence_item['sentence_lower'].split())
        
        return sentence_item
    
    def _generate_query_variants(self, question: str) -> List[str]:
        """Wrapper around helper to simplify testing."""
        return build_query_variants(question)
    
    def _enrich_query_with_context(self, question: str, context: str) -> str:
        """
        Enrich short or vague queries with keywords from conversation context.
        
        This is especially useful for follow-up queries like "explain" that need
        context from previous questions to find relevant documents.
        
        Args:
            question: The current user query
            context: Conversation context from previous messages
            
        Returns:
            Enriched query string
        """
        if not context or not context.strip():
            return question
        
        # Check if query is too short or vague (likely a follow-up)
        question_lower = question.lower().strip()
        question_words = question_lower.split()
        
        # Consider queries as "short/vague" if:
        # 1. Single word queries (like "explain", "describe", "tell me")
        # 2. Very short queries (2-3 words) that are definitional/instructional
        is_short_vague = (
            len(question_words) <= 2 and 
            (is_definitional_question(question_lower) or 
             question_lower in ["explain", "describe", "tell me", "what", "how", "why", "list", "show"])
        )
        
        if not is_short_vague:
            return question
        
        # Extract keywords from context
        # Look for the most recent user query and assistant response in the context
        context_lines = context.split('\n')
        previous_user_query = None
        previous_assistant_response = None
        
        # Find the last user message and assistant response in context
        for line in reversed(context_lines):
            if line.startswith('user:') and not previous_user_query:
                previous_user_query = line.replace('user:', '').strip()
            elif line.startswith('assistant:') and not previous_assistant_response:
                previous_assistant_response = line.replace('assistant:', '').strip()
            # Stop if we found both
            if previous_user_query and previous_assistant_response:
                break
        
        # Prioritize keywords from the previous user query, but also consider assistant response
        if previous_user_query:
            # Extract keywords from the previous user query (most relevant)
            context_keywords = extract_keywords_from_query(previous_user_query, min_length=3, max_keywords=5)
            # If we got few keywords, supplement with assistant response keywords
            if len(context_keywords) < 3 and previous_assistant_response:
                assistant_keywords = extract_keywords_from_query(previous_assistant_response, min_length=3, max_keywords=3)
                # Add unique keywords from assistant response
                for kw in assistant_keywords:
                    if kw not in context_keywords:
                        context_keywords.append(kw)
                        if len(context_keywords) >= 5:
                            break
        elif previous_assistant_response:
            # If no user query found, extract from assistant response
            context_keywords = extract_keywords_from_query(previous_assistant_response, min_length=3, max_keywords=5)
        else:
            # Fallback: extract keywords from the entire context
            context_keywords = extract_keywords_from_query(context, min_length=3, max_keywords=5)
        
        if not context_keywords:
            return question
        
        # Enrich the query by appending context keywords
        enriched_query = f"{question} {' '.join(context_keywords)}"
        
        self.logger.info(f"Enriched short query '{question}' with context keywords: {context_keywords}")
        self.logger.debug(f"Original query: '{question}' -> Enriched: '{enriched_query}'")
        
        return enriched_query
    
    def _extract_keywords_cached(self, query: str) -> List[str]:
        """
        Extract keywords from query with caching for performance.
        
        Args:
            query: The user's query string
        
        Returns:
            List of normalized keywords
        """
        if not query:
            return []
        
        # Normalize query for cache key (lowercase, strip)
        cache_key = query.strip().lower()
        
        # Check cache
        if cache_key in self.keyword_extraction_cache:
            # Move to end (most recently used)
            keywords = self.keyword_extraction_cache.pop(cache_key)
            self.keyword_extraction_cache[cache_key] = keywords
            return keywords
        
        # Extract keywords
        keywords = extract_keywords_from_query(query)
        
        # Add to cache
        if len(self.keyword_extraction_cache) >= self.keyword_cache_max_size:
            # Remove oldest entry (first item)
            self.keyword_extraction_cache.popitem(last=False)
        
        self.keyword_extraction_cache[cache_key] = keywords
        
        return keywords
    
    def _merge_retrieval_results(
        self,
        existing_results: List[Dict],
        new_results: List[Dict],
        max_results: Optional[int] = None
    ) -> List[Dict]:
        """Merge and deduplicate retrieval outputs while keeping the best scoring snippets."""
        if max_results is None:
            max_results = self.context_sentence_limit
        
        merged: Dict[Tuple[str, str], Dict] = {}
        for item in existing_results or []:
            if not item:
                continue
            key = (item.get('sentence', ''), str(item.get('document_id')))
            merged[key] = dict(item)
        
        for item in new_results or []:
            if not item:
                continue
            key = (item.get('sentence', ''), str(item.get('document_id')))
            similarity = float(item.get('similarity', 0.0) or 0.0)
            if key in merged:
                if similarity > float(merged[key].get('similarity', 0.0) or 0.0):
                    merged[key] = dict(item)
            else:
                merged[key] = dict(item)
        
        merged_list = list(merged.values())
        merged_list.sort(key=lambda x: float(x.get('similarity', 0.0) or 0.0), reverse=True)
        return merged_list[:max_results]
    
    def _deduplicate_results(self, results: List[Dict], top_k: int) -> List[Dict]:
        return self._merge_retrieval_results([], results, max_results=top_k)
    
    def _lookup_sentence_index(self, item: Dict) -> Optional[int]:
        """Find the index of a sentence inside the in-memory dataset for neighbor lookup."""
        if not item:
            return None
        
        if 'original_index' in item and isinstance(item['original_index'], int):
            return item['original_index']
        
        if not self.sentences_data:
            try:
                self.load_sentences()
            except Exception:
                return None
            if not self.document_indices:
                self._build_document_indices()
        
        doc_id = item.get('document_id')
        if doc_id and self.document_indices:
            candidate_indices = self.document_indices.get(doc_id)
            if candidate_indices:
                chunk_index = item.get('chunk_index')
                sentence_id = item.get('id')
                for idx in candidate_indices:
                    sentence_item = self.sentences_data[idx]
                    if chunk_index is not None and sentence_item.get('chunk_index') == chunk_index:
                        return idx
                    if sentence_id and sentence_item.get('id') == sentence_id:
                        return idx
        
        sentence_text = item.get('sentence')
        if sentence_text and self.sentences_data:
            for idx, sentence_item in enumerate(self.sentences_data):
                if sentence_item.get('sentence') == sentence_text:
                    return idx
        
        return None
    
    def _get_neighbor_sentences(self, item: Dict, window: int) -> List[Dict]:
        """Collect neighboring sentences around a retrieved snippet for richer context."""
        if window <= 0:
            return []
        
        sentence_index = self._lookup_sentence_index(item)
        if sentence_index is None or not self.sentences_data:
            return []
        
        neighbors: List[Dict] = []
        start = max(0, sentence_index - window)
        end = min(len(self.sentences_data), sentence_index + window + 1)
        base_similarity = float(item.get('similarity', 0.0) or 0.0)
        
        for idx in range(start, end):
            if idx == sentence_index:
                continue
            neighbor_source = self.sentences_data[idx]
            neighbor_entry = {
                'sentence': neighbor_source.get('sentence', ''),
                'similarity': max(base_similarity - 0.01 * abs(idx - sentence_index), 0.0),
                'id': neighbor_source.get('id', ''),
                'document_id': neighbor_source.get('document_id', ''),
                'document_name': neighbor_source.get('document_name', 'Unknown Document'),
                'chunk_index': neighbor_source.get('chunk_index', 0),
                'drive_link': neighbor_source.get('drive_link', ''),
                'original_index': idx,
                'neighbor_source': item.get('id') or item.get('sentence', '')[:40]
            }
            neighbors.append(neighbor_entry)
        
        return neighbors
    
    def _augment_with_neighbor_context(self, results: List[Dict]) -> List[Dict]:
        """Add nearby sentences to help the model infer unstated consequences."""
        if not results or not self.sentences_data or self.neighbor_sentence_window <= 0:
            return results
        
        neighbor_payloads: List[Dict] = []
        for item in results:
            neighbor_payloads.extend(self._get_neighbor_sentences(item, self.neighbor_sentence_window))
        
        if not neighbor_payloads:
            return results
        
        max_results = max(self.context_sentence_limit, len(results) + len(neighbor_payloads))
        return self._merge_retrieval_results(results, neighbor_payloads, max_results=max_results)
    
    def _build_reasoning_hint(self, question: str, document_context: str) -> str:
        """Provide additional guidance when the question requires inferred penalties."""
        if not question or not document_context:
            return ""
        
        if not is_consequence_question(question):
            return ""
        
        context_lower = document_context.lower()
        if any(keyword in context_lower for keyword in ['penalty', 'penalties', 'fine', 'suspend', 'cancellation', 'sanction']):
            return ""
        
        return (
            "The user is asking about regulatory consequences. Use the cited obligations to explain why "
            "compliance is mandatory, clearly state that the documents do not list explicit penalties, and "
            "describe the typical Bangladesh Bank enforcement steps (warnings, show-cause notices, fines, or "
            "license suspension) that would reasonably follow non-compliance."
        )
    
    def _retrieve_with_variants(self, question: str, profiler: Optional[Dict]) -> List[Dict]:
        """Retrieve context using query expansion to capture implicit compliance scenarios."""
        # Limit query variants for performance (1-2 max)
        if not self.enable_query_variants:
            # Skip variant generation entirely if disabled
            return self.search_relevant_content(question, top_k=self.context_sentence_limit, profiler=profiler)
        
        variants = self._generate_query_variants(question)
        # Limit to max_query_variants (default 2) for performance
        variants = variants[:self.max_query_variants]
        
        aggregated: List[Dict] = []
        target_limit = max(self.context_sentence_limit, 5)
        
        # Parallelize variant searches for better performance
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(variants), 3)) as executor:
            future_to_variant = {
                executor.submit(self.search_relevant_content, variant, target_limit, profiler): variant
                for variant in variants
            }
            
            for future in concurrent.futures.as_completed(future_to_variant):
                variant = future_to_variant[future]
                try:
                    variant_results = future.result()
                    aggregated = self._merge_retrieval_results(aggregated, variant_results, max_results=target_limit)
                    if len(aggregated) >= self.context_sentence_limit:
                        # Cancel remaining futures if we have enough results
                        for f in future_to_variant:
                            f.cancel()
                        break
                except Exception as e:
                    self.logger.warning(f"Error searching variant '{variant}': {e}")
        
        return aggregated
    
    @staticmethod
    def _convert_embedding_to_array(embedding_obj) -> Optional[np.ndarray]:
        """Normalize embedding objects from SDKs into numpy arrays."""
        if embedding_obj is None:
            return None
        data = getattr(embedding_obj, "values", embedding_obj)
        if data is None:
            return None
        return np.asarray(data, dtype=np.float32)
    
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity with numerical safety checks."""
        if vec1 is None or vec2 is None:
            return 0.0
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    @staticmethod
    def _compute_vectorized_similarities(query_embedding: np.ndarray, sentence_vectors: List[np.ndarray]) -> np.ndarray:
        """Compute cosine similarities between query and multiple sentence vectors using vectorized operations.
        
        Args:
            query_embedding: Query embedding vector (1D array)
            sentence_vectors: List of sentence embedding vectors (each is 1D array)
        
        Returns:
            Array of similarity scores (same length as sentence_vectors)
        """
        if not sentence_vectors or query_embedding is None:
            return np.array([])
        
        # Convert to numpy array: shape (n_sentences, embedding_dim)
        vectors_matrix = np.array(sentence_vectors, dtype=np.float32)
        
        # Compute norms once
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return np.zeros(len(sentence_vectors))
        
        vector_norms = np.linalg.norm(vectors_matrix, axis=1)
        
        # Compute all dot products at once: (n_sentences,)
        dot_products = np.dot(vectors_matrix, query_embedding)
        
        # Compute all similarities at once with safety checks (avoid division by zero)
        norms_product = query_norm * vector_norms
        similarities = np.divide(
            dot_products, 
            norms_product, 
            out=np.zeros_like(dot_products, dtype=np.float32), 
            where=norms_product != 0
        )
        
        return similarities
    
    @staticmethod
    def _strip_discourse_marker(text: str) -> str:
        """Remove leading discourse markers like 'However,' from a text segment."""
        if not text:
            return text
        markers = ("however", "but", "nevertheless", "nonetheless", "still", "yet")
        stripped = text.lstrip()
        lowered = stripped.lower()
        for marker in markers:
            if lowered.startswith(marker):
                remainder = stripped[len(marker):].lstrip(" ,:-")
                if not remainder:
                    return text
                return remainder[0].upper() + remainder[1:]
        return text
    
    def _normalize_bullet_discourse(self, line: str) -> str:
        """Normalize bullet/numbered list lines to avoid awkward discourse markers."""
        if not line:
            return line
        bullet_prefix = ""
        remainder = line
        bullet_prefixes = ("- ", "* ", "• ")
        for prefix in bullet_prefixes:
            if remainder.startswith(prefix):
                bullet_prefix = prefix
                remainder = remainder[len(prefix):]
                break
        else:
            match = re.match(r"^(\d+\.\s+)", remainder)
            if match:
                bullet_prefix = match.group(1)
                remainder = remainder[len(bullet_prefix):]
        cleaned_remainder = self._strip_discourse_marker(remainder)
        return (bullet_prefix + cleaned_remainder).strip()
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Fetch an embedding from cache or Vertex AI with LRU eviction."""
        if not (VERTEX_AI_AVAILABLE and self.embedding_model and text):
            return None
        
        cached = self.embeddings_cache.get(text)
        if cached is not None:
            # Move to end to mark as recently used
            self.embeddings_cache.move_to_end(text)
            return cached
        
        embedding_obj = self.embedding_model.get_embeddings([text])[0]
        embedding = self._convert_embedding_to_array(embedding_obj)
        if embedding is None or embedding.size == 0:
            return None
        self.embeddings_cache[text] = embedding
        self.embeddings_cache.move_to_end(text)
        
        if len(self.embeddings_cache) > self.embedding_cache_max_size:
            self.embeddings_cache.popitem(last=False)
        
        return embedding
    
    def _normalize_query_for_cache(self, query: str) -> str:
        """Normalize query for cache key: lowercase, remove punctuation, trim."""
        import string
        normalized = query.strip().lower()
        # Remove punctuation but keep spaces
        normalized = normalized.translate(str.maketrans('', '', string.punctuation))
        # Normalize whitespace
        normalized = ' '.join(normalized.split())
        return normalized
    
    def _adjust_generation_config(self, base_config: Dict, prompt_length: int) -> Dict:
        """Dynamically cap max tokens based on prompt length to avoid over-generation."""
        config = dict(base_config or {})
        estimated_tokens = max(256, int(prompt_length / max(self.prompt_chars_per_token, 1)))
        base_limit = int(config.get("max_output_tokens", 8192) or 8192)
        # Don't reduce tokens below 80% of base limit to ensure complete responses
        min_allowed = int(base_limit * 0.8)
        dynamic_limit = int(min(base_limit, estimated_tokens * self.dynamic_token_factor))
        config["max_output_tokens"] = max(min_allowed, dynamic_limit)
        return config
    
    def _generation_model_candidates(self, prompt_length: int) -> List[Tuple[str, Optional['GenerativeModel'], Dict]]:
        """Return model candidates ordered by speed (fastest first) for performance optimization."""
        candidates: List[Tuple[str, Optional[GenerativeModel], Dict]] = []
        
        # Fast model first (always try if available)
        if self.fast_generative_model:
            # In fast-only mode, use fast model for all prompts; otherwise use threshold
            if self.fast_model_only or prompt_length <= self.fast_model_char_limit * 2:
                candidates.append(("fast", self.fast_generative_model, self._adjust_generation_config(self.fast_generation_config, prompt_length)))
        
        # Default model only if not in fast-only mode
        if self.generative_model and not self.fast_model_only:
            candidates.append(("default", self.generative_model, self._adjust_generation_config(self.default_generation_config, prompt_length)))
        
        # Deduplicate identical model instances while preserving order
        unique_candidates: List[Tuple[str, Optional[GenerativeModel], Dict]] = []
        seen_ids = set()
        for label, model, config in candidates:
            if not model:
                continue
            model_id = id(model)
            if model_id in seen_ids:
                continue
            seen_ids.add(model_id)
            unique_candidates.append((label, model, config))
        
        return unique_candidates
    
    def _extract_text_from_vertex_response(self, response) -> str:
        """Safely extract text content from Vertex AI responses or stream chunks."""
        if response is None:
            return ""
        
        # Try direct text attribute first
        text_attr = getattr(response, "text", None)
        if isinstance(text_attr, str) and text_attr:
            return text_attr
        
        # Handle TextContent objects (from newer Vertex AI SDK)
        if hasattr(response, 'type') and hasattr(response, 'text'):
            if response.type == 'text' and isinstance(response.text, str):
                text_value = response.text
                # Log if TextContent has empty text (might indicate safety filter or error)
                if not text_value:
                    self.logger.warning(f"TextContent object has empty text field. "
                                      f"Response: {response}, Has annotations: {hasattr(response, 'annotations')}")
                return text_value
        
        # Try to get text from parts directly
        parts = getattr(response, "parts", None)
        if parts:
            extracted_parts = []
            for part in parts:
                # Handle TextContent objects in parts
                if hasattr(part, 'type') and hasattr(part, 'text'):
                    if part.type == 'text' and isinstance(part.text, str):
                        extracted_parts.append(part.text)
                # Handle regular text attribute
                elif hasattr(part, 'text'):
                    part_text = getattr(part, "text", None)
                    if isinstance(part_text, str) and part_text:
                        extracted_parts.append(part_text)
            if extracted_parts:
                return "".join(extracted_parts)
        
        # Try candidates structure (standard Vertex AI response format)
        candidates = getattr(response, "candidates", None)
        extracted_parts: List[str] = []
        if candidates:
            for candidate in candidates:
                content = getattr(candidate, "content", None)
                if not content:
                    continue
                
                # Try parts from content
                parts = getattr(content, "parts", None)
                if parts:
                    for part in parts:
                        # Handle TextContent objects
                        if hasattr(part, 'type') and hasattr(part, 'text'):
                            if part.type == 'text' and isinstance(part.text, str):
                                extracted_parts.append(part.text)
                        # Handle regular text attribute
                        elif hasattr(part, 'text'):
                            part_text = getattr(part, "text", None)
                            if isinstance(part_text, str) and part_text:
                                extracted_parts.append(part_text)
        
        result = "".join(extracted_parts)
        
        # Log warning if we couldn't extract text (for debugging)
        if not result:
            self.logger.warning(f"Could not extract text from Vertex AI response. Response type: {type(response)}, "
                              f"attributes: {dir(response)[:10] if hasattr(response, '__dict__') else 'N/A'}")
        
        return result
    
    def _invoke_vertex_model(
        self,
        model: 'GenerativeModel',
        config: Dict,
        prompt_text: str,
        profiler: Optional[Dict],
        label: str,
        stop_event: Optional[threading.Event] = None,
        retry_on_truncation: bool = True,
    ) -> Optional[str]:
        """Call a Vertex AI generative model with optional streaming support.
        
        Args:
            retry_on_truncation: If True, automatically retry with higher token limit if truncation is detected.
        """
        if not model:
            return None
        if stop_event and stop_event.is_set():
            return None
        
        start_time = time.perf_counter()
        result_text = ""
        finish_reason = None
        is_truncated = False
        
        if self.use_vertex_streaming:
            try:
                with self._stage(profiler, f"generation.vertex_stream_{label}"):
                    stream = model.generate_content(
                        prompt_text,
                        generation_config=config,
                        stream=True
                    )
                    response_chunks: List[str] = []
                    first_token_latency = None
                    chunk_count = 0
                    finish_reason = None
                    try:
                        for chunk in stream:
                            if stop_event and stop_event.is_set():
                                self.logger.debug(f"Streaming generation cancelled for model '{label}'")
                                try:
                                    if hasattr(stream, "close"):
                                        stream.close()
                                except Exception:
                                    pass
                                break
                            chunk_count += 1
                            # Check for finish reason in chunk (for truncation detection)
                            if hasattr(chunk, 'candidates') and chunk.candidates:
                                for candidate in chunk.candidates:
                                    if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                                        finish_reason = candidate.finish_reason
                                        if finish_reason == 'MAX_TOKENS':
                                            self.logger.warning(
                                                f"Response truncated due to MAX_TOKENS limit. "
                                                f"Consider increasing max_output_tokens from {config.get('max_output_tokens', 'unknown')}"
                                            )
                            
                            chunk_text = self._extract_text_from_vertex_response(chunk)
                            if chunk_text:
                                if first_token_latency is None:
                                    first_token_latency = time.perf_counter() - start_time
                                    if self.streaming_first_token_log:
                                        self.logger.info(
                                            f"Vertex {label} first token latency: {first_token_latency * 1000:.1f}ms"
                                        )
                                response_chunks.append(chunk_text)
                                # Optimize chunk processing: accumulate efficiently
                                # No need for immediate processing, just collect chunks
                        
                        # Log if we got chunks but no text (might indicate safety filters)
                        if chunk_count > 0 and not response_chunks:
                            self.logger.warning(f"Received {chunk_count} chunks but extracted no text. "
                                              f"This might indicate safety filters are blocking content.")
                        result_text = "".join(response_chunks).strip()
                        
                        # Log warning if streaming returned empty result
                        if not result_text:
                            self.logger.warning(f"Streaming returned empty result for model '{label}'. "
                                              f"Chunks collected: {len(response_chunks)}, Finish reason: {finish_reason}")
                        
                        # Check for incomplete responses that end with just headers
                        if result_text:
                            result_lower = result_text.lower().strip()
                            # Check if response is just a header with no content
                            incomplete_headers = ['key observations:', 'observations:', 'summary:', 'findings:']
                            if result_lower in incomplete_headers:
                                self.logger.warning(
                                    f"Streaming returned incomplete response (only header): '{result_text}'. "
                                    f"Finish reason: {finish_reason}, Chunks: {len(response_chunks)}"
                                )
                            # Check if response ends with a header but has no content after
                            elif any(result_lower.endswith(header) for header in incomplete_headers):
                                # Check if there's actual content before the header
                                for header in incomplete_headers:
                                    if result_lower.endswith(header):
                                        content_before = result_lower[:-len(header)].strip()
                                        if not content_before or len(content_before) < 10:
                                            self.logger.warning(
                                                f"Streaming returned response ending with header '{header}' but minimal content. "
                                                f"Response: '{result_text[:200]}...', Finish reason: {finish_reason}"
                                            )
                                        break
                        
                        # Check if response seems truncated (ends without punctuation or is incomplete)
                        if result_text:
                            # Check for truncation indicators
                            ends_with_punctuation = result_text[-1] in '.!?;'
                            # Also check if it ends mid-word or mid-sentence (common truncation patterns)
                            ends_abruptly = (
                                not ends_with_punctuation and 
                                len(result_text) > 50 and  # Only check for longer responses
                                not result_text.rstrip().endswith((':', ',', '-', ')', ']', '}'))
                            )
                            if ends_abruptly or finish_reason == 'MAX_TOKENS':
                                is_truncated = True
                                self.logger.warning(
                                    f"Response may be truncated (ends with '{result_text[-30:]}', "
                                    f"finish_reason: {finish_reason}). "
                                    f"Current max_output_tokens: {config.get('max_output_tokens', 'unknown')}. "
                                    f"Will retry with higher limit if enabled."
                                )
                    except Exception as stream_iter_error:
                        self.logger.warning(f"Error iterating stream: {stream_iter_error}")
                        import traceback
                        self.logger.debug(f"Stream iteration error traceback: {traceback.format_exc()}")
                        # Use what we have so far
                        result_text = "".join(response_chunks).strip()
                        if not result_text:
                            self.logger.warning(f"No text collected from stream before error")
            except Exception as stream_error:
                self.logger.warning(f"Streaming generation failed for model '{label}': {stream_error}")
                result_text = ""
        
        # Check for truncation in non-streaming fallback
        if not result_text:
            # Fallback to non-streaming if streaming failed or was disabled
            try:
                with self._stage(profiler, f"generation.vertex_call_{label}"):
                    if stop_event and stop_event.is_set():
                        return None
                    response = model.generate_content(
                        prompt_text,
                        generation_config=config
                    )
                
                # Extract text with improved method
                extracted_text = self._extract_text_from_vertex_response(response)
                result_text = extracted_text.strip() if extracted_text else None
                
                # Check for finish reasons in non-streaming response
                if response:
                    candidates = getattr(response, "candidates", None)
                    if candidates:
                        for candidate in candidates:
                            candidate_finish_reason = getattr(candidate, "finish_reason", None)
                            if candidate_finish_reason == 'MAX_TOKENS':
                                finish_reason = 'MAX_TOKENS'
                                is_truncated = True
                                self.logger.warning(
                                    f"Non-streaming response truncated due to MAX_TOKENS limit. "
                                    f"Current max_output_tokens: {config.get('max_output_tokens', 'unknown')}"
                                )
                
                # If we still got empty text, check for safety filters or finish reasons
                if not result_text:
                    self.logger.error(f"Vertex AI model '{label}' returned empty response. "
                                    f"Prompt length: {len(prompt_text)}, Config: {config}")
                    
                    # Check for finish reasons or safety filters that might explain empty response
                    if response:
                        # Check candidates for finish reasons
                        candidates = getattr(response, "candidates", None)
                        if candidates:
                            for i, candidate in enumerate(candidates):
                                finish_reason = getattr(candidate, "finish_reason", None)
                                safety_ratings = getattr(candidate, "safety_ratings", None)
                                if finish_reason:
                                    self.logger.warning(f"Candidate {i} finish_reason: {finish_reason}")
                                if safety_ratings:
                                    self.logger.warning(f"Candidate {i} safety_ratings: {safety_ratings}")
                        
                        # Try to inspect response structure for debugging
                        self.logger.debug(f"Response type: {type(response)}, "
                                        f"Has text attr: {hasattr(response, 'text')}, "
                                        f"Has candidates: {hasattr(response, 'candidates')}, "
                                        f"Has parts: {hasattr(response, 'parts')}")
                        
                        # Check if response has any content at all
                        if hasattr(response, '__dict__'):
                            self.logger.debug(f"Response attributes: {list(response.__dict__.keys())[:10]}")
                    
                    return None
            except Exception as call_error:
                self.logger.error(f"Vertex generation error ({label}): {call_error}")
                import traceback
                self.logger.error(f"Full traceback: {traceback.format_exc()}")
                return None
        
        # If truncation detected and retry is enabled, retry with higher token limit
        if is_truncated and retry_on_truncation and result_text:
            current_max_tokens = config.get('max_output_tokens', 8192)
            # Increase token limit for retry, cap at 8192 (maximum Vertex AI limit)
            # If already at max, try with max again (in case of other issues)
            new_max_tokens = min(current_max_tokens * 2, 8192) if current_max_tokens < 8192 else 8192
            
            if new_max_tokens > current_max_tokens:
                self.logger.info(
                    f"Retrying generation with increased max_output_tokens: {current_max_tokens} -> {new_max_tokens} "
                    f"to complete truncated response"
                )
                retry_config = config.copy()
                retry_config['max_output_tokens'] = new_max_tokens
                
                # Retry with higher limit (disable retry on retry to avoid infinite loops)
                retry_result = self._invoke_vertex_model(
                    model, retry_config, prompt_text, profiler, label, stop_event, retry_on_truncation=False
                )
                
                if retry_result and len(retry_result) > len(result_text):
                    self.logger.info(f"Retry successful: got {len(retry_result)} chars vs {len(result_text)} chars originally")
                    return retry_result
                else:
                    self.logger.warning(f"Retry did not produce longer response, returning original")
        
        return result_text
    
    def _generate_vertex_response(self, prompt_text: str, profiler: Optional[Dict]) -> Tuple[Optional[str], str]:
        """Generate a response using available Vertex AI models with fallback ordering."""
        prompt_length = len(prompt_text)
        candidates = self._generation_model_candidates(prompt_length)
        if len(candidates) >= 2 and candidates[0][0] == "fast":
            fast_candidate = candidates[0]
            default_candidate = candidates[1]
            result, label = self._generate_with_fast_path(fast_candidate, default_candidate, prompt_text, profiler)
            if result:
                return result, label
            candidates = candidates[2:]
        for label, model, config in candidates:
            result = self._invoke_vertex_model(model, config, prompt_text, profiler, label)
            if result:
                return result, label
        
        return None, "unavailable"
    
    def _generate_with_fast_path(
        self,
        fast_candidate: Tuple[str, Optional['GenerativeModel'], Dict],
        default_candidate: Tuple[str, Optional['GenerativeModel'], Dict],
        prompt_text: str,
        profiler: Optional[Dict],
    ) -> Tuple[Optional[str], str]:
        """Run fast model first with timeout, then race remaining candidates."""
        fast_label, fast_model, fast_config = fast_candidate
        default_label, default_model, default_config = default_candidate
        if not fast_model:
            return None, "unavailable"
        
        # In fast_model_only mode, use only the fast model without escalation
        if self.fast_model_only:
            result = self._invoke_vertex_model(
                fast_model,
                fast_config,
                prompt_text,
                profiler,
                fast_label,
                None,
            )
            if result:
                return result, fast_label
            return None, "unavailable"
        
        # Original logic for when both models are available
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures: Dict[concurrent.futures.Future, Tuple[str, threading.Event]] = {}
            fast_event = threading.Event()
            fast_future = executor.submit(
                self._invoke_vertex_model,
                fast_model,
                fast_config,
                prompt_text,
                profiler,
                fast_label,
                fast_event,
            )
            futures[fast_future] = (fast_label, fast_event)
            try:
                fast_result = fast_future.result(timeout=self.fast_response_timeout)
                if fast_result:
                    return fast_result, fast_label
                self.logger.debug("Fast model returned empty result; escalating to default model")
            except concurrent.futures.TimeoutError:
                self.logger.debug(f"Fast model timeout after {self.fast_response_timeout}s; escalating to default model.")
            except Exception as fast_error:
                self.logger.warning(f"Fast model error: {fast_error}")
            
            if default_model:
                default_event = threading.Event()
                default_future = executor.submit(
                    self._invoke_vertex_model,
                    default_model,
                    default_config,
                    prompt_text,
                    profiler,
                    default_label,
                    default_event,
                )
                futures[default_future] = (default_label, default_event)
            else:
                default_future = None
            
            pending_futures = list(futures.keys())
            while pending_futures:
                done, pending = concurrent.futures.wait(pending_futures, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    label, _ = futures[future]
                    try:
                        result = future.result()
                    except Exception as future_error:
                        self.logger.warning(f"Generation future error ({label}): {future_error}")
                        result = None
                    if result:
                        for other in pending:
                            other_label, other_event = futures[other]
                            other_event.set()
                            other.cancel()
                        return result, label
                pending_futures = list(pending)
        
        return None, "unavailable"
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using Vertex AI embeddings or fallback method."""
        try:
            # Use Vertex AI embeddings if available
            if VERTEX_AI_AVAILABLE and self.embedding_model:
                emb1 = self._get_embedding(text1)
                emb2 = self._get_embedding(text2)
                
                if emb1 is None or emb2 is None:
                    self.logger.debug("Embedding fetch returned None; falling back to keyword similarity")
                    raise ValueError("Embedding fetch failed")
                
                # Calculate cosine similarity
                dot_product = np.dot(emb1, emb2)
                norm1 = np.linalg.norm(emb1)
                norm2 = np.linalg.norm(emb2)
                
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                
                similarity = dot_product / (norm1 * norm2)
                return float(similarity)
            else:
                # Fallback to simple string similarity (basic implementation)
                # This is a very basic fallback - in a real implementation, you might want to use
                # a more sophisticated local embedding model
                words1 = set(text1.lower().split())
                words2 = set(text2.lower().split())
                intersection = words1.intersection(words2)
                union = words1.union(words2)
                if len(union) == 0:
                    return 0.0
                return len(intersection) / len(union)
            
        except Exception as e:
            self.logger.warning(f"Similarity calculation error: {e}")
            return 0.0
    
    def search_similar_documents_faiss(self, query: str, top_k: int = 3, profiler: Optional[Dict] = None) -> List[Dict]:
        """Search for similar documents using FAISS similarity search as primary method."""
        self.logger.info(f"Searching for similar documents using FAISS primary method: '{query}'")
        self.logger.info("FAISS search method: Using precomputed FAISS index for fast similarity search")
        
        try:
            if not self.faiss_vector_store:
                self.logger.error("FAISS vector store not loaded")
                return []
            
            # Extract keywords from query for filtering (with caching)
            query_keywords = []
            if self.keyword_filter_enabled:
                with self._stage(profiler, "retrieval.extract_keywords"):
                    query_keywords = self._extract_keywords_cached(query)
                    if query_keywords:
                        self.logger.debug(f"Extracted {len(query_keywords)} keywords from query: {query_keywords}")
            
            # Perform similarity search (search more results if we'll filter)
            # Reduced multiplier from 3x to 2x for faster responses
            search_k = top_k * 2 if query_keywords and self.keyword_filter_enabled else top_k
            with self._stage(profiler, "retrieval.faiss_similarity_search"):
                search_results = self.faiss_vector_store.similarity_search_with_score(query, k=search_k)
            
            # Format results and get sentence IDs for keyword filtering
            results = []
            result_ids = []
            for doc, score in search_results:
                result_id = doc.metadata.get('id', '')
                results.append({
                    'sentence': doc.page_content,
                    'similarity': float(1 - score),  # Convert distance to similarity
                    'id': result_id,
                    'document_id': doc.metadata.get('document_id', ''),
                    'document_name': doc.metadata.get('document_name', 'Unknown Document'),
                    'chunk_index': doc.metadata.get('chunk_index', 0),
                    'drive_link': doc.metadata.get('drive_link', ''),
                    'folder_name': doc.metadata.get('folder_name', '')
                })
                result_ids.append(result_id)
            
            # Apply folder filtering if active (supports multiple folders)
            if self.active_folder_filters:
                self.logger.info(f"📁 FOLDER FILTER ACTIVE: Filtering by folders: {self.active_folder_filters}")
                with self._stage(profiler, "retrieval.folder_filter_faiss"):
                    filtered_results = []
                    missing_folder_count = 0
                    folder_distribution = {}
                    
                    for result in results:
                        folder_name = result.get('folder_name', '')
                        # Track folder distribution for debugging
                        folder_distribution[folder_name] = folder_distribution.get(folder_name, 0) + 1
                        
                        if not folder_name:
                            missing_folder_count += 1
                        # Match if folder is in any of the active filters
                        if folder_name and folder_name in self.active_folder_filters:
                            filtered_results.append(result)
                    
                    # Log folder distribution for debugging
                    self.logger.info(f"📊 Folder distribution in search results: {folder_distribution}")
                    
                    # Log if folder_name metadata is missing
                    if missing_folder_count > 0:
                        self.logger.warning(
                            f"Found {missing_folder_count} documents without folder_name metadata. "
                            f"FAISS index may need to be rebuilt with folder_name field."
                        )
                    
                    # Strictly apply filter - only use filtered results
                    if filtered_results:
                        results = filtered_results
                        self.logger.info(f"✅ Folder filter SUCCESS: {self.active_folder_filters} reduced results from {len(search_results)} to {len(results)}")
                    else:
                        # No results match the filter - return empty list
                        results = []
                        self.logger.warning(
                            f"❌ Folder filter returned ZERO results. Active filters: {self.active_folder_filters}. "
                            f"Available folders in results: {list(folder_distribution.keys())}. "
                            f"Returning empty results to enforce strict filtering."
                        )
            
            # Apply keyword filtering if enabled and keywords were extracted
            if query_keywords and self.keyword_filter_enabled and self.sentences_data and self.id_to_index:
                with self._stage(profiler, "retrieval.keyword_filter_faiss"):
                    # Use O(1) lookup instead of O(n) linear search
                    result_indices = []
                    for result_id in result_ids:
                        idx = self.id_to_index.get(str(result_id))
                        if idx is not None:
                            result_indices.append(idx)
                    
                    # Skip keyword filtering if we have very few results (already fast)
                    # Skip for small result sets to improve performance
                    if len(result_indices) <= top_k:
                        self.logger.debug(f"Skipping keyword filtering for {len(result_indices)} results (already small)")
                    else:
                        # Filter by keywords using O(1) lookups
                        filtered_indices = self._filter_by_keywords(query_keywords, result_indices)
                        filtered_indices_set = set(filtered_indices)
                        
                        # Filter results to only include those that passed keyword filter
                        filtered_results = []
                        for idx, result in enumerate(results):
                            if result_indices[idx] in filtered_indices_set:
                                filtered_results.append(result)
                        
                        # If filtering removed too many results, keep top results by similarity
                        if len(filtered_results) < top_k and len(results) > 0:
                            self.logger.debug(f"Keyword filtering reduced results from {len(results)} to {len(filtered_results)}, "
                                             f"keeping top {top_k} by similarity")
                            # Sort by similarity and take top_k
                            filtered_results = sorted(filtered_results, key=lambda x: x['similarity'], reverse=True)[:top_k]
                        
                        results = filtered_results
                        self.logger.info(f"After keyword filtering: {len(results)} results from FAISS")
            
            # Ensure we return at most top_k results
            results = results[:top_k]
            
            self.logger.info(f"Found {len(results)} similar documents using FAISS")
            for i, result in enumerate(results):
                self.logger.debug(f"{i+1}. ({result['similarity']:.3f}) {result.get('sentence', '')[:100]}...")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching documents with FAISS: {e}")
            return []
    
    def search_relevant_content(self, query: str, top_k: int = 3, profiler: Optional[Dict] = None) -> List[Dict]:
        """Search for relevant content - optimized for sub-second response times."""
        """Search for relevant content based on the query with performance optimizations."""
        self.logger.info(f"Searching for content related to: '{query}'")
        
        # Try Vertex AI embedding-based similarity search first (primary method)
        self.logger.info("Using Vertex AI as primary search method")
        
        if not self.sentences_data:
            with self._stage(profiler, "retrieval.load_sentences"):
                if not self.load_sentences():
                    self.logger.error("No data available for search")
                    return []
        
        self.logger.info("Vertex AI search method: Using embedding-based similarity search")
        self.logger.info("Data source: Loaded from pickle cache or GCS")
        
        # Extract keywords from query for filtering (with caching)
        query_keywords = []
        if self.keyword_filter_enabled:
            with self._stage(profiler, "retrieval.extract_keywords"):
                query_keywords = self._extract_keywords_cached(query)
                if query_keywords:
                    self.logger.debug(f"Extracted {len(query_keywords)} keywords from query: {query_keywords}")
        
        # Apply keyword filtering before embedding calculation (performance optimization)
        with self._stage(profiler, "retrieval.keyword_filter"):
            if query_keywords and self.keyword_filter_enabled:
                # Filter sentences by keywords before expensive embedding calculation
                all_indices = list(range(len(self.sentences_data)))
                filtered_indices = self._filter_by_keywords(query_keywords, all_indices)
                
                if filtered_indices and len(filtered_indices) < len(self.sentences_data):
                    # Use filtered sentences
                    filtered_sentences = [(idx, self.sentences_data[idx]) for idx in filtered_indices]
                    self.logger.info(f"Keyword filtering: reduced from {len(self.sentences_data)} to {len(filtered_sentences)} sentences "
                                   f"before embedding calculation ({(1 - len(filtered_sentences)/len(self.sentences_data))*100:.1f}% reduction)")
                else:
                    # No filtering applied or all sentences match, use all
                    filtered_sentences = list(enumerate(self.sentences_data))
                    self.logger.info(f"Processing all {len(filtered_sentences)} sentences for embedding similarity (no keyword filtering applied)")
            else:
                # No keyword filtering, process all sentences
                # This enables cross-lingual search: English queries can find Bangla content
                # via multilingual embedding model (text-embedding-004)
                filtered_sentences = list(enumerate(self.sentences_data))
                self.logger.info(f"Processing all {len(filtered_sentences)} sentences for embedding similarity (cross-lingual support enabled)")
        
        # Apply folder filtering if active (supports multiple folders) - BEFORE embedding calculation for performance
        if self.active_folder_filters:
            self.logger.info(f"📁 FOLDER FILTER ACTIVE (Vertex AI path): Filtering by folders: {self.active_folder_filters}")
            with self._stage(profiler, "retrieval.folder_filter_vertex"):
                folder_filtered_sentences = []
                missing_folder_count = 0
                folder_distribution = {}
                
                for idx, sentence_item in filtered_sentences:
                    folder_name = sentence_item.get('folder_name', '')
                    # Track folder distribution for debugging
                    folder_distribution[folder_name] = folder_distribution.get(folder_name, 0) + 1
                    
                    if not folder_name:
                        missing_folder_count += 1
                    # Match if folder is in any of the active filters
                    if folder_name and folder_name in self.active_folder_filters:
                        folder_filtered_sentences.append((idx, sentence_item))
                
                # Log folder distribution for debugging
                self.logger.info(f"📊 Folder distribution in search results: {folder_distribution}")
                
                # Log if folder_name metadata is missing
                if missing_folder_count > 0:
                    self.logger.warning(
                        f"Found {missing_folder_count} documents without folder_name metadata. "
                        f"These will be excluded from folder-filtered results."
                    )
                
                # Strictly apply filter - only use filtered results
                original_count = len(filtered_sentences)
                if folder_filtered_sentences:
                    filtered_sentences = folder_filtered_sentences
                    self.logger.info(f"Folder filter SUCCESS: {self.active_folder_filters} reduced results from {original_count} to {len(filtered_sentences)}")
                else:
                    # No results match the filter - return empty list early
                    self.logger.warning(
                        f"Folder filter returned ZERO results. Active filters: {self.active_folder_filters}. "
                        f"Available folders in results: {list(folder_distribution.keys())}. "
                        f"Returning empty results to enforce strict filtering."
                    )
                    return []
        
        if self.embedding_fallback_limit > 0 and len(filtered_sentences) > self.embedding_fallback_limit:
            self.logger.info(
                f"Embedding fallback limit active: truncating sentences from {len(filtered_sentences)} "
                f"to {self.embedding_fallback_limit} for performance"
            )
            filtered_sentences = filtered_sentences[:self.embedding_fallback_limit]
        
        with self._stage(profiler, "retrieval.prepare_all_sentences"):
            self.logger.debug(f"Prepared {len(filtered_sentences)} sentences for similarity calculation")
        
        sentence_indices = [idx for idx, _ in filtered_sentences]
        
        # Check cache hit rate before deciding whether to prefetch
        cache_hit_rate = 0.0
        missing_count = 0
        if sentence_indices and self.sentence_embedding_cache:
            self._ensure_sentence_embedding_cache_initialized()
            missing_count = sum(
                1 for idx in sentence_indices
                if 0 <= idx < len(self.sentence_embedding_cache)
                and self.sentence_embedding_cache[idx] is None
            )
            cache_hit_rate = 1.0 - (missing_count / len(sentence_indices)) if sentence_indices else 1.0
        
        # Only prefetch if cache hit rate is low (< 50%) and missing count is reasonable (< 100)
        # Otherwise, use batch embeddings path directly for better performance
        should_prefetch = (
            sentence_indices 
            and cache_hit_rate < 0.5 
            and missing_count > 0 
            and missing_count < 100
        )
        
        if should_prefetch:
            with self._stage(profiler, "retrieval.prefetch_embeddings"):
                # Limit prefetch to avoid blocking too long
                limited_indices = sentence_indices[:min(50, len(sentence_indices))]
                self._prefetch_sentence_embeddings(limited_indices)
        elif missing_count > 0:
            self.logger.debug(
                f"Skipping prefetch: cache_hit_rate={cache_hit_rate:.1%}, missing={missing_count}. "
                f"Will use batch embeddings path for missing embeddings."
            )
        
        # Get query embedding first (outside similarity timing)
        with self._stage(profiler, "retrieval.get_query_embedding"):
            query_embedding = self._get_embedding(query)
        
        # Calculate similarities for all sentences using batch embedding API
        similarities = []
        with self._stage(profiler, "retrieval.similarity_scoring"):
            self.logger.debug(f"Calculating embedding similarities for {len(filtered_sentences)} sentences...")
            
            # Try cached embeddings path first (fastest)
            if query_embedding is not None and self.sentence_embedding_cache:
                # Collect cached and missing sentence vectors (optimized single pass)
                cached_vectors = []
                cached_metadata = []
                missing_sentences = []
                missing_metadata = []
                
                # Pre-initialize to avoid repeated lookups
                cache_len = len(self.sentence_embedding_cache)
                
                for idx, sentence_item in filtered_sentences:
                    sentence_item = self._prepare_sentence_metadata(sentence_item)
                    
                    # Fast path: check cache bounds and get vector
                    if 0 <= idx < cache_len:
                        sentence_vector = self.sentence_embedding_cache[idx]
                        if sentence_vector is not None and sentence_vector.size > 0:
                            cached_vectors.append(sentence_vector)
                            cached_metadata.append((idx, sentence_item))
                            continue
                    
                    # Missing from cache - will use batch embeddings
                    stext = sentence_item.get('sentence', '')
                    if stext:
                        missing_sentences.append(stext)
                        missing_metadata.append((idx, sentence_item))
                
                # Compute similarities for cached embeddings using vectorized operations (fast)
                cached_results = []
                if cached_vectors:
                    with self._stage(profiler, "retrieval.similarity_cached"):
                        cached_similarities_array = self._compute_vectorized_similarities(query_embedding, cached_vectors)
                        
                        # Build results from cached embeddings
                        for i, (idx, sentence_item) in enumerate(cached_metadata):
                            cached_results.append({
                                'sentence': sentence_item.get('sentence', ''),
                                'similarity': float(cached_similarities_array[i]),
                                'id': sentence_item['id'],
                                'document_id': sentence_item.get('document_id', ''),
                                'document_name': sentence_item.get('document_name', 'Unknown Document'),
                                'chunk_index': sentence_item.get('chunk_index', 0),
                                'drive_link': sentence_item.get('drive_link', ''),
                                'folder_name': sentence_item.get('folder_name', ''),
                                'original_index': idx
                            })
                
                # Get missing embeddings (separate timing to see the cost)
                missing_results = []
                if missing_sentences and VERTEX_AI_AVAILABLE and self.embedding_model:
                    try:
                        with self._stage(profiler, "retrieval.batch_embeddings_missing"):
                            # Get embeddings for missing sentences
                            missing_embeddings = self.embedding_model.get_embeddings(missing_sentences)
                            
                            # Compute similarities for missing embeddings
                            if missing_embeddings and len(missing_embeddings) == len(missing_sentences):
                                missing_vectors = [
                                    self._convert_embedding_to_array(emb) 
                                    for emb in missing_embeddings
                                ]
                                valid_missing = [
                                    (vec, meta) for vec, meta in zip(missing_vectors, missing_metadata)
                                    if vec is not None and vec.size > 0
                                ]
                                
                                if valid_missing:
                                    missing_vecs = [vec for vec, meta in valid_missing]
                                    missing_meta = [meta for vec, meta in valid_missing]
                                    
                                    with self._stage(profiler, "retrieval.similarity_missing"):
                                        missing_similarities = self._compute_vectorized_similarities(query_embedding, missing_vecs)
                                    
                                    # Add missing embeddings to cache for future use (batch update)
                                    with self.embedding_cache_lock:
                                        self._ensure_sentence_embedding_cache_initialized()
                                        for (vec, (idx, _)), sim in zip(valid_missing, missing_similarities):
                                            if 0 <= idx < len(self.sentence_embedding_cache):
                                                self.sentence_embedding_cache[idx] = vec
                                    
                                    # Build results from missing embeddings
                                    for i, (idx, sentence_item) in enumerate(missing_meta):
                                        missing_results.append({
                                            'sentence': sentence_item.get('sentence', ''),
                                            'similarity': float(missing_similarities[i]),
                                            'id': sentence_item['id'],
                                            'document_id': sentence_item.get('document_id', ''),
                                            'document_name': sentence_item.get('document_name', 'Unknown Document'),
                                            'chunk_index': sentence_item.get('chunk_index', 0),
                                            'drive_link': sentence_item.get('drive_link', ''),
                                            'folder_name': sentence_item.get('folder_name', ''),
                                            'original_index': idx
                                        })
                    except Exception as batch_error:
                        self.logger.warning(f"Failed to compute batch embeddings for missing sentences: {batch_error}")
                
                # Combine results (cached first, then missing)
                similarities = cached_results + missing_results
            
            # Fallback: use batch embeddings for all sentences if cache path failed
            elif VERTEX_AI_AVAILABLE and self.embedding_model and len(filtered_sentences) > 1:
                self.logger.warning("Sentence embedding cache unavailable; falling back to batch embedding computation")
                try:
                    sentence_texts = []
                    sentence_metadata = []
                    for idx, sentence_item in filtered_sentences:
                        sentence_item = self._prepare_sentence_metadata(sentence_item)
                        sentence_text = sentence_item.get('sentence', '')
                        if sentence_text:
                            sentence_texts.append(sentence_text)
                            sentence_metadata.append({
                                'idx': idx,
                                'sentence_item': sentence_item,
                                'sentence_text': sentence_text
                            })
                    with self._stage(profiler, "retrieval.batch_embeddings"):
                        all_texts = [query] + sentence_texts
                        batch_embeddings = self.embedding_model.get_embeddings(all_texts)
                    if batch_embeddings and len(batch_embeddings) == len(all_texts):
                        query_embedding = self._convert_embedding_to_array(batch_embeddings[0])
                        with self._stage(profiler, "retrieval.batch_similarity_calc"):
                            # Convert all sentence embeddings to arrays
                            sentence_embeddings = []
                            valid_metadata = []
                            for i, meta in enumerate(sentence_metadata):
                                sentence_emb = self._convert_embedding_to_array(batch_embeddings[i + 1])
                                if sentence_emb is not None and sentence_emb.size > 0:
                                    sentence_embeddings.append(sentence_emb)
                                    valid_metadata.append(meta)
                            
                            # Compute all similarities at once using vectorized operations
                            if sentence_embeddings and query_embedding is not None:
                                similarities_array = self._compute_vectorized_similarities(query_embedding, sentence_embeddings)
                                
                                # Build results from vectorized similarities
                                for i, meta in enumerate(valid_metadata):
                                    similarities.append({
                                        'sentence': meta['sentence_text'],
                                        'similarity': float(similarities_array[i]),
                                        'id': meta['sentence_item']['id'],
                                        'document_id': meta['sentence_item'].get('document_id', ''),
                                        'document_name': meta['sentence_item'].get('document_name', 'Unknown Document'),
                                        'chunk_index': meta['sentence_item'].get('chunk_index', 0),
                                        'drive_link': meta['sentence_item'].get('drive_link', ''),
                                        'folder_name': meta['sentence_item'].get('folder_name', ''),
                                        'original_index': meta['idx']
                                    })
                    else:
                        self.logger.warning("Batch embedding response size mismatch; falling back to sequential calculation")
                        raise ValueError("Batch embedding mismatch")
                except Exception as batch_error:
                    self.logger.warning(f"Batch embedding path failed: {batch_error}")
                    similarities = []
            if not similarities:
                for idx, sentence_item in filtered_sentences:
                    sentence_item = self._prepare_sentence_metadata(sentence_item)
                    sentence_text = sentence_item.get('sentence', '')
                    if not sentence_text:
                        continue
                    similarity = self.calculate_similarity(query, sentence_text)
                    
                    similarities.append({
                        'sentence': sentence_text,
                        'similarity': similarity,
                        'id': sentence_item['id'],
                        'document_id': sentence_item.get('document_id', ''),
                        'document_name': sentence_item.get('document_name', 'Unknown Document'),
                        'chunk_index': sentence_item.get('chunk_index', 0),
                        'drive_link': sentence_item.get('drive_link', ''),
                        'folder_name': sentence_item.get('folder_name', ''),
                        'original_index': idx
                    })
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        with self._stage(profiler, "retrieval.rank_and_select"):
            top_results = similarities[:top_k]
        
        self.logger.info(f"Found {len(top_results)} relevant excerpts from {len(filtered_sentences)} total sentences")
        for i, result in enumerate(top_results, 1):
            self.logger.debug(f"{i}. ({result['similarity']:.3f}) {result.get('sentence', '')[:80]}...")
        
        # If Vertex AI returned results, use them immediately
        if top_results:
            self.logger.info(f"Vertex AI returned {len(top_results)} results, using them (primary path)")
            return top_results
        
        # Only fall back to FAISS if Vertex AI returned no results
        if self.faiss_vector_store:
            self.logger.warning("Vertex AI returned no results, falling back to FAISS search (fallback path)")
            self.logger.info("FAISS search method: Using precomputed FAISS index for fast similarity search")
            with self._stage(profiler, "retrieval.faiss_total"):
                faiss_results = self.search_similar_documents_faiss(query, top_k, profiler=profiler)
            
            # If FAISS returned results, use them
            if faiss_results:
                self.logger.info(f"FAISS returned {len(faiss_results)} results, using them (fallback path)")
                return faiss_results
            else:
                self.logger.warning("FAISS also returned no results")
        
        # Return empty results if both methods failed
        return top_results

    async def query(self, question: str, context: str = "") -> str:
        """Query the RAG system with enhanced context awareness."""
        wall_start = time.time()
        profiler = self._start_latency_session("query")
        
        # Log folder filter state and context at the start of query
        if self.active_folder_filters:
            self.logger.info(f"🔍 Query with FOLDER FILTERS ACTIVE: {self.active_folder_filters} | Question: '{question[:100]}'")
        else:
            self.logger.info(f"🔍 Query with NO folder filters | Question: '{question[:100]}'")
        
        if context:
            self.logger.info(f"📝 Conversation context provided: {context[:200]}...")
        
        # Check response cache for exact and fuzzy query matches (aggressive caching)
        normalized_question = self._normalize_query_for_cache(question)
        cache_key = f"{normalized_question}:{context.strip().lower()}"
        
        # Try exact match first (fastest)
        if cache_key in self.response_cache:
            # Move to end (most recently used)
            cached_response = self.response_cache.pop(cache_key)
            self.response_cache[cache_key] = cached_response
            self.logger.info(f"Exact cache hit for query: '{question[:50]}...'")
            response_time = time.time() - wall_start
            self._end_latency_session(profiler, response_time=f"{response_time:.3f}s", result="cached_exact")
            return cached_response
        
        try:
            # Check if this is a greeting or simple expression
            # BUT: If there's conversation context, it's likely a follow-up question, not a greeting
            with self._stage(profiler, "classification.greeting_detection"):
                # If there's context, don't treat short queries as greetings (they're likely follow-ups)
                if context and context.strip():
                    is_greeting = False  # Context exists, treat as follow-up question
                    self.logger.debug(f"Context present, treating '{question}' as follow-up question, not greeting")
                else:
                    is_greeting = self._is_greeting_or_simple_expression(question)
            if is_greeting:
                self.logger.info(f"Identified as greeting: '{question}'")
                # Use a simpler prompt for greetings that doesn't require document search
                if VERTEX_AI_AVAILABLE and self.generative_model:
                    greeting_prompt = f"""
You are a professional financial and banking regulations expert chatbot. 
The user has sent a simple greeting: "{question}"

Respond appropriately to this greeting in a friendly and professional manner, 
letting them know you're here to help with questions about payment systems regulations. 
Keep your response concise and welcoming.

Response:"""
                    
                    try:
                        self.logger.info("Attempting to generate greeting response with Vertex AI")
                        greeting_model = self.fast_generative_model or self.generative_model
                        greeting_config = self.fast_generation_config.copy()
                        greeting_config.update({
                            "temperature": 0.7,
                            "max_output_tokens": 150,
                            "top_p": 0.8,
                            "top_k": 40
                        })
                        greeting_label = "greeting_fast" if greeting_model is self.fast_generative_model else "greeting_default"
                        result_text = self._invoke_vertex_model(
                            greeting_model,
                            greeting_config,
                            greeting_prompt,
                            profiler,
                            greeting_label
                        )
                        self.logger.info(f"Raw Vertex AI response: '{result_text}'")
                        if result_text:
                            # Ensure we have a complete, friendly response
                            # Check if response is too short or seems incomplete
                            is_too_short = len(result_text) < 30
                            seems_incomplete = (
                                result_text.lower().startswith(('hi', 'hello', 'hey')) and len(result_text) < 40 or
                                result_text.endswith(('...', '..', '.')) or
                                'as a' in result_text.lower() and len(result_text) < 50
                            )
                            
                            if is_too_short or seems_incomplete:
                                self.logger.warning(f"Response appears too short or incomplete: '{result_text}'")
                            else:
                                self.logger.info(f"Generated greeting response: {result_text[:50]}...")
                                
                                # Cache greeting responses too
                                if len(self.response_cache) >= self.response_cache_max_size:
                                    self.response_cache.popitem(last=False)
                                self.response_cache[cache_key] = result_text
                                self.response_cache.move_to_end(cache_key)
                                
                                response_time = time.time() - wall_start
                                total_elapsed = self._end_latency_session(
                                    profiler,
                                    response_time=f"{response_time:.3f}s",
                                    result="greeting_vertex",
                                    model=greeting_label
                                )
                                self.logger.info(f"Greeting response time: {response_time:.2f} seconds")
                                return result_text
                        else:
                            self.logger.warning("Empty response from Vertex AI for greeting")
                    except Exception as e:
                        self.logger.error(f"Error generating greeting response: {e}")
                        import traceback
                        self.logger.error(f"Full error: {traceback.format_exc()}")
                
                # Fallback response for greetings
                self.logger.info("Using fallback greeting response")
                response = "Hello! I'm here to help you with questions about payment systems regulations. How can I assist you today?"
                
                # Cache fallback greeting responses too
                if len(self.response_cache) >= self.response_cache_max_size:
                    self.response_cache.popitem(last=False)
                self.response_cache[cache_key] = response
                self.response_cache.move_to_end(cache_key)
                
                response_time = time.time() - wall_start
                total_elapsed = self._end_latency_session(
                    profiler,
                    response_time=f"{response_time:.3f}s",
                    result="greeting_fallback",
                    model="fallback"
                )
                self.logger.info(f"Greeting response time: {response_time:.2f} seconds")
                return response
            
            # Enrich short/vague queries with context before retrieval
            # This helps follow-up queries like "explain" find relevant documents
            enriched_question = self._enrich_query_with_context(question, context)
            
            # Search for relevant content for non-greeting queries with query expansion support
            # Vertex AI is primary, will fall back to FAISS internally if needed
            with self._stage(profiler, "retrieval.total"):
                relevant_content = self._retrieve_with_variants(enriched_question, profiler)
            
            # If Vertex AI search failed or returned no results, explicitly try FAISS as fallback
            if not relevant_content and self.faiss_vector_store and not self.use_faiss_primary:
                self.logger.info("Vertex AI search returned no results or failed, trying FAISS fallback")
                # Temporarily enable FAISS to force FAISS path
                original_use_faiss = self.use_faiss_primary
                self.use_faiss_primary = True
                with self._stage(profiler, "retrieval.faiss_fallback"):
                    # Call search directly with FAISS using enriched query
                    relevant_content = self.search_similar_documents_faiss(enriched_question, top_k=3, profiler=profiler)
                # Restore original setting
                self.use_faiss_primary = original_use_faiss
            
            if relevant_content:
                relevant_content = self._augment_with_neighbor_context(relevant_content)
                relevant_content = self._deduplicate_results(relevant_content, self.context_sentence_limit)
            
            if not relevant_content:
                no_results_response = "I couldn't find any relevant information in the documents to answer your question."
                # Cache no-results responses too
                if len(self.response_cache) >= self.response_cache_max_size:
                    self.response_cache.popitem(last=False)
                self.response_cache[cache_key] = no_results_response
                self.response_cache.move_to_end(cache_key)
                
                response_time = time.time() - wall_start
                total_elapsed = self._end_latency_session(profiler, response_time=f"{response_time:.3f}s", result="no_results")
                self.logger.info(f"Query response time: {response_time:.2f} seconds")
                return no_results_response
            
            # Prepare context from relevant documents and track source documents
            source_documents = {}
            existing_references = self.session_references or {}
            normalized_existing_references = {}
            doc_index_map = {}

            if existing_references:
                sortable_refs = []
                for doc_id, doc_info in existing_references.items():
                    if not isinstance(doc_info, dict):
                        continue
                    doc_id_str = str(doc_id)
                    sortable_refs.append((doc_info.get('index', float('inf')), doc_id_str, doc_info))
                sortable_refs.sort(key=lambda x: x[0])
                for new_index, (_, doc_id, doc_info) in enumerate(sortable_refs, start=1):
                    normalized_info = dict(doc_info)
                    normalized_info['index'] = new_index
                    normalized_existing_references[doc_id] = normalized_info
                    doc_index_map[doc_id] = new_index
            else:
                normalized_existing_references = {}
            
            # Create the source documents mapping with indices
            doc_counter = len(doc_index_map) + 1
            with self._stage(profiler, "prompt.build_source_documents"):
                for item in relevant_content:
                    raw_doc_id = item['document_id']
                    if raw_doc_id is None or raw_doc_id == "":
                        continue
                    doc_id = str(raw_doc_id)
                    if doc_id not in doc_index_map:
                        doc_index_map[doc_id] = doc_counter
                        doc_counter += 1
                    
                    doc_name = item['document_name']
                    drive_link = item.get('drive_link', '')
                    existing_info = normalized_existing_references.get(doc_id, {})
                    if doc_id not in source_documents:
                        source_documents[doc_id] = {
                            'name': doc_name or existing_info.get('name', ''),
                            'count': 0,
                            'drive_link': drive_link or existing_info.get('drive_link', ''),
                            'index': doc_index_map[doc_id]
                        }
                    source_documents[doc_id]['count'] += 1
                    if doc_name and not source_documents[doc_id]['name']:
                        source_documents[doc_id]['name'] = doc_name
                    if drive_link and not source_documents[doc_id]['drive_link']:
                        source_documents[doc_id]['drive_link'] = drive_link
            
            # Store references for this session (to be used when user asks for references)
            reference_candidates = {}

            merged_references = {
                doc_id: dict(doc_info)
                for doc_id, doc_info in normalized_existing_references.items()
                if isinstance(doc_info, dict)
            }
            for doc_id, doc_info in source_documents.items():
                if doc_id in merged_references:
                    merged_entry = merged_references[doc_id]
                    merged_entry['count'] = merged_entry.get('count', 0) + doc_info.get('count', 0)
                    if doc_info.get('name'):
                        merged_entry['name'] = doc_info['name']
                    if doc_info.get('drive_link'):
                        merged_entry['drive_link'] = doc_info['drive_link']
                    merged_entry['index'] = doc_info['index']
                else:
                    merged_references[doc_id] = dict(doc_info)
            
            # Ensure session reference indices remain sequential from 1..n
            sorted_reference_items = sorted(
                merged_references.items(),
                key=lambda item: doc_index_map.get(item[0], float('inf'))
            )
            renumbered_references = {}
            for new_index, (doc_id, doc_info) in enumerate(sorted_reference_items, start=1):
                updated_info = dict(doc_info)
                updated_info['index'] = new_index
                renumbered_references[doc_id] = updated_info
                doc_index_map[doc_id] = new_index
            reference_candidates = renumbered_references

            # Align current source document indices with the renumbered mapping
            for doc_id, doc_info in source_documents.items():
                doc_info['index'] = doc_index_map[doc_id]
            
            # Create document context with reference indices
            with self._stage(profiler, "prompt.build_context"):
                document_context_items = []
                for item in relevant_content:
                    raw_doc_id = item['document_id']
                    if raw_doc_id is None or raw_doc_id == "":
                        continue
                    doc_id = str(raw_doc_id)
                    doc_index = doc_index_map.get(doc_id)
                    if doc_index is None:
                        continue
                    document_context_items.append(f"[{doc_index}] {item.get('sentence', '')}")
                
                document_context = "\n".join(document_context_items)
            
            # If Vertex AI generative model is available, use it
            if VERTEX_AI_AVAILABLE and self.generative_model:
                # Create document mapping for the prompt
                with self._stage(profiler, "prompt.build_mapping"):
                    document_mapping_lines = []
                    sorted_docs = sorted(source_documents.items(), key=lambda x: x[1]['index'])
                    for doc_id, doc_info in sorted_docs:
                        document_mapping_lines.append(f"[{doc_info['index']}] Document: {doc_info['name']}")
                    document_mapping = "\n".join(document_mapping_lines)
                
                reasoning_hint = self._build_reasoning_hint(question, document_context)
                prompt_context = context
                if reasoning_hint:
                    hint_text = f"System Guidance: {reasoning_hint}"
                    prompt_context = f"{context}\n\n{hint_text}" if context else hint_text
                
                with self._stage(profiler, "prompt.template_format"):
                    prompt = self.prompt_template
                    formatted_prompt = prompt.format(
                        context=prompt_context,
                        document_context=document_context,
                        document_mapping=document_mapping,
                        question=question
                    )
                
                # Get response from Vertex AI generative model
                try:
                    result_text, model_label = self._generate_vertex_response(formatted_prompt, profiler)
                    
                    # Handle empty or None response
                    if not result_text:
                        self.logger.error(f"Vertex AI model '{model_label}' returned empty response for query: '{question[:50]}...'")
                        # Fallback: try to extract key information directly from document context
                        result_text = self._extract_key_information(document_context, question, source_documents)
                        if not result_text:
                            result_text = "I apologize, but I couldn't generate a response. Please try rephrasing your question or contact support."
                    
                    # Validate response completeness - check for incomplete responses that end with headers but no content
                    if result_text:
                        result_text_stripped = result_text.strip()
                        is_incomplete = False
                        
                        # Check if response ends with "Key Observations:" or similar headers with no content after
                        header_patterns = [
                            r'Key Observations:\s*$',
                            r'Observations:\s*$',
                            r'Summary:\s*$',
                            r'Findings:\s*$',
                        ]
                        
                        for pattern in header_patterns:
                            # Check if response ends with just the header (case-insensitive)
                            if re.search(pattern, result_text_stripped, re.IGNORECASE | re.MULTILINE):
                                # Split by the header to check what comes after
                                parts = re.split(pattern, result_text_stripped, flags=re.IGNORECASE | re.MULTILINE)
                                if len(parts) >= 2:
                                    # Check content after the header
                                    content_after = parts[-1].strip()
                                    # If there's no meaningful content after the header, it's incomplete
                                    if not content_after or len(content_after) < 10:
                                        is_incomplete = True
                                        self.logger.warning(
                                            f"Detected incomplete response ending with header '{pattern}' but no content. "
                                            f"Response length: {len(result_text_stripped)}, "
                                            f"Response preview: '{result_text_stripped[:150]}...'"
                                        )
                                        break
                                else:
                                    # Header found but no content after it
                                    is_incomplete = True
                                    self.logger.warning(
                                        f"Detected incomplete response with only header '{pattern}' and no content. "
                                        f"Response: '{result_text_stripped}'"
                                    )
                                    break
                        
                        # Also check if response is suspiciously short (less than 50 chars) and ends with a colon
                        if not is_incomplete and len(result_text_stripped) < 50 and result_text_stripped.endswith(':'):
                            is_incomplete = True
                            self.logger.warning(
                                f"Detected suspiciously short response ending with colon: '{result_text_stripped}'"
                            )
                        
                        # Check if response is just "Key Observations:" with nothing else
                        if not is_incomplete:
                            normalized_response = result_text_stripped.lower().strip()
                            if normalized_response in ['key observations:', 'observations:', 'summary:', 'findings:']:
                                is_incomplete = True
                                self.logger.warning(
                                    f"Detected response that is only a header with no content: '{result_text_stripped}'"
                                )
                        
                        # If incomplete, try fallback extraction
                        if is_incomplete:
                            self.logger.info("Attempting fallback extraction due to incomplete response")
                            fallback_text = self._extract_key_information(document_context, question, source_documents)
                            if fallback_text and len(fallback_text.strip()) > len(result_text_stripped):
                                self.logger.info("Using fallback extraction result as it's more complete")
                                result_text = fallback_text
                            else:
                                # If fallback didn't help, try regenerating with adjusted config
                                self.logger.warning("Fallback extraction didn't provide better result")
                                # Try one more time with a different model/config if available
                                if model_label == "fast" and self.generative_model:
                                    self.logger.info("Retrying with default model due to incomplete fast model response")
                                    try:
                                        default_config = self.default_generation_config.copy()
                                        default_config["max_output_tokens"] = min(
                                            default_config.get("max_output_tokens", 1500) + 200,
                                            2048
                                        )
                                        retry_result = self._invoke_vertex_model(
                                            self.generative_model,
                                            default_config,
                                            formatted_prompt,
                                            profiler,
                                            "default_retry"
                                        )
                                        if retry_result and len(retry_result.strip()) > len(result_text_stripped):
                                            self.logger.info("Retry with default model produced better result")
                                            result_text = retry_result
                                        else:
                                            self.logger.warning("Retry with default model didn't improve response")
                                    except Exception as retry_error:
                                        self.logger.warning(f"Retry generation failed: {retry_error}")
                    
                    # Post-process the response for better formatting
                    if result_text:
                        # Clean up the response
                        result_text = result_text.strip()
                        
                        # Remove document formatting artifacts
                        with self._stage(profiler, "post_process.clean_response"):
                            lines = result_text.split('\n')
                            cleaned_lines = []
                            
                            skip_patterns = self.skip_patterns
                        
                            for line in lines:
                                line = line.strip()
                                # Skip lines that contain document artifacts
                                should_skip = any(pattern in line for pattern in skip_patterns)
                                if line and not should_skip and len(line) > 3:
                                    # Clean up spacing and formatting
                                    line = ' '.join(line.split())
                                    line = self._normalize_bullet_discourse(line)
                                    cleaned_lines.append(line)
                            
                            result_text = '\n\n'.join(cleaned_lines).strip()

                        index_remap = self._finalize_session_references(reference_candidates, result_text)
                        if index_remap:
                            result_text = self._renumber_inline_citations(result_text, index_remap)

                        aligned_source_documents = self._align_source_documents_with_references(source_documents)
                        with self._stage(profiler, "post_process.citations"):
                            # Fix responses that incorrectly start with Usage/Example (before making citations clickable)
                            result_text = self._fix_response_format(result_text, question)
                            
                            # Remove empty section headings
                            result_text = self._remove_empty_sections(result_text)
                            
                            # Make citations clickable after fixing format
                            result_text = self._make_inline_citations_clickable(result_text, aligned_source_documents)
                        
                        with self._stage(profiler, "post_process.log_and_finalize"):
                            # Log the query for debugging
                            self.logger.info(f"Processed query: {question[:50]}...")
                            response_time = time.time() - wall_start
                            self.logger.info(f"Query response time: {response_time:.2f} seconds")
                        
                        # Cache the response for future exact and fuzzy matches
                        if len(self.response_cache) >= self.response_cache_max_size:
                            self.response_cache.popitem(last=False)
                        self.response_cache[cache_key] = result_text
                        self.response_cache.move_to_end(cache_key)
                        
                        total_elapsed = self._end_latency_session(
                            profiler,
                            response_time=f"{response_time:.3f}s",
                            result="vertex",
                            model=model_label
                        )
                        return result_text
                
                except Exception as e:
                    self.logger.error(f"Error generating response: {e}")
                    self.logger.error("Falling back to structured response due to Vertex AI error")
            
            # Fallback to structured response if no generative model available or if it failed
            self.logger.info("Using fallback structured response method")
            with self._stage(profiler, "fallback.extract_information"):
                result_text = self._extract_key_information(document_context, question, source_documents)

            index_remap = self._finalize_session_references(reference_candidates, result_text)
            if index_remap:
                result_text = self._renumber_inline_citations(result_text, index_remap)

            aligned_source_documents = self._align_source_documents_with_references(source_documents)
            with self._stage(profiler, "fallback.citations"):
                # Remove empty section headings
                result_text = self._remove_empty_sections(result_text)
                # Make inline citations clickable
                result_text = self._make_inline_citations_clickable(result_text, aligned_source_documents)
            
            with self._stage(profiler, "fallback.log_and_finalize"):
                # Log the query for debugging
                self.logger.info(f"Processed query: {question[:50]}...")
                response_time = time.time() - wall_start
                self.logger.info(f"Query response time: {response_time:.2f} seconds")
            
            # Cache the fallback response too
            if len(self.response_cache) >= self.response_cache_max_size:
                self.response_cache.popitem(last=False)
            self.response_cache[cache_key] = result_text
            self.response_cache.move_to_end(cache_key)
            
            total_elapsed = self._end_latency_session(profiler, response_time=f"{response_time:.3f}s", result="fallback")
            return result_text
            
        except Exception as e:
            self.logger.error(f"Error in query processing: {e}")
            response_time = time.time() - wall_start
            total_elapsed = self._end_latency_session(profiler, response_time=f"{response_time:.3f}s", result="error")
            self.logger.info(f"Query response time: {response_time:.2f} seconds")
            return f"I apologize, but I encountered an error while processing your question: {str(e)}"
    
    def _is_greeting_or_simple_expression(self, question: str) -> bool:
        """Check if the question is a greeting or simple expression that doesn't require document search."""
        question_lower = question.lower().strip()
        
        # Direct match for exact greetings
        greetings = [
            'hi', 'hello', 'hey', 'good morning', 'good afternoon', 
            'good evening', 'thanks', 'thank you', 'bye', 'goodbye'
        ]
        
        if question_lower in greetings:
            return True
        
        # Pattern matching for more flexible greetings
        greeting_patterns = [
            r'^\s*hi\s*!*\s*$', r'^\s*hello\s*!*\s*$', r'^\s*hey\s*!*\s*$',
            r'^\s*good\s*morning\s*!*\s*$', r'^\s*good\s*afternoon\s*!*\s*$', 
            r'^\s*good\s*evening\s*!*\s*$', r'^\s*thanks+\s*!*\s*$', 
            r'^\s*thank\s*you+\s*!*\s*$', r'^\s*bye+\s*!*\s*$', r'^\s*goodbye+\s*!*\s*$'
        ]
        
        for pattern in greeting_patterns:
            if re.match(pattern, question_lower):
                return True
        
        # For very short queries that are likely greetings
        # BUT exclude query words and action verbs that indicate real questions
        query_indicators = [
            'what', 'how', 'why', 'when', 'where', 'which', 'regulation', 
            'payment', 'bank', 'financial', 'psd', 'circular', 'license',
            'compliance', 'requirement', 'implement', 'process',
            'explain', 'describe', 'tell', 'show', 'define', 'detail',  # Added query action words
            'list', 'give', 'provide', 'elaborate', 'clarify'  # More action words
        ]
        
        if len(question_lower) <= 10 and not any(word in question_lower for word in query_indicators):
            return True
        
        return False
    
    def _extract_key_information(self, raw_text: str, question: str, source_documents: Dict = None) -> str:
        """Extract key information and provide a structured response using a generic approach."""
        try:
            # Clean and structure the raw text first
            cleaned_content = self._clean_document_content(raw_text)
            
            if not cleaned_content:
                return "I found relevant information in the documents, but it requires further clarification. Please contact the relevant authorities for complete details."
            
            # Create a professional response structure
            return self._format_professional_response(cleaned_content, question, source_documents)
                
        except Exception as e:
            self.logger.error(f"Error extracting key information: {e}")
            return "I found relevant information but encountered difficulties processing it clearly."
    
    def _clean_document_content(self, raw_text: str) -> str:
        """Clean document content by removing administrative artifacts and formatting properly."""
        # Split into sentences and clean each one
        sentences = raw_text.replace('\n', ' ').split('.')
        cleaned_sentences = []
        
        # Patterns to skip (administrative content)
        skip_patterns = [
            'payment systems department',
            'bangladesh bank head office',
            'website:', 'http://', 'www.',
            'psd circular', 'date:', 'dear sir',
            'managing director', 'ceo',
            'phone:', '+88-', 'yours sincerely',
            'general manager', 'acknowledge receipt',
            'circular no', 'dated',
            'subject to',
            'please refer to',
            'this is to inform',
            'in this regard',
            'with reference to'
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Skip if too short or contains administrative patterns
            if len(sentence) < 20:
                continue
                
            sentence_lower = sentence.lower()
            if any(pattern in sentence_lower for pattern in skip_patterns):
                continue
            
            # Clean up formatting and spacing
            sentence = ' '.join(sentence.split())
            
            # Add if it contains meaningful content
            if sentence and len(sentence) > 20:
                cleaned_sentences.append(sentence)
        
        # Join with proper sentence separation
        return '. '.join(cleaned_sentences) + '.' if cleaned_sentences else ''

    def _format_professional_response(self, content: str, question: str, source_documents: Dict = None) -> str:
        """Format content into a professional regulatory response with improved structure."""
        # Extract key points from the content with better organization
        key_points = self._extract_key_points_improved(content)
        
        # Create a more structured response
        response_parts = []
        
        # Determine which documents to cite (use first document index or default to 1)
        first_doc_index = 1
        if source_documents:
            # Get the first document index
            sorted_docs = sorted(source_documents.items(), key=lambda x: x[1]['index'])
            if sorted_docs:
                first_doc_index = sorted_docs[0][1]['index']
        
        # Main topic identification
        main_topic = "Definition and Overview"
        if key_points:
            identified_topic = self._identify_main_topic(question, key_points)
            # Only use the identified topic if it's different from the default
            if identified_topic != "Regulatory Information":
                main_topic = identified_topic
        
        # Add main topic as the first element
        response_parts.append(f"{main_topic}")
        
        # Add a brief overview sentence that captures the essence
        overview = "This response provides essential information about the topic and its key aspects."
        if key_points:
            generated_overview = self._generate_overview(question, key_points)
            if generated_overview:
                overview = generated_overview
        response_parts.append(f"\n{overview}")
        
        # Key points section with better formatting
        response_parts.append("\nKey Details:")
        if key_points:
            for i, point in enumerate(key_points, 1):
                # Make each point more structured and remove redundancy
                formatted_point = self._format_key_point(point)
                # Add proper citation to indicate this information comes from the documents
                response_parts.append(f"{i}. {formatted_point} [{first_doc_index}]")
        else:
            # Fallback if no clear key points
            clean_content = self._clean_document_content(content)
            if clean_content:
                # Limit the content length and add proper citation
                limited_content = clean_content[:500] + "..." if len(clean_content) > 500 else clean_content
                response_parts.append(f"1. {limited_content} [{first_doc_index}]")
            else:
                response_parts.append("1. Relevant information was found but could not be processed into a coherent response. [1]")
        
        # Join all parts with proper formatting
        response_text = '\n'.join(response_parts)
        
        # Make inline citations clickable
        response_text = self._make_inline_citations_clickable(response_text, source_documents)
        
        return response_text

    def _generate_overview(self, question: str, key_points: list) -> str:
        """Generate a concise overview that captures the main essence of the response."""
        if not key_points:
            return ""
            
        # Heuristic approach to create an overview based on question type
        question_lower = question.lower()
        
        if 'what' in question_lower:
            return "This response provides essential information about the topic and its key aspects."
        elif 'how' in question_lower:
            return "This response outlines the process and key steps involved."
        elif 'why' in question_lower:
            return "This response explains the reasons and justifications behind the policy."
        elif any(word in question_lower for word in ['benefit', 'advantage']):
            return "This response highlights the key advantages and value propositions."
        elif any(word in question_lower for word in ['require', 'must', 'should']):
            return "This response details the essential requirements and compliance obligations."
        elif 'implement' in question_lower:
            return "This response details the implementation framework and guidelines."
        else:
            return "This response provides key insights and important details about the topic."

    def _extract_key_points_improved(self, content: str) -> list:
        """Extract key points with improved logic for better relevance and coherence."""
        # Split content into sentences, preserving sentence boundaries
        import re
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        key_points = []
        
        # Define important keywords that indicate key information
        important_keywords = [
            'will', 'shall', 'must', 'required', 'advised', 'decided', 
            'implementation', 'system', 'bank', 'transaction', 'platform',
            'process', 'service', 'payment', 'switch', 'channel', 'regulation',
            'policy', 'procedure', 'guideline', 'standard', 'requirement',
            'purpose', 'objective', 'benefit', 'advantage', 'ensure', 'maintain',
            'establish', 'define', 'specify', 'authorize', 'prohibit', 'restrict',
            'effective', 'immediately', 'limit', 'maximum', 'minimum',
            'obligation', 'responsibility', 'authority', 'approval', 'consent',
            'license', 'permission', 'grant', 'issue', 'directive'
        ]
        
        # Combine short sentences that are likely fragments of the same point
        combined_sentences = []
        current_sentence = ""
        
        for sentence in sentences:
            # If sentence is very short, it might be a fragment
            if len(sentence) < 30:
                current_sentence += " " + sentence
            else:
                # If we have accumulated fragments, add them to the current sentence
                if current_sentence:
                    combined_sentences.append(current_sentence.strip() + " " + sentence)
                    current_sentence = ""
                else:
                    combined_sentences.append(sentence)
        
        # Add any remaining fragments
        if current_sentence:
            if combined_sentences:
                combined_sentences[-1] += " " + current_sentence.strip()
            else:
                combined_sentences.append(current_sentence.strip())
        
        for sentence in combined_sentences:
            sentence = sentence.strip()
            
            # Skip very short sentences or those with only administrative content
            if len(sentence) < 30:
                continue
            
            # Check if sentence contains important keywords
            sentence_lower = sentence.lower()
            has_important_keywords = any(keyword in sentence_lower for keyword in important_keywords)
            
            # Include sentences that either:
            # 1. Have important keywords, OR
            # 2. Are longer and seem substantive (heuristic: > 70 chars and not starting with transition words)
            is_substantive = (
                len(sentence) > 70 and 
                not sentence_lower.startswith(('therefore', 'however', 'moreover', 'furthermore', 'additionally', 'consequently'))
            )
            
            if has_important_keywords or is_substantive:
                # Clean and format the sentence
                clean_sentence = ' '.join(sentence.split())
                if clean_sentence and clean_sentence not in key_points:
                    # Ensure the sentence ends with proper punctuation
                    if not clean_sentence.endswith(('.', '!', '?')):
                        clean_sentence += '.'
                    key_points.append(clean_sentence)
        
        # Sort by relevance - prioritize sentences with more important keywords
        def relevance_score(sentence):
            score = 0
            sentence_lower = sentence.lower()
            for keyword in important_keywords:
                if keyword in sentence_lower:
                    score += 1
            # Also consider length as a factor for substantive information
            score += len(sentence) / 100
            return score
        
        key_points.sort(key=lambda x: relevance_score(x), reverse=True)
        return key_points[:5]  # Limit to 5 key points for readability

    def _format_key_point(self, point: str) -> str:
        """Format individual key points for better readability and precision."""
        # Remove redundant introductory phrases
        point = point.strip()
        
        # Remove common redundant phrases
        redundant_phrases = [
            'it is important to note that',
            'it should be noted that',
            'furthermore',
            'moreover',
            'in addition',
            'additionally',
            'however',
            'therefore',
            'consequently',
            'as a result',
            'in this context',
            'in this regard',
            'it is to be mentioned that',
            'it may be mentioned that'
        ]
        
        point_lower = point.lower()
        for phrase in redundant_phrases:
            if point_lower.startswith(phrase):
                # Remove the phrase and clean up the beginning
                point = point[len(phrase):].strip()
                # Capitalize the first letter
                if point:
                    point = point[0].upper() + point[1:]
                break
        
        # Clean up extra spaces and ensure proper formatting
        point = ' '.join(point.split())
        
        # Ensure proper ending punctuation
        if point and not point.endswith(('.', '!', '?')):
            point += '.'
            
        return point

    def _identify_main_topic(self, question: str, key_points: list) -> str:
        """Identify the main topic based on the question and key points with improved heuristics."""
        # Enhanced topic identification based on question keywords and content analysis
        question_lower = question.lower()
        
        if 'what' in question_lower:
            if 'is' in question_lower or 'are' in question_lower:
                return "Definition and Overview"
            else:
                return "Key Information"
        elif 'how' in question_lower:
            if 'implement' in question_lower or 'implementation' in question_lower:
                return "Implementation Process"
            elif 'apply' in question_lower or 'application' in question_lower:
                return "Application Procedure"
            else:
                return "Process Explanation"
        elif 'requirement' in question_lower or 'must' in question_lower or 'should' in question_lower:
            return "Regulatory Requirements"
        elif 'benefit' in question_lower or 'advantage' in question_lower:
            return "Benefits and Advantages"
        elif 'implement' in question_lower or 'implementation' in question_lower:
            return "Implementation Guidelines"
        elif 'purpose' in question_lower or 'objective' in question_lower:
            return "Purpose and Objectives"
        elif 'effective' in question_lower or 'date' in question_lower:
            return "Effective Date and Timeline"
        else:
            # Analyze key points to determine the main topic
            if key_points:
                first_point = key_points[0].lower()
                if any(word in first_point for word in ['implement', 'establish', 'create', 'launch']):
                    return "Implementation Framework"
                elif any(word in first_point for word in ['require', 'must', 'shall', 'comply']):
                    return "Compliance Requirements"
                elif any(word in first_point for word in ['benefit', 'advantage', 'improve', 'enhance']):
                    return "Value Proposition"
                elif any(word in first_point for word in ['process', 'procedure', 'step', 'method']):
                    return "Procedural Guidelines"
                elif any(word in first_point for word in ['effective', 'immediately', 'date']):
                    return "Effective Date and Timeline"
                elif any(word in first_point for word in ['limit', 'maximum', 'minimum']):
                    return "Limitations and Parameters"
            
            # Default to a general summary based on key points
            return "Regulatory Information"

    def _extract_citation_indices(self, text: str) -> Set[int]:
        """Extract citation indices like [1], [1, 2] from the provided text."""
        if not text:
            return set()

        # Remove HTML tags (e.g., from clickable links) before parsing citations
        text_without_tags = re.sub(r'<[^>]+>', '', text)
        matches = re.findall(r'\[(\d+(?:,\s*\d+)*)\]', text_without_tags)

        indices: Set[int] = set()
        for match in matches:
            parts = match.split(',')
            for part in parts:
                stripped = part.strip()
                if stripped.isdigit():
                    indices.add(int(stripped))
        return indices

    def _finalize_session_references(self, reference_candidates: Dict, response_text: str) -> Dict[int, int]:
        """Update session references to include only documents explicitly cited in the response.

        Returns:
            Dict[int, int]: Mapping of old citation indices to their new sequential indices.
        """
        index_remap: Dict[int, int] = {}

        if reference_candidates is None:
            return index_remap

        cited_indices = self._extract_citation_indices(response_text)
        if not cited_indices:
            return index_remap

        cited_documents = {}
        for doc_id, doc_info in reference_candidates.items():
            index = doc_info.get('index')
            if index in cited_indices:
                cited_documents[str(doc_id)] = dict(doc_info)

        if not cited_documents:
            return index_remap

        # Get existing references (documents cited in previous responses)
        # These should only contain documents that were cited before
        existing_references = {}
        if isinstance(self.session_references, dict):
            for doc_id, doc_info in self.session_references.items():
                if isinstance(doc_info, dict):
                    existing_references[str(doc_id)] = dict(doc_info)

        # Build combined references: only include documents that are explicitly cited
        # Start with cited documents from current response, then merge in existing references
        # that are also cited in this response (to preserve counts for documents cited multiple times)
        # This ensures uncited documents don't persist in session_references
        combined_references = {}
        cited_doc_ids = {str(doc_id) for doc_id in cited_documents.keys()}
        
        # First, add all currently cited documents
        for doc_id, doc_info in cited_documents.items():
            doc_id_str = str(doc_id)
            combined_references[doc_id_str] = dict(doc_info)
        
        # Then, merge in existing references only if they're also cited in this response
        # This preserves counts and metadata for documents cited multiple times across queries
        for doc_id_str in cited_doc_ids:
            if doc_id_str in existing_references:
                existing_info = existing_references[doc_id_str]
                cited_info = combined_references[doc_id_str]
                # Merge counts
                existing_count = existing_info.get('count', 0)
                combined_references[doc_id_str]['count'] = existing_count + cited_info.get('count', 0)
                # Preserve name and drive_link from existing if cited doesn't have them
                if not cited_info.get('name') and existing_info.get('name'):
                    combined_references[doc_id_str]['name'] = existing_info['name']
                if not cited_info.get('drive_link') and existing_info.get('drive_link'):
                    combined_references[doc_id_str]['drive_link'] = existing_info['drive_link']

        sorted_reference_items = sorted(
            combined_references.items(),
            key=lambda item: item[1].get('index', float('inf'))
        )

        renumbered_references = {}
        for new_index, (doc_id, doc_info) in enumerate(sorted_reference_items, start=1):
            updated_info = dict(doc_info)
            old_index = updated_info.get('index')
            updated_info['index'] = new_index
            renumbered_references[doc_id] = updated_info

            old_index_int: Optional[int] = None
            if old_index is not None:
                try:
                    old_index_int = int(old_index)
                except (TypeError, ValueError, OverflowError):
                    old_index_int = None
            if old_index_int is not None and old_index_int != new_index:
                index_remap[old_index_int] = new_index

        self.session_references = renumbered_references
        return index_remap

    def _renumber_inline_citations(self, text: str, index_remap: Dict[int, int]) -> str:
        """Renumber inline citation markers according to the provided index mapping."""
        if not text or not index_remap:
            return text

        def replace_citation(match):
            numbers_text = match.group(1)
            numbers = [part.strip() for part in numbers_text.split(',')]
            updated_numbers = []
            for number_str in numbers:
                if number_str.isdigit():
                    original_index = int(number_str)
                    new_index = index_remap.get(original_index, original_index)
                    updated_numbers.append(str(new_index))
                else:
                    updated_numbers.append(number_str)
            return '[' + ', '.join(updated_numbers) + ']'

        pattern = r'\[(\d+(?:,\s*\d+)*)\]'
        return re.sub(pattern, replace_citation, text)

    def _align_source_documents_with_references(self, source_documents: Dict) -> Dict:
        """Align source document indices with the finalized session references."""
        if not source_documents or not isinstance(source_documents, dict):
            return source_documents

        active_references = self.session_references or {}
        if not active_references:
            return {}

        aligned_documents: Dict[str, Dict] = {}
        for doc_id, doc_info in source_documents.items():
            doc_id_str = str(doc_id)
            reference_info = active_references.get(doc_id_str)
            if not reference_info:
                continue

            updated_info = dict(doc_info) if isinstance(doc_info, dict) else {}
            if 'index' in reference_info:
                updated_info['index'] = reference_info['index']
            if reference_info.get('drive_link'):
                updated_info['drive_link'] = reference_info['drive_link']
            if reference_info.get('name'):
                updated_info['name'] = reference_info['name']

            aligned_documents[doc_id_str] = updated_info

        return aligned_documents

    def _make_inline_citations_clickable(self, text: str, source_documents: Dict) -> str:
        """Convert inline citations like [1], [2] to clickable links pointing to Google Drive files."""
        if not source_documents or not text:
            return text
        
        # Create a mapping from index to drive link
        index_to_link = {}
        sorted_docs = sorted(source_documents.items(), key=lambda x: x[1]['index'])
        for doc_id, doc_info in sorted_docs:
            index = doc_info['index']
            drive_link = doc_info.get('drive_link', '')
            if drive_link:
                index_to_link[index] = drive_link
        
        # Replace inline citations with clickable links
        # Pattern to match citations like [1], [2], [1, 2], etc.
        def replace_citation(match):
            numbers_text = match.group(1)   # e.g., "1" or "1, 2"
            
            # Split numbers and create links for each
            numbers = [int(n.strip()) for n in numbers_text.split(',')]
            linked_parts = []
            
            for num in numbers:
                if num in index_to_link:
                    # Create clickable link for this citation (opens in new tab)
                    linked_parts.append(f'<a href="{index_to_link[num]}" target="_blank" rel="noopener noreferrer" style="color: #60a5fa; text-decoration: underline; cursor: pointer;">{num}</a>')
                else:
                    # Keep as plain text if no link available
                    linked_parts.append(str(num))
            
            # Join all parts with commas and wrap in brackets
            return '[' + ', '.join(linked_parts) + ']'
        
        # Replace all citation patterns
        # Pattern matches [1], [2], [1, 2], [1, 2, 3], etc.
        pattern = r'\[(\d+(?:,\s*\d+)*)\]'
        result = re.sub(pattern, replace_citation, text)
        
        return result
    
    def _make_custom_citations_clickable(self, text: str, source_documents: Dict) -> str:
        """Convert custom citation markers like [CITE:1], [CITE:2] to clickable links pointing to Google Drive files."""
        if not source_documents or not text:
            return text
        
        # Create a mapping from index to drive link
        index_to_link = {}
        sorted_docs = sorted(source_documents.items(), key=lambda x: x[1]['index'])
        for doc_id, doc_info in sorted_docs:
            index = doc_info['index']
            drive_link = doc_info.get('drive_link', '')
            if drive_link:
                index_to_link[index] = drive_link
        
        # Replace custom citation markers with clickable links
        # Pattern to match citations like [CITE:1], [CITE:2], etc.
        def replace_citation(match):
            number_text = match.group(1)   # e.g., "1"
            
            try:
                num = int(number_text)
                if num in index_to_link:
                    # Create clickable link for this citation (opens in new tab)
                    return f'[<a href="{index_to_link[num]}" target="_blank" rel="noopener noreferrer" style="color: #60a5fa; text-decoration: underline; cursor: pointer;">{num}</a>]'
                else:
                    # Keep as plain text if no link available
                    return f'[{num}]'
            except ValueError:
                # If conversion to int fails, keep original
                return match.group(0)
        
        # Replace all citation patterns
        # Pattern matches [CITE:1], [CITE:2], etc.
        pattern = r'\[CITE:(\d+)\]'
        result = re.sub(pattern, replace_citation, text)
        
        return result
    
    def _fix_response_format(self, text: str, question: str) -> str:
        """Fix responses that incorrectly start with Usage/Example by extracting and synthesizing direct answer."""
        if not text or not text.strip():
            return text
        
        text = text.strip()
        
        # Check if response incorrectly starts with Usage/Example/Key Observations
        if text.startswith(("Usage:", "Example:", "Key Observations:")):
            self.logger.warning(f"Response incorrectly starts with Usage/Example/Key Observations. Fixing format. Question: {question[:50]}")
            
            # Extract sections
            sections = {}
            current_section = None
            current_content = []
            
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    if current_section and current_content:
                        sections[current_section] = '\n'.join(current_content).strip()
                        current_content = []
                    continue
                
                if line.startswith("Usage:"):
                    if current_section and current_content:
                        sections[current_section] = '\n'.join(current_content).strip()
                    current_section = "usage"
                    current_content = []
                    # Extract content after "Usage:"
                    usage_content = line[6:].strip()  # Remove "Usage:"
                    if usage_content:
                        current_content.append(usage_content)
                elif line.startswith("Example:"):
                    if current_section and current_content:
                        sections[current_section] = '\n'.join(current_content).strip()
                    current_section = "example"
                    current_content = []
                    # Extract content after "Example:"
                    example_content = line[8:].strip()  # Remove "Example:"
                    if example_content:
                        current_content.append(example_content)
                elif line.startswith("Key Observations:"):
                    if current_section and current_content:
                        sections[current_section] = '\n'.join(current_content).strip()
                    current_section = "key_observations"
                    current_content = []
                    # Extract content after "Key Observations:"
                    ko_content = line[17:].strip()  # Remove "Key Observations:"
                    if ko_content:
                        current_content.append(ko_content)
                else:
                    if current_section:
                        current_content.append(line)
                    else:
                        # Content before any section - this shouldn't happen if we're fixing
                        current_content.append(line)
            
            # Save last section
            if current_section and current_content:
                sections[current_section] = '\n'.join(current_content).strip()
            
            # Synthesize direct answer from available sections
            direct_answer_parts = []
            
            # For "what is" or "what is the definition of" questions, synthesize from usage
            question_lower = question.lower()
            if "what is" in question_lower or "definition" in question_lower:
                # Extract the term being asked about - look for quoted terms or terms after "definition of"
                term = None
                # Try to find quoted term first
                quoted_match = re.search(r'["\']([^"\']+)["\']', question)
                if quoted_match:
                    term = quoted_match.group(1)
                else:
                    # Try to find term after "definition of" or "what is"
                    def_match = re.search(r'(?:definition of|what is)\s+([^"\']+?)(?:\s+under|\s+in|$)', question, re.IGNORECASE)
                    if def_match:
                        term = def_match.group(1).strip()
                
                if not term:
                    term = "the term"
                
                # Use usage content to synthesize definition
                if "usage" in sections and sections["usage"]:
                    usage_text = sections["usage"]
                    # Extract citations from usage
                    citations = re.findall(r'\[\d+(?:,\s*\d+)*\]', usage_text)
                    citation_str = citations[0] if citations else "[1]"
                    
                    # Synthesize definition from usage content
                    # Remove citations temporarily for synthesis
                    usage_clean = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', usage_text).strip()
                    if usage_clean.endswith('.'):
                        usage_clean = usage_clean[:-1]
                    
                    # Create definition sentence
                    direct_answer = f"{term} refers to {usage_clean.lower()}{citation_str}."
                    direct_answer_parts.append(direct_answer)
                elif "example" in sections and sections["example"]:
                    # Fallback to example if usage not available
                    example_text = sections["example"]
                    citations = re.findall(r'\[\d+(?:,\s*\d+)*\]', example_text)
                    citation_str = citations[0] if citations else "[1]"
                    example_clean = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', example_text).strip()
                    if example_clean.endswith('.'):
                        example_clean = example_clean[:-1]
                    direct_answer = f"{term} refers to entities that {example_clean.lower()}{citation_str}."
                    direct_answer_parts.append(direct_answer)
                else:
                    # Last resort - create a generic definition
                    direct_answer = f"{term} is defined in the regulations as described below{citation_str if 'usage' in sections or 'example' in sections else ''}."
                    direct_answer_parts.append(direct_answer)
            else:
                # For other question types, synthesize from available content
                if "usage" in sections and sections["usage"]:
                    usage_text = sections["usage"]
                    citations = re.findall(r'\[\d+(?:,\s*\d+)*\]', usage_text)
                    citation_str = citations[0] if citations else "[1]"
                    usage_clean = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', usage_text).strip()
                    direct_answer_parts.append(f"{usage_clean}{citation_str}")
                elif "example" in sections and sections["example"]:
                    example_text = sections["example"]
                    citations = re.findall(r'\[\d+(?:,\s*\d+)*\]', example_text)
                    citation_str = citations[0] if citations else "[1]"
                    example_clean = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', example_text).strip()
                    direct_answer_parts.append(f"{example_clean}{citation_str}")
            
            # Reconstruct response with direct answer first
            fixed_parts = []
            if direct_answer_parts:
                fixed_parts.append('\n'.join(direct_answer_parts))
            
            if "usage" in sections and sections["usage"]:
                fixed_parts.append(f"Usage:\n{sections['usage']}")
            
            if "example" in sections and sections["example"]:
                fixed_parts.append(f"Example:\n{sections['example']}")
            
            if "key_observations" in sections and sections["key_observations"]:
                fixed_parts.append(f"Key Observations:\n{sections['key_observations']}")
            
            return '\n\n'.join(fixed_parts) if fixed_parts else text
        
        return text
    
    def _remove_empty_sections(self, text: str) -> str:
        """Remove section headings (Usage:, Example:, Key Observations:) that have no content."""
        if not text or not text.strip():
            return text
        
        def has_actual_content(content_lines: list) -> bool:
            """Check if content lines contain actual text (not just whitespace or HTML tags)."""
            if not content_lines:
                return False
            
            combined = ' '.join(content_lines)
            # Remove HTML tags to check for actual text
            text_only = re.sub(r'<[^>]+>', '', combined).strip()
            # Remove citations to check for actual text
            text_only = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text_only).strip()
            return len(text_only) > 0
        
        lines = text.split('\n')
        result_lines = []
        current_section = None
        current_section_content = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Check if this line is a section heading
            if line == "Usage:" or line.startswith("Usage:"):
                # Save previous section if it had content
                if current_section and has_actual_content(current_section_content):
                    result_lines.append(f"{current_section}")
                    result_lines.extend(current_section_content)
                    result_lines.append("")  # Add blank line after section
                # If previous section was empty, skip it
                
                # Start new section
                current_section = "Usage:"
                current_section_content = []
                # Check if there's content on the same line after "Usage:"
                if len(line) > 6:  # More than just "Usage:"
                    content = line[6:].strip()
                    if content:
                        current_section_content.append(content)
                
            elif line == "Example:" or line.startswith("Example:"):
                # Save previous section if it had content
                if current_section and has_actual_content(current_section_content):
                    result_lines.append(f"{current_section}")
                    result_lines.extend(current_section_content)
                    result_lines.append("")  # Add blank line after section
                # If previous section was empty, skip it
                
                # Start new section
                current_section = "Example:"
                current_section_content = []
                # Check if there's content on the same line after "Example:"
                if len(line) > 8:  # More than just "Example:"
                    content = line[8:].strip()
                    if content:
                        current_section_content.append(content)
                
            elif line == "Key Observations:" or line.startswith("Key Observations:"):
                # Save previous section if it had content
                if current_section and has_actual_content(current_section_content):
                    result_lines.append(f"{current_section}")
                    result_lines.extend(current_section_content)
                    result_lines.append("")  # Add blank line after section
                # If previous section was empty, skip it
                
                # Start new section
                current_section = "Key Observations:"
                current_section_content = []
                # Check if there's content on the same line after "Key Observations:"
                if len(line) > 17:  # More than just "Key Observations:"
                    content = line[17:].strip()
                    if content:
                        current_section_content.append(content)
                
            else:
                # Regular content line
                if current_section:
                    # This is content for the current section
                    if line:  # Only add non-empty lines
                        current_section_content.append(line)
                else:
                    # This is content before any section (direct answer)
                    result_lines.append(line)
            
            i += 1
        
        # Handle the last section
        if current_section and has_actual_content(current_section_content):
            result_lines.append(f"{current_section}")
            result_lines.extend(current_section_content)
        # If last section is empty, we don't add it (it's already skipped)
        
        # Join lines and clean up extra blank lines
        result = '\n'.join(result_lines)
        # Remove multiple consecutive blank lines
        result = re.sub(r'\n{3,}', '\n\n', result)
        return result.strip()
    
    def get_session_references(self) -> Dict:
        """Get the references from the current session."""
        return self.session_references

    def format_references(self) -> str:
        """Format the session references as a clickable reference list."""
        if not self.session_references:
            return "No references available for this session."
        
        # Sort documents by their index
        sorted_docs = sorted(self.session_references.items(), key=lambda x: x[1]['index'])
        
        references_lines = ["**References:**"]
        for doc_id, doc_info in sorted_docs:
            index = doc_info['index']
            doc_name = doc_info['name']
            drive_link = doc_info.get('drive_link', '')
            
            # Create reference line with or without link
            if drive_link:
                references_lines.append(f"{index}. [{doc_name}]({drive_link})")
            else:
                references_lines.append(f"{index}. {doc_name}")
        
        return "\n".join(references_lines)