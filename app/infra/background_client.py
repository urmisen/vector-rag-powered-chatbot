import asyncio
import threading
import time
from concurrent.futures import Future
from functools import lru_cache
from typing import Any, Callable, Dict, Optional

from app.client import MCPClient
from app.infra.logger import logger
from app.infra.metrics import record_latency_metric


CallableWithClient = Callable[[MCPClient], Any]
manager_logger = logger.getChild("BackgroundClientManager")


class ClientWorker:
    """Run an MCPClient inside a persistent background event loop."""

    def __init__(self, email: Optional[str]):
        self.email = email
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._client: Optional[MCPClient] = None
        self._startup_future: Future = Future()
        self._shutdown_event = threading.Event()
        self._lock = threading.Lock()
        self._logger = logger.getChild(f"BackgroundClient[{email or 'anonymous'}]")
        self._started_at: Optional[float] = None
        self._ready_at: Optional[float] = None
        self._last_error: Optional[BaseException] = None

    def start(self) -> Future:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return self._startup_future

            self._started_at = time.perf_counter()
            self._ready_at = None
            self._last_error = None
            self._thread = threading.Thread(target=self._run, name="mcp-worker", daemon=True)
            self._thread.start()
            return self._startup_future

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)

        stage_metrics = []
        overall_start = time.perf_counter()
        try:
            ctor_start = time.perf_counter()
            client = MCPClient(authenticated_user_email=self.email)
            stage_metrics.append(("ctor", time.perf_counter() - ctor_start))

            init_start = time.perf_counter()
            loop.run_until_complete(client.initialize())
            stage_metrics.append(("initialize", time.perf_counter() - init_start))

            self._client = client
            elapsed = time.perf_counter() - overall_start
            self._ready_at = time.perf_counter()
            record_latency_metric(
                label="background_client_bootstrap",
                stages=stage_metrics,
                total_seconds=elapsed,
                extra={"user": self.email or "anonymous", "status": "success"},
            )
            if not self._startup_future.done():
                self._startup_future.set_result(True)
            self._logger.info(
                "Background MCP client ready in %.0fms",
                elapsed * 1000,
            )
            loop.run_forever()
        except Exception as exc:
            elapsed = time.perf_counter() - overall_start
            self._last_error = exc
            record_latency_metric(
                label="background_client_bootstrap",
                stages=stage_metrics,
                total_seconds=elapsed,
                extra={
                    "user": self.email or "anonymous",
                    "status": "error",
                    "error_message": str(exc),
                },
            )
            self._logger.exception("Background MCP client failed to start")
            if not self._startup_future.done():
                self._startup_future.set_exception(exc)
        finally:
            try:
                if loop.is_running():
                    loop.stop()
            finally:
                loop.close()
            self._shutdown_event.set()

    def is_ready(self) -> bool:
        fut = self._startup_future
        return fut.done() and fut.exception() is None

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        if not self._loop:
            raise RuntimeError("Background loop not initialized yet")
        return self._loop

    @property
    def client(self) -> MCPClient:
        if not self._client:
            raise RuntimeError("Background client not initialized yet")
        return self._client

    def status(self) -> Dict[str, Any]:
        return {
            "ready": self.is_ready(),
            "error": str(self._last_error) if self._last_error else None,
            "started_at": self._started_at,
            "ready_at": self._ready_at,
        }

    def adopt_user(self, email: Optional[str]) -> None:
        """Retarget the worker to a different authenticated user without restarting."""
        # Ensure the worker has started before adopting
        self.start().result()

        def _reset(client: MCPClient):
            client.reset_for_user(email)

        self.run_sync(_reset)
        self.email = email
        self._logger = logger.getChild(f"BackgroundClient[{email or 'anonymous'}]")
        self._logger.info("Worker adopted for user=%s", email or "anonymous")

    def run_sync(self, func: CallableWithClient) -> Any:
        future: Future = Future()

        def _invoke():
            try:
                result = func(self.client)
                future.set_result(result)
            except Exception as exc:  # pragma: no cover - defensive
                future.set_exception(exc)

        self.loop.call_soon_threadsafe(_invoke)
        return future.result()

    def run_async(self, coro_factory: CallableWithClient) -> Any:
        outer_future: Future = Future()

        def _start():
            try:
                coro = coro_factory(self.client)
                task = asyncio.ensure_future(coro, loop=self.loop)

                def _done(task: asyncio.Future):
                    exc = task.exception()
                    if exc:
                        outer_future.set_exception(exc)
                    else:
                        outer_future.set_result(task.result())

                task.add_done_callback(_done)
            except Exception as exc:  # pragma: no cover - defensive
                outer_future.set_exception(exc)

        self.loop.call_soon_threadsafe(_start)
        return outer_future.result()

    def shutdown(self):
        if not self._loop:
            return

        def _stop():
            self.loop.stop()

        self.loop.call_soon_threadsafe(_stop)
        self._shutdown_event.wait(timeout=5)


class ClientBridge:
    """Thread-safe facade exposed to the Streamlit layer."""

    def __init__(self, worker: ClientWorker):
        self._worker = worker

    def wait_until_ready(self, timeout: Optional[float] = None) -> None:
        self._worker.start().result(timeout=timeout)

    def is_ready(self) -> bool:
        fut = self._worker.start()
        return fut.done() and not fut.cancelled() and fut.exception() is None

    def get_error(self) -> Optional[BaseException]:
        fut = self._worker.start()
        if fut.done() and fut.exception():
            return fut.exception()
        return None

    # Identity helpers
    def get_authenticated_user(self) -> Optional[str]:
        return self._worker.run_sync(lambda client: client.authenticated_user)

    def get_gcp_username(self) -> str:
        return self._worker.run_sync(lambda client: client.gcp_username)

    def get_user_id(self) -> str:
        return self._worker.run_sync(lambda client: client.user_id)

    # Conversation helpers
    def get_conversation_id(self) -> str:
        return self._worker.run_sync(lambda client: client.get_conversation_id())

    def set_conversation_id(self, conversation_id: str) -> None:
        self._worker.run_sync(lambda client: client.set_conversation_id(conversation_id))

    def is_conversation_saved(self) -> bool:
        return self._worker.run_sync(lambda client: client.is_conversation_session_saved())

    def set_conversation_saved(self, value: bool) -> None:
        self._worker.run_sync(lambda client: client.set_conversation_session_saved(value))

    def add_message(self, role: str, content: str, timestamp: str) -> None:
        self._worker.run_sync(lambda client: client.add_message_entry(role, content, timestamp))

    def get_chat_history(self):
        return self._worker.run_sync(lambda client: client.get_chat_history())

    def clear_chat_history(self) -> None:
        self._worker.run_sync(lambda client: client.clear_chat_history())

    def save_conversation_session(self) -> None:
        self._worker.run_sync(lambda client: client.save_conversation_session_now())

    def load_conversation_session(self, conversation_id: str):
        return self._worker.run_sync(lambda client: client.load_conversation_session(conversation_id))

    # Query helpers
    def process_query(self, prompt: str) -> str:
        return self._worker.run_async(lambda client: client.process_query(prompt))

    def format_references(self) -> str:
        return self._worker.run_sync(lambda client: client.rag.format_references())

    # Folder filtering helpers (supports multiple folders)
    def get_available_folders(self):
        """Get list of available folders for document filtering."""
        return self._worker.run_sync(lambda client: client.get_available_folders())
    
    def set_folder_filters(self, folder_names) -> None:
        """Set the active folder filters for document search."""
        self._worker.run_sync(lambda client: client.set_folder_filters(folder_names))
    
    def add_folder_filter(self, folder_name: str) -> None:
        """Add a folder to the active filters."""
        self._worker.run_sync(lambda client: client.add_folder_filter(folder_name))
    
    def remove_folder_filter(self, folder_name: str) -> None:
        """Remove a folder from the active filters."""
        self._worker.run_sync(lambda client: client.remove_folder_filter(folder_name))
    
    def get_current_folder_filters(self):
        """Get the currently active folder filters."""
        return self._worker.run_sync(lambda client: client.get_current_folder_filters())
    
    def clear_folder_filters(self) -> None:
        """Clear all active folder filters."""
        self._worker.run_sync(lambda client: client.clear_folder_filters())

    # BigQuery accessors
    def get_user_conversation_sessions(self, user_id: str):
        return self._worker.run_sync(
            lambda client: client.bq_manager.get_user_conversation_sessions(user_id)
        )

    def get_conversation_session(self, conversation_id: str):
        return self._worker.run_sync(
            lambda client: client.bq_manager.get_conversation_session(conversation_id)
        )

    # Shutdown
    def close(self):
        self._worker.run_async(lambda client: client.close())


class BackgroundClientManager:
    """Manage background MCPClient workers keyed by user email."""

    def __init__(self):
        self._workers: Dict[str, ClientWorker] = {}
        self._lock = threading.Lock()
        self._prewarmed_worker: Optional[ClientWorker] = None

    def _create_worker(self, email: Optional[str]) -> ClientWorker:
        worker = ClientWorker(email=email)
        worker.start()
        return worker

    def ensure_prewarmed(self) -> None:
        """Ensure a warm worker is starting or ready for upcoming authenticated users."""
        with self._lock:
            if self._prewarmed_worker:
                # If the worker thread died, drop it and start a new one
                if self._prewarmed_worker._thread and not self._prewarmed_worker._thread.is_alive():
                    self._prewarmed_worker = None
                else:
                    return
            manager_logger.info("Starting prewarmed MCP client worker")
            self._prewarmed_worker = self._create_worker(email=None)

    def get_or_create(self, email: Optional[str]) -> ClientBridge:
        key = email or "anonymous"
        with self._lock:
            if key in self._workers:
                worker = self._workers[key]
                worker.start()
                return ClientBridge(worker)

            # Reuse prewarmed worker when assigning an authenticated user
            if email and self._prewarmed_worker:
                worker = self._prewarmed_worker
                self._prewarmed_worker = None
                future = worker.start()
                self._workers[key] = worker

                def _adopt_when_ready():
                    try:
                        future.result()
                        worker.adopt_user(email)
                    except Exception:
                        logger.exception("Failed to adopt prewarmed worker for user=%s; shutting down warm worker.", email)
                        worker.shutdown()
                        with self._lock:
                            if self._workers.get(key) is worker:
                                self._workers.pop(key, None)

                threading.Thread(target=_adopt_when_ready, daemon=True).start()
                # Immediately kick off a new prewarm cycle
                threading.Thread(target=self.ensure_prewarmed, daemon=True).start()
                return ClientBridge(worker)

            worker = self._create_worker(email)
            self._workers[key] = worker
        return ClientBridge(worker)

    def shutdown_all(self):
        with self._lock:
            if self._prewarmed_worker:
                self._prewarmed_worker.shutdown()
                self._prewarmed_worker = None
            for worker in self._workers.values():
                worker.shutdown()
            self._workers.clear()


@lru_cache(maxsize=1)
def get_background_client_manager() -> BackgroundClientManager:
    return BackgroundClientManager()

