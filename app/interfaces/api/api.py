### api.py

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from datetime import datetime
import asyncio
import time
import uvicorn

from app.client import MCPClient
from app.core import ChatRequest, ChatResponse, RegulationItem
from app.infra.logger import logger

app = FastAPI(title="Pay Regulations Chatbot API")
api_logger = logger.getChild("API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@asynccontextmanager ## The context manager ensures the file is closed even if an exception occurs.
async def lifespan(app: FastAPI): ## In FastAPI, the lifespan function is an advanced pattern to manage resources when your application starts and stops.
    api_logger.info("Starting application...")
    init_start = time.perf_counter()
    app.state.mcp_client = MCPClient() ## app.state is a special attribute of the application object that lets you store arbitrary global variables or objects for the whole app.
    await app.state.mcp_client.initialize()
    api_logger.info("Client initialized in %.3fs", time.perf_counter() - init_start)
    app.state.startup_time = datetime.utcnow()
    yield ## Signal that the app is ready and can accept requests
    await app.state.mcp_client.close() ## On shutdown, gracefully close the client

app.router.lifespan_context = lifespan

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker and load balancers"""
    try:
        # Verify client is initialized
        if not app.state.mcp_client:
            raise HTTPException(status_code=503, detail="Client not initialized")

        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "uptime": str(datetime.utcnow() - app.state.startup_time),
                "services": {
                    "rag_client": "active",
                    "vector_store": "available"
                }
            }
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")

@app.post("/chat")
async def chat(request: ChatRequest): ## Registers the function as the handler for POST requests at /chat and Expects the input body to match the ChatRequest model
    request_started = time.perf_counter()
    query_length = len(request.query or "")
    api_logger.info("Chat request received; query_length=%d", query_length)
    try:
        response = await app.state.mcp_client.process_query(request.query)
        elapsed = time.perf_counter() - request_started
        api_logger.info("Chat request completed in %.3fs", elapsed)
        return {"response": response}
    except Exception as e:
        elapsed = time.perf_counter() - request_started
        api_logger.error("API error after %.3fs: %s", elapsed, str(e))
        raise HTTPException(500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)