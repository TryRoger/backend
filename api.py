"""
FastAPI server for step-by-step task execution with UI element detection.

Endpoints:
- /execute_step: Single-prompt endpoint for bounding box detection + next step preview
- /health: Health check

Model: gemini-3-flash-preview

Flow:
1. Step 1: Full analysis - returns current_step (what, where, box_2d) + next_step preview
2. Step 2+: Focused analysis - client sends known what/where, gets box_2d + next_step preview
   (Faster because model just finds bounding box for known element)

Usage:
    uvicorn api:app --port 8000
"""

import json
import time
import datetime

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

from dotenv import load_dotenv
from controllers.basic_mvp import MODEL
from controllers import stripe_handler

# Import routers
from routes.step import router as step_router
from routes.user import router as user_router
from routes.payments import router as payments_router

# Load environment variables
load_dotenv()

# Server startup time for uptime tracking
SERVER_START_TIME = time.time()

app = FastAPI(
    title="Screen Element Detection & Guidance API",
    description="Detects UI elements and provides step-by-step guidance using Gemini",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    """Pre-initialize caches on server startup for faster first requests."""
    try:
        stripe_handler.initialize_stripe_cache()
    except Exception as e:
        print(f"[STARTUP] Warning: Could not initialize Stripe cache: {e}")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Print request details
        print("\n" + "=" * 60)
        print("INCOMING REQUEST")
        print("=" * 60)
        print(f"Method: {request.method}")
        print(f"URL: {request.url}")
        print(f"Path: {request.url.path}")
        print(f"Query Params: {dict(request.query_params)}")
        print(f"Client: {request.client.host}:{request.client.port}" if request.client else "Client: Unknown")

        # Print headers
        print("\nHeaders:")
        for name, value in request.headers.items():
            print(f"  {name}: {value}")

        # Read and print body
        body = await request.body()
        if body:
            print(f"\nBody ({len(body)} bytes):")
            try:
                # Try to decode as text
                body_text = body.decode('utf-8')
                # Try to parse as JSON for pretty printing
                try:
                    body_json = json.loads(body_text)
                    print(json.dumps(body_json, indent=2))
                except json.JSONDecodeError:
                    print(body_text)
            except UnicodeDecodeError:
                print(f"  [Binary data, {len(body)} bytes]")
        else:
            print("\nBody: (empty)")

        print("=" * 60 + "\n")

        # Restore body for downstream handlers
        async def receive():
            return {"type": "http.request", "body": body}
        request._receive = receive

        response = await call_next(request)
        return response


app.add_middleware(RequestLoggingMiddleware)

# Include routers
app.include_router(step_router)
app.include_router(user_router)
app.include_router(payments_router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    uptime_seconds = time.time() - SERVER_START_TIME
    return {
        "status": "healthy",
        "model": MODEL,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "uptime_seconds": round(uptime_seconds, 2),
        "uptime_human": f"{int(uptime_seconds // 3600)}h {int((uptime_seconds % 3600) // 60)}m {int(uptime_seconds % 60)}s"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

