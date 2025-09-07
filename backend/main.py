import os
import uvicorn
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# --- CORS Configuration ---
# This is crucial for allowing your Vercel frontend to communicate with this backend.
origins = [
      # For local development
    "https://the-route-cause.vercel.app", # Replace with your actual frontend URL
    # You can add more origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- WebSocket Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New connection: {websocket.client}. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Connection closed: {websocket.client}. Total clients: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# --- API Endpoints ---

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    The main WebSocket endpoint for the AI agent to connect to.
    """
    await manager.connect(websocket)
    try:
        while True:
            # We are just keeping the connection alive. 
            # The AI agent (run_live_agent.py) will be sending data.
            # You can add logic here to receive messages from the frontend if needed.
            data = await websocket.receive_text()
            logger.info(f"Received message from client {websocket.client}: {data}")
            # Example of echoing back or broadcasting
            # await manager.broadcast(f"A client says: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.warning(f"Client {websocket.client} disconnected.")
    except Exception as e:
        logger.error(f"An error occurred with client {websocket.client}: {e}")
        if websocket in manager.active_connections:
            manager.disconnect(websocket)

@app.get("/api/health")
async def health_check():
    """
    A simple health check endpoint that Railway can use to verify the app is running.
    """
    return {"status": "ok"}

# --- Static File Serving (for a combined deployment) ---
# This part serves the built React app from the 'build' directory.
# Ensure your frontend build output is placed in `backend/build`.
# If you are deploying frontend and backend separately (Vercel + Railway), this is not strictly necessary but is good practice.

# Uncomment the following lines if you want to serve your React build from this backend
# try:
#     app.mount("/", StaticFiles(directory="build", html=True), name="static")
# except RuntimeError:
#     logger.info("Static files directory not found. Serving API only.")

# --- Server Startup Logic ---
if __name__ == "__main__":
    # Get the port from environment variables, defaulting to 8000
    port = int(os.environ.get("PORT", 8000))
    # Uvicorn is a lightning-fast ASGI server, perfect for production.
    # We use 0.0.0.0 to bind to all available network interfaces, which is required by Railway.
    uvicorn.run(app, host="0.0.0.0", port=port)
