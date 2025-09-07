# main.py
# This single file contains the complete, optimized backend logic for speed and accuracy.

import asyncio
import time
from typing import List, Dict, Any
from collections import deque

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# in main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()

# IMPORTANT: Replace this with your REAL Vercel frontend URL
origins = [
    "https://my-traffic-dashboard.vercel.app", # Your frontend's public URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# ... your existing WebSocket and API endpoints continue here ...
# ... your existing WebSocket and API endpoints continue here ..


# ==============================================================================
# 1. PYDANTIC MODELS (Copied from models.py for a self-contained file)
# ==============================================================================

class DecisionPayload(BaseModel):
    """Defines the structure of the incoming data from the AI agent."""
    timestamp: float
    lane_counts: Dict[str, int]
    pedestrian_count: int
    decision: Dict[str, Any]
    signal_state: Dict[str, Any]


# ==============================================================================
# 2. STATE MANAGEMENT (Copied from state_manager.py for a self-contained file)
# ==============================================================================

# This dictionary acts as the single source of truth for the system's current state.
traffic_state = {
    "last_update_time": time.time(),
    "current_phase": "Unknown",
    "last_decision_reason": "System Initialized",
    "lane_counts": {"Northbound": 0, "Southbound": 0, "Eastbound": 0, "Westbound": 0},
    "pedestrian_count": 0,
    "signal_state": {},
    "ai_status": "DISCONNECTED"
}


# ==============================================================================
# 3. CONNECTION & METRICS LOGIC (Dev B's core logic, optimized)
# ==============================================================================

class ConnectionManager:
    """Manages WebSocket connections to all frontend dashboards."""
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"✅ Dashboard client connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"🔌 Dashboard client disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast(self, data: Dict[str, Any]):
        """Sends a JSON payload to all connected dashboard clients concurrently."""
        if self.active_connections:
            # asyncio.gather is highly efficient for sending to multiple clients at once.
            await asyncio.gather(*[connection.send_json(data) for connection in self.active_connections])

class MetricsProcessor:
    """
    Calculates and formats all frontend metrics. This class is designed to be
    extremely fast, performing only simple arithmetic and dictionary lookups.
    """
    def __init__(self, baseline_queue: float = 20.0):
        # Long-term state for performance metrics
        self.total_decisions = 0
        self.ai_decisions = 0
        self.emergency_events = 0
        self.queue_overrides = 0
        
        self.baseline_queues = baseline_queue
        # A deque is a highly efficient list-like object for appending and popping
        self.recent_queue_lengths = deque(maxlen=500) # Rolling average over the last 500 frames

    def update_long_term_metrics(self, decision_reason: str):
        """
        This method is called ONLY when a significant decision changes,
        not on every frame. This keeps metric counting accurate.
        """
        self.total_decisions += 1
        reason = decision_reason.upper()
        if "EMERGENCY" in reason:
            self.emergency_events += 1
        elif "AGENT" in reason:
            self.ai_decisions += 1
        elif "QUEUE" in reason or "STARVATION" in reason:
            self.queue_overrides += 1

    def get_full_payload(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the current state and returns the complete payload for the frontend.
        This is called on every frame and is optimized for speed.
        """
        # --- 1. Extract live data from the current state ---
        decision_reason = current_state.get("last_decision_reason", "N/A")
        lane_counts = current_state.get("lane_counts", {})
        signal_state = current_state.get("signal_state", {})
        
        # --- 2. Format the Main Dashboard data ---
        total_vehicles = sum(lane_counts.values())
        main_dashboard = {
            "signal_state": signal_state,
            "vehicle_counters": lane_counts,
            "total_vehicles": total_vehicles,
        }

        # --- 3. Calculate Performance Metrics ---
        # Update rolling queue average for a smooth graph on the frontend
        self.recent_queue_lengths.append(total_vehicles)
        avg_queue = sum(self.recent_queue_lengths) / len(self.recent_queue_lengths) if self.recent_queue_lengths else 0
        
        # Calculate efficiency and reduction percentages
        efficiency_pct = (self.ai_decisions / self.total_decisions) * 100 if self.total_decisions > 0 else 100
        reduction_pct = ((self.baseline_queues - avg_queue) / self.baseline_queues) * 100 if self.baseline_queues > 0 else 0
        
        # Determine status strings based on thresholds
        eff_status = "GOOD" if efficiency_pct >= 80 else "AVERAGE" if efficiency_pct >= 60 else "POOR"
        queue_status = "EXCELLENT" if reduction_pct > 25 else "AVERAGE" if reduction_pct >= 0 else "POOR"
        emergency_status = "EXCELLENT" if self.emergency_events > 0 else "NORMAL"

        performance_metrics = [
            {"title": "AI Decision Efficiency", "value": f"{efficiency_pct:.0f}%", "status": eff_status, "details": f"{self.ai_decisions} AI / {self.total_decisions} total"},
            {"title": "Queue Reduction", "value": f"{reduction_pct:.0f}%", "status": queue_status, "details": f"{avg_queue:.1f} avg vehicles (was {self.baseline_queues})"},
            {"title": "Emergency Response", "value": str(self.emergency_events), "status": emergency_status, "details": f"{self.emergency_events} responses"}
        ]

        # --- 4. Format the Emergency Mode data ---
        is_emergency = "EMERGENCY" in decision_reason.upper()
        emergency_mode = None
        if is_emergency:
            priority_direction = signal_state.get("active_direction")
            delayed_vehicles = sum(v for d, v in lane_counts.items() if d != priority_direction)
            emergency_mode = {
                "priority_direction": priority_direction,
                "delayed_vehicles": delayed_vehicles,
                "total_vehicles": total_vehicles
            }

        # --- 5. Combine into the final, complete payload ---
        return {
            "main_dashboard": main_dashboard,
            "performance_metrics": performance_metrics,
            "emergency_mode": emergency_mode
        }


# ==============================================================================
# 4. FASTAPI APPLICATION AND ENDPOINTS
# ==============================================================================

app = FastAPI(title="High-Speed AI Traffic Backend")

# --- Global instances for our logic handlers ---
dashboard_manager = ConnectionManager()
metrics_processor = MetricsProcessor()
last_significant_reason = "" # Used to track when a core decision changes

@app.websocket("/ws/ai")
async def websocket_ai_endpoint(websocket: WebSocket):
    """
    Receives data from AI agent, processes it, and broadcasts to dashboards instantly.
    This is the high-throughput core of the application.
    """
    global last_significant_reason
    await websocket.accept()
    traffic_state["ai_status"] = "CONNECTED"
    print("✅ AI Agent Connected")
    try:
        while True:
            data = await websocket.receive_json()
            try:
                # 1. Validate incoming data against our Pydantic model
                validated_data = DecisionPayload(**data)
                
                # 2. Instantly update the shared state dictionary. This is a very fast operation.
                traffic_state.update({
                    "last_update_time": validated_data.timestamp,
                    "current_phase": validated_data.signal_state.get("active_direction", "Unknown"),
                    "last_decision_reason": validated_data.decision.get("reason", "N/A"),
                    "lane_counts": validated_data.lane_counts,
                    "pedestrian_count": validated_data.pedestrian_count,
                    "signal_state": validated_data.signal_state
                })

                # 3. Check if the core decision reason has changed to update long-term metrics
                new_reason = traffic_state.get("last_decision_reason", "")
                if new_reason != last_significant_reason and "Observing" not in new_reason:
                    metrics_processor.update_long_term_metrics(new_reason)
                    last_significant_reason = new_reason # Update the last reason

                # 4. Get the full payload with freshly calculated metrics (also very fast)
                frontend_payload = metrics_processor.get_full_payload(traffic_state)
                
                # 5. Broadcast to all connected dashboards on every single frame
                await dashboard_manager.broadcast(frontend_payload)

            except ValidationError as e:
                print(f"❌ Invalid data from AI: {e}")
            except Exception as e:
                print(f"🚨 Unexpected error in AI endpoint: {e}")

    except WebSocketDisconnect:
        traffic_state["ai_status"] = "DISCONNECTED"
        print("🔴 AI Agent Disconnected")


@app.websocket("/ws/dashboard")
async def websocket_dashboard_endpoint(websocket: WebSocket):
    """Connects a frontend dashboard client and sends initial state."""
    await dashboard_manager.connect(websocket)
    try:
        # Send the most recent state immediately upon connection
        initial_payload = metrics_processor.get_full_payload(traffic_state)
        await websocket.send_json(initial_payload)
        # Keep the connection alive to receive pushed updates from the broadcast
        while True:
            await asyncio.sleep(3600) 
    except WebSocketDisconnect:
        dashboard_manager.disconnect(websocket)

