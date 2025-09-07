import cv2
import numpy as np
import time
import json
import os
import sys
from ultralytics import YOLO
import asyncio
import websockets
from q_learning_agent import AdaptiveQLearningAgent
from optimization_engine import OptimizationEngine
import threading
import queue
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =================================================================================
# === CONFIGURATION                                                             ===
# =================================================================================
VIDEO_FILE = os.getenv("VIDEO_SOURCE", "my_video.mp4")  # Can be file path or camera index
SAVED_AGENT_MODEL_PATH = "traffic_agent"
LANE_POLYGONS = {
    "Northbound": np.array([[2124, 487], [2830, 514], [2103, 1657], [2829, 1592]], np.int32),
    "Southbound": np.array([[966, 1568], [1380, 1574], [1467, 2048], [830, 2085]], np.int32),
    "Eastbound": np.array([[0,0], [1,1], [2,2], [3,3]], np.int32),
    "Westbound": np.array([[0,0], [1,1], [2,2], [3,3]], np.int32),
}
LANE_NAMES_ORDER = ["Northbound", "Southbound", "Eastbound", "Westbound"] 
CROSSWALK_POLYGONS = { "crosswalk_1": np.array([[1900, 1000], [2100, 1000], [2100, 1200], [1900, 1200]], np.int32) }
TRAFFIC_LIGHT_POSITIONS = { "Northbound": (150, 250), "Southbound": (150, 450), "Eastbound": (150, 650), "Westbound": (150, 850) }
GREEN_LIGHT_DURATION = 8.0; YELLOW_LIGHT_DURATION = 1.5; EMERGENCY_GREEN_DURATION = 5.0 
EMERGENCY_CLEARING_TIME = 2.0; STARVATION_THRESHOLD = 30.0; MAX_QUEUE_LENGTH = 30
PEDESTRIAN_THRESHOLD = 10; PEDESTRIAN_CROSSING_DURATION = 15
YOLO_MODEL = 'yolov8s.pt'; CONF_THRESHOLD = 0.3
PERSON_CLASS_ID = 0; VEHICLE_CLASSES = [2, 3, 5, 7]
ALL_DETECTABLE_CLASSES = [PERSON_CLASS_ID] + VEHICLE_CLASSES
MAX_VEHICLES_PER_LANE = 40

# Production WebSocket URL configuration
WEBSOCKET_URI = os.getenv("BACKEND_WEBSOCKET_URL", "ws://localhost:8000/ws/ai")

# Check if we're in a production environment (Railway)
IS_PRODUCTION = os.getenv("RAILWAY_ENVIRONMENT") is not None
HEADLESS_MODE = os.getenv("HEADLESS", "false").lower() == "true" or IS_PRODUCTION

logger.info(f"🔧 Configuration loaded:")
logger.info(f"   - Video source: {VIDEO_FILE}")
logger.info(f"   - WebSocket URI: {WEBSOCKET_URI}")
logger.info(f"   - Production mode: {IS_PRODUCTION}")
logger.info(f"   - Headless mode: {HEADLESS_MODE}")

# =================================================================================
# === NETWORKING THREAD                                                         ===
# =================================================================================
def websocket_thread(data_queue: queue.Queue, stop_event: threading.Event):
    """
    Handles all asynchronous WebSocket communication in a dedicated thread.
    """
    async def sender():
        retry_count = 0
        max_retries = 10
        
        while not stop_event.is_set() and retry_count < max_retries:
            try:
                logger.info(f"🔌 Attempting to connect to {WEBSOCKET_URI} (attempt {retry_count + 1})")
                async with websockets.connect(WEBSOCKET_URI, ping_interval=20, ping_timeout=10) as websocket:
                    logger.info("✅ Successfully connected to backend WebSocket.")
                    retry_count = 0  # Reset retry count on successful connection
                    
                    while not stop_event.is_set():
                        try:
                            # Get data from the queue without blocking the event loop
                            output_data = data_queue.get_nowait()
                            if output_data:
                                await websocket.send(json.dumps(output_data))
                        except queue.Empty:
                            await asyncio.sleep(0.01)  # Yield control
                        except websockets.exceptions.ConnectionClosed:
                            logger.warning("🔴 Backend connection was lost.")
                            break
                        except Exception as e:
                            logger.error(f"🔴 Error sending data: {e}")
                            break
                            
            except Exception as e:
                retry_count += 1
                logger.error(f"🔴 WebSocket connection failed: {e}")
                if retry_count < max_retries:
                    wait_time = min(retry_count * 2, 30)  # Exponential backoff, max 30s
                    logger.info(f"⏳ Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("❌ Max retries reached. Stopping network thread.")
                    break
        
        logger.info("🔴 Network thread shutting down.")
        stop_event.set()

    asyncio.run(sender())

# =================================================================================
# === MAIN SCRIPT (GUI and AI Logic)                                            ===
# =================================================================================
def run_live_inference(data_queue: queue.Queue, stop_event: threading.Event):
    """
    Handles all OpenCV, YOLO, and AI logic in the main thread.
    """
    def draw_single_traffic_light(frame, position, status):
        x, y = position; radius = 35; color = (0, 0, 255)
        if status == "GREEN": color = (0, 255, 0)
        elif status == "YELLOW": color = (0, 255, 255)
        cv2.circle(frame, (x, y), radius, color, -1); cv2.circle(frame, (x, y), radius, (50, 50, 50), 3)

    # Initialize AI components
    logger.info("🧠 Initializing AI components...")
    try:
        agent = AdaptiveQLearningAgent(action_size=4)
        if os.path.exists(SAVED_AGENT_MODEL_PATH):
            agent.load_model(SAVED_AGENT_MODEL_PATH)
            logger.info(f"✅ Loaded agent model from {SAVED_AGENT_MODEL_PATH}")
        else:
            logger.warning(f"⚠️ Agent model not found at {SAVED_AGENT_MODEL_PATH}, using default")
        
        engine = OptimizationEngine(starvation_threshold=STARVATION_THRESHOLD)
        model = YOLO(YOLO_MODEL)
        logger.info("✅ AI components initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize AI components: {e}")
        stop_event.set()
        return

    # Initialize video capture
    logger.info(f"📹 Opening video source: {VIDEO_FILE}")
    try:
        # Try to convert to int (for camera index), otherwise use as string (file path)
        try:
            video_source = int(VIDEO_FILE)
        except ValueError:
            video_source = VIDEO_FILE
            
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            logger.error(f"❌ Could not open video source '{VIDEO_FILE}'")
            stop_event.set()
            return
        logger.info("✅ Video source opened successfully")
    except Exception as e:
        logger.error(f"❌ Failed to open video source: {e}")
        stop_event.set()
        return

    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    
    # Initialize traffic state
    signal_state = "GREEN"; current_green_lane_index = 0; next_green_lane_index = -1
    state_timer = 0.0; pedestrian_crossing_timer = 0
    emergency_override_state = None; lane_to_clear_index = -1; emergency_target_lane = -1
    
    logger.info(f"🚦 Starting traffic control loop (FPS: {fps}, Headless: {HEADLESS_MODE})")
    frame_count = 0
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            logger.warning("⚠️ Failed to read frame, restarting video...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1
        if frame_count % 100 == 0:  # Log every 100 frames
            logger.info(f"📊 Processed {frame_count} frames")

        # --- Object Detection & State Machine Logic ---
        try:
            results = model(frame, verbose=False)
            lane_counts = {name: 0 for name in LANE_NAMES_ORDER}; pedestrian_count = 0
            
            for box in results[0].boxes:
                class_id = int(box.cls[0].item())
                if class_id not in ALL_DETECTABLE_CLASSES or box.conf[0].item() < CONF_THRESHOLD: continue
                x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0].tolist()]; center_point = ((x1 + x2) // 2, (y1 + y2) // 2)
                if class_id == PERSON_CLASS_ID:
                    for poly in CROSSWALK_POLYGONS.values():
                        if cv2.pointPolygonTest(poly, center_point, False) >= 0: 
                            pedestrian_count += 1
                            if not HEADLESS_MODE:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                            break
                else:
                    for name, poly in LANE_POLYGONS.items():
                        if cv2.pointPolygonTest(poly, center_point, False) >= 0: 
                            lane_counts[name] += 1
                            if not HEADLESS_MODE:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            break
        except Exception as e:
            logger.error(f"🔴 Error in object detection: {e}")
            continue
        
        # Handle keyboard input only in non-headless mode
        manual_emergency_lane = -1
        if not HEADLESS_MODE:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('1'): manual_emergency_lane = 0
            elif key == ord('2'): manual_emergency_lane = 1
            elif key == ord('3'): manual_emergency_lane = 2
            elif key == ord('4'): manual_emergency_lane = 3
            if key == ord('q'): 
                logger.info("🛑 Quit command received")
                stop_event.set()
                break
        
        # Traffic control logic (unchanged)
        state_timer += 1/fps; decision_reason, send_update_to_backend = "Observing", False
        
        if manual_emergency_lane != -1 and emergency_override_state is None:
            if not (current_green_lane_index == manual_emergency_lane and signal_state == "GREEN"):
                emergency_override_state, lane_to_clear_index, state_timer, emergency_target_lane = "CLEARING_YELLOW", current_green_lane_index, 0.0, manual_emergency_lane
                decision_reason, send_update_to_backend = f"MANUAL EMERGENCY: Clearing for {LANE_NAMES_ORDER[manual_emergency_lane]}", True
        if emergency_override_state is not None:
            if emergency_override_state == "CLEARING_YELLOW":
                signal_state = "YELLOW"
                if state_timer >= YELLOW_LIGHT_DURATION: emergency_override_state, state_timer, decision_reason, send_update_to_backend = "CLEARING_ALL_RED", 0.0, "EMERGENCY: All Red", True
            elif emergency_override_state == "CLEARING_ALL_RED":
                signal_state = "ALL_RED"
                if state_timer >= EMERGENCY_CLEARING_TIME: emergency_override_state, current_green_lane_index, signal_state, state_timer, decision_reason, send_update_to_backend = "ACTIVE", emergency_target_lane, "GREEN", 0.0, f"EMERGENCY: Green for {LANE_NAMES_ORDER[emergency_target_lane]}", True
            elif emergency_override_state == "ACTIVE" and state_timer >= EMERGENCY_GREEN_DURATION:
                emergency_override_state, signal_state, state_timer, decision_reason, send_update_to_backend = None, "YELLOW", 0.0, "Emergency cleared", True
        else:
            if signal_state == "GREEN" and (state_timer >= GREEN_LIGHT_DURATION or any(lane_counts[name] >= MAX_QUEUE_LENGTH for i, name in enumerate(LANE_NAMES_ORDER) if i != current_green_lane_index)):
                signal_state, state_timer = "YELLOW", 0.0
            elif signal_state == "YELLOW" and state_timer >= YELLOW_LIGHT_DURATION:
                counts_ordered = [lane_counts[n] for n in LANE_NAMES_ORDER]; norm_counts = [c/MAX_VEHICLES_PER_LANE for c in counts_ordered]; observation = np.array(norm_counts + [current_green_lane_index/3, 0])
                agent_recommendation = agent.choose_action(observation, training=False)
                context = {'emergency_active': False, 'emergency_lane': None, 'pedestrian_count': pedestrian_count}
                final_action, decision_reason, _ = engine.optimize_action(lane_counts, LANE_NAMES_ORDER[current_green_lane_index], agent_recommendation, context)
                current_green_lane_index, signal_state, state_timer, send_update_to_backend = final_action, "GREEN", 0.0, True
                engine.last_service_times[LANE_NAMES_ORDER[current_green_lane_index]] = time.time(); engine.update_performance(final_action, -sum(lane_counts.values()), False, context)
        
        # Prepare output data for backend
        output_data = {
            "timestamp": time.time(), 
            "lane_counts": lane_counts, 
            "pedestrian_count": pedestrian_count, 
            "decision": {"reason": decision_reason}, 
            "signal_state": {
                "active_direction": LANE_NAMES_ORDER[current_green_lane_index] if signal_state not in ["PEDESTRIAN", "ALL_RED"] else signal_state, 
                "state": signal_state, 
                "timer": int(state_timer)
            }
        }
        data_queue.put(output_data)

        # Visualization (only in non-headless mode)
        if not HEADLESS_MODE:
            try:
                # --- FULL VISUALIZATION PANEL ---
                overlay = frame.copy(); cv2.rectangle(overlay, (frame.shape[1] - 700, 0), (frame.shape[1], 800), (0, 0, 0), -1)
                alpha = 0.6; frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                
                for i, (name, pos) in enumerate(TRAFFIC_LIGHT_POSITIONS.items()):
                    status = "RED"
                    if emergency_override_state == "CLEARING_YELLOW" and i == lane_to_clear_index: status = "YELLOW"
                    elif emergency_override_state == "CLEARING_ALL_RED": status = "RED"
                    elif emergency_override_state == "ACTIVE" and i == current_green_lane_index: status = "GREEN"
                    elif i == current_green_lane_index and emergency_override_state is None: status = signal_state
                    draw_single_traffic_light(frame, pos, status)
                    cv2.putText(frame, name, (pos[0] + 50, pos[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)

                y_pos, x_pos = 50, frame.shape[1] - 650
                status_text = f"{LANE_NAMES_ORDER[current_green_lane_index]}: {signal_state}"
                if emergency_override_state is not None: status_text = f"EMERGENCY OVERRIDE ({emergency_override_state})"
                
                cv2.putText(frame, "SYSTEM STATUS", (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3); y_pos += 60
                cv2.putText(frame, f"Current State: {status_text}", (x_pos, y_
