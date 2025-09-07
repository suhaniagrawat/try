import os
import asyncio
import websockets
import logging
import time
from dotenv import load_dotenv

# Load environment variables from a .env file (for local development)
load_dotenv()

# --- Configuration ---
# Use environment variables for production, with sensible defaults for local dev.
# The WebSocket URL for the backend server.
BACKEND_URL = os.getenv("BACKEND_URL", "ws://localhost:8000/ws")
# Determines if any GUI should be shown (e.g., OpenCV windows). MUST be True in production.
IS_HEADLESS = os.getenv("HEADLESS", "true").lower() == "true"
# Source of the video to be processed.
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "my_video.mp4")
RECONNECT_DELAY = 5  # Delay in seconds before trying to reconnect

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def run_agent():
    """
    Main function to run the AI agent, connect to the WebSocket,
    and handle reconnection logic.
    """
    logger.info(f"Starting AI Agent. Headless mode: {IS_HEADLESS}")
    logger.info(f"Attempting to connect to WebSocket at: {BACKEND_URL}")

    while True: # Keep trying to connect forever
        try:
            async with websockets.connect(BACKEND_URL) as websocket:
                logger.info("Successfully connected to WebSocket server.")
                
                # --- YOUR AI AGENT LOGIC GOES HERE ---
                # This is where you would process your video, run your model, etc.
                # For demonstration, we'll just send a message every few seconds.
                
                counter = 0
                while True:
                    # Replace this with your actual agent's data
                    message = f"Agent sending data packet {counter} from video {VIDEO_SOURCE}"
                    
                    await websocket.send(message)
                    logger.info(f"Sent: {message}")
                    
                    counter += 1
                    await asyncio.sleep(2) # Simulate processing time

        except (websockets.exceptions.ConnectionClosedError, ConnectionRefusedError) as e:
            logger.error(f"Connection failed: {e}. Retrying in {RECONNECT_DELAY} seconds...")
            await asyncio.sleep(RECONNECT_DELAY)
        except Exception as e:
            logger.critical(f"An unexpected error occurred: {e}. Retrying in {RECONNECT_DELAY} seconds...")
            await asyncio.sleep(RECONNECT_DELAY)

if __name__ == "__main__":
    try:
        asyncio.run(run_agent())
    except KeyboardInterrupt:
        logger.info("Agent stopped by user.")
