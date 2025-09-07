# in business_logic.py
import time
from state_manager import traffic_state
from models import DecisionPayload

# Dev A can keep this rule
MIN_GREEN_TIME_SECONDS = 10

def apply_ai_decision(data: DecisionPayload) -> bool:
    """
    Applies the AI's decision to the shared state and returns a boolean
    indicating if a broadcast to dashboards is needed.
    """
    # This comes from the run_live_agent.py script's output
    # It contains more detailed state info like the timer.
    traffic_state["signal_state"] = data.signal_state
    
    # We only need to process a full state change if the reason is not "Observing"
    if "Observing" in data.decision.get("reason", ""):
        # This is just a periodic update of lane counts, not a new decision.
        # We update the counts but don't trigger a full metric recalculation/broadcast.
        traffic_state["lane_counts"] = data.lane_counts
        traffic_state["pedestrian_count"] = data.pedestrian_count
        return False # No broadcast needed
    
    # --- A real decision is being made, process it ---

    # Rule: Enforce minimum green time (Dev A's logic)
    time_since_last_change = time.time() - traffic_state["last_update_time"]
    proposed_phase_from_reason = data.signal_state.get("active_direction") # More reliable
    current_phase_from_reason = traffic_state.get("signal_state", {}).get("active_direction")

    if proposed_phase_from_reason != current_phase_from_reason and time_since_last_change < MIN_GREEN_TIME_SECONDS:
        print(f"IGNORING AI: Tried to change light after {time_since_last_change:.1f}s.")
        # Update counts but reject the state change
        traffic_state["lane_counts"] = data.lane_counts
        return False # No broadcast needed

    # --- Rule passed, update the full state ---
    if proposed_phase_from_reason != current_phase_from_reason:
        traffic_state["last_update_time"] = time.time()

    traffic_state["last_decision_reason"] = data.decision.get("reason")
    traffic_state["lane_counts"] = data.lane_counts
    traffic_state["pedestrian_count"] = data.pedestrian_count
    
    print(f"STATE UPDATED: Reason -> {traffic_state['last_decision_reason']}")
    
    return True # Broadcast IS needed
