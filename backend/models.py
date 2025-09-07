# in models.py
from pydantic import BaseModel, Field
from typing import Dict, Any

class DecisionPayload(BaseModel):
    timestamp: float
    lane_counts: Dict[str, int]
    pedestrian_count: int
    decision: Dict[str, str | int]
    signal_state: Dict[str, Any]


    # Example of the 'decision' dict:
    # {"action_code": 1, "reason": "AI Agent Decision: Green for lane_2"}