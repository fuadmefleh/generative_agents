from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from enum import Enum
from pydantic import validator
from datetime import datetime
from typing import List, Tuple, Set
import random



class Point2D(BaseModel):
    """2D point representation."""
    x: float
    y: float

    class Config:
        # make instances immutable and therefore hashable
        frozen = True

class Plan(BaseModel):
    """Structured plan for agent actions."""
    agent_id: str
    description: str
    start_time: datetime = Field(default_factory=datetime.utcnow)
    duration_minutes: int = Field(gt=0)
    location: str
    priority: int = Field(ge=1, le=5)
    sub_plans: List['Plan'] = Field(default_factory=list)
    completed: bool = False
    meta_data: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        # Ensure datetimes are serialized to ISO format for JSON transport
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

