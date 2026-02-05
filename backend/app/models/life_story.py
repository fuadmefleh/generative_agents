"""Models for agent life stories and personality development."""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from datetime import datetime


class EventCategory(str, Enum):
    """Categories of life events that shape personality."""
    FAMILY = "family"
    EDUCATION = "education"
    SOCIAL = "social"
    ACHIEVEMENT = "achievement"
    TRAUMA = "trauma"
    LOSS = "loss"
    RELATIONSHIP = "relationship"
    CAREER = "career"
    HEALTH = "health"
    ADVENTURE = "adventure"
    CREATIVE = "creative"
    FAILURE = "failure"
    DISCOVERY = "discovery"


class PersonalityImpact(BaseModel):
    """How an event impacts personality traits."""
    openness_delta: float = Field(default=0.0, ge=-0.3, le=0.3)
    conscientiousness_delta: float = Field(default=0.0, ge=-0.3, le=0.3)
    extraversion_delta: float = Field(default=0.0, ge=-0.3, le=0.3)
    agreeableness_delta: float = Field(default=0.0, ge=-0.3, le=0.3)
    neuroticism_delta: float = Field(default=0.0, ge=-0.3, le=0.3)
    
    # Behavioral impacts
    work_ethic_delta: float = Field(default=0.0, ge=-0.2, le=0.2)
    spontaneity_delta: float = Field(default=0.0, ge=-0.2, le=0.2)
    
    def has_impact(self) -> bool:
        """Check if this impact changes any traits."""
        return any([
            abs(self.openness_delta) > 0.01,
            abs(self.conscientiousness_delta) > 0.01,
            abs(self.extraversion_delta) > 0.01,
            abs(self.agreeableness_delta) > 0.01,
            abs(self.neuroticism_delta) > 0.01,
            abs(self.work_ethic_delta) > 0.01,
            abs(self.spontaneity_delta) > 0.01
        ])


class LifeEvent(BaseModel):
    """A significant event in an agent's life."""
    age: int = Field(ge=4, le=100)
    category: EventCategory
    description: str = Field(min_length=10, max_length=500)
    emotional_impact: float = Field(ge=-1.0, le=1.0, default=0.0)  # -1 very negative, +1 very positive
    personality_impact: PersonalityImpact = Field(default_factory=PersonalityImpact)
    
    # Additional context
    involved_people: List[str] = Field(default_factory=list)  # Names of people involved
    location: Optional[str] = None
    lasting_effects: List[str] = Field(default_factory=list)  # Long-term consequences
    

class LifeStory(BaseModel):
    """Complete life story of an agent with personality-shaping events."""
    agent_name: str
    current_age: int = Field(ge=18, le=100)
    birthplace: str
    
    # Generated events (2-3 per age period)
    events: List[LifeEvent] = Field(default_factory=list)
    
    # Core identity formed from events
    core_values: List[str] = Field(default_factory=list)  # e.g., "family", "independence", "creativity"
    fears: List[str] = Field(default_factory=list)  # Deep-seated fears from events
    aspirations: List[str] = Field(default_factory=list)  # What they want to achieve
    
    # Relationships formed through life
    important_people: Dict[str, str] = Field(default_factory=dict)  # name -> relationship type
    
    def get_events_by_age_range(self, min_age: int, max_age: int) -> List[LifeEvent]:
        """Get events that occurred within an age range."""
        return [e for e in self.events if min_age <= e.age <= max_age]
    
    def get_events_by_category(self, category: EventCategory) -> List[LifeEvent]:
        """Get all events of a specific category."""
        return [e for e in self.events if e.category == category]
    
    def get_total_personality_impact(self) -> PersonalityImpact:
        """Calculate cumulative personality impact from all events."""
        total = PersonalityImpact()
        for event in self.events:
            total.openness_delta += event.personality_impact.openness_delta
            total.conscientiousness_delta += event.personality_impact.conscientiousness_delta
            total.extraversion_delta += event.personality_impact.extraversion_delta
            total.agreeableness_delta += event.personality_impact.agreeableness_delta
            total.neuroticism_delta += event.personality_impact.neuroticism_delta
            total.work_ethic_delta += event.personality_impact.work_ethic_delta
            total.spontaneity_delta += event.personality_impact.spontaneity_delta
        return total
    
    def get_narrative_summary(self) -> str:
        """Generate a natural language summary of the life story."""
        summary_parts = []
        
        summary_parts.append(f"{self.agent_name} was born in {self.birthplace}.")
        
        # Childhood (4-12)
        childhood_events = self.get_events_by_age_range(4, 12)
        if childhood_events:
            summary_parts.append(f"During childhood, {self.agent_name} experienced: " + 
                                 ", ".join([e.description[:80] for e in childhood_events[:2]]) + ".")
        
        # Adolescence (13-19)
        teen_events = self.get_events_by_age_range(13, 19)
        if teen_events:
            summary_parts.append(f"As a teenager, " + 
                                 ", ".join([e.description[:80] for e in teen_events[:2]]) + ".")
        
        # Adulthood
        adult_events = self.get_events_by_age_range(20, self.current_age)
        if adult_events:
            summary_parts.append(f"In adulthood, " + 
                                 ", ".join([e.description[:80] for e in adult_events[:3]]) + ".")
        
        # Core identity
        if self.core_values:
            summary_parts.append(f"They deeply value {', '.join(self.core_values[:3])}.")
        
        return " ".join(summary_parts)


class LifeStoryPrompt(BaseModel):
    """Structured prompt for generating life story events."""
    agent_name: str
    current_age: int
    starting_age: int = 4
    events_per_period: int = Field(default=2, ge=1, le=3)
    personality_seed: Optional[Dict[str, float]] = None  # Optional initial personality to influence story
    occupation: Optional[str] = None
    cultural_background: Optional[str] = None
    

class GeneratedLifeStoryResponse(BaseModel):
    """LLM response for generated life story."""
    birthplace: str
    events: List[LifeEvent]
    core_values: List[str] = Field(min_items=2, max_items=5)
    fears: List[str] = Field(min_items=1, max_items=4)
    aspirations: List[str] = Field(min_items=2, max_items=4)
    important_people: Dict[str, str] = Field(default_factory=dict)
