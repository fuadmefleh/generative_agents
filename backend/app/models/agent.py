from enum import Enum
from app.llm.client import OllamaClient, StructuredPromptBuilder
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
from datetime import datetime, timedelta
from pydantic import validator
import random
import math
from app.models.models import Point2D, Plan
from loguru import logger

# Import Sims-like behavior system
from app.models.sims_behavior import SimsBehaviorManager, SimsAction, NeedType

if TYPE_CHECKING:
    from app.models.life_story import LifeStory
    from app.models.conversation import ConversationManager, InnerLifeManager


class ActionType(str, Enum):
    """Types of actions agents can perform."""
    MOVE = "move"
    INTERACT = "interact"
    CONVERSE = "converse"
    USE_OBJECT = "use_object"
    WAIT = "wait"
    REFLECT = "reflect"
    WORK = "work"
    LEISURE = "leisure"
    EXERCISE = "exercise"
    ATTEND_CLASS = "attend_class"
    STUDY = "study"
    
    # Detailed mundane activities
    SLEEP = "sleep"
    WAKE_UP = "wake_up"
    USE_TOILET = "use_toilet"
    SHOWER = "shower"
    BRUSH_TEETH = "brush_teeth"
    GET_DRESSED = "get_dressed"
    
    # Kitchen/eating activities
    COOK = "cook"
    EAT = "eat"
    DRINK = "drink"
    LOOK_IN_FRIDGE = "look_in_fridge"
    LOOK_IN_CUPBOARD = "look_in_cupboard"
    WASH_DISHES = "wash_dishes"
    MAKE_COFFEE = "make_coffee"
    
    # Living activities
    WATCH_TV = "watch_tv"
    READ_BOOK = "read_book"
    LISTEN_MUSIC = "listen_music"
    USE_COMPUTER = "use_computer"
    CLEAN_ROOM = "clean_room"
    DO_LAUNDRY = "do_laundry"
    
    # Social activities
    MAKE_PHONE_CALL = "make_phone_call"
    CHAT = "chat"
    
    # Outdoor activities
    WALK = "walk"
    JOG = "jog"
    SIT_ON_BENCH = "sit_on_bench"


class JobType(str, Enum):
    """Types of jobs/occupations agents can have."""
    BARISTA = "barista"
    ARTIST = "artist"
    STUDENT = "student"
    TEACHER = "teacher"
    CHEF = "chef"
    ORGANIZER = "organizer"
    WRITER = "writer"
    MUSICIAN = "musician"
    UNEMPLOYED = "unemployed"


class ScheduledActivity(BaseModel):
    """A scheduled activity for an agent's daily routine."""
    start_hour: int = Field(ge=0, le=23)
    start_minute: int = Field(ge=0, le=59, default=0)  # Minute precision for activities
    end_hour: int = Field(ge=0, le=23)
    end_minute: int = Field(ge=0, le=59, default=0)
    activity_type: ActionType
    location: str
    sub_location: Optional[str] = None
    description: str
    priority: int = Field(ge=1, le=10, default=7)
    days_of_week: List[int] = Field(default_factory=lambda: list(range(7)))  # 0=Monday, 6=Sunday
    
    # Detailed activity tracking
    specific_objects: List[str] = Field(default_factory=list)  # Objects used (bed, fridge, toilet, etc.)
    sub_activities: List[ActionType] = Field(default_factory=list)  # Sub-steps (e.g., shower -> brush_teeth)
    
    def is_active_at(self, hour: int, minute: int, day_of_week: int) -> bool:
        """Check if this activity is active at the given time."""
        if day_of_week not in self.days_of_week:
            return False
        
        start_minutes = self.start_hour * 60 + self.start_minute
        end_minutes = self.end_hour * 60 + self.end_minute
        current_minutes = hour * 60 + minute
        
        if start_minutes <= end_minutes:
            return start_minutes <= current_minutes < end_minutes
        else:  # Activity crosses midnight
            return current_minutes >= start_minutes or current_minutes < end_minutes
    
    def get_duration_minutes(self) -> int:
        """Get the duration of this activity in minutes."""
        start_minutes = self.start_hour * 60 + self.start_minute
        end_minutes = self.end_hour * 60 + self.end_minute
        
        if end_minutes >= start_minutes:
            return end_minutes - start_minutes
        else:  # Crosses midnight
            return (24 * 60) - start_minutes + end_minutes


class AgentJob(BaseModel):
    """Information about an agent's job or occupation."""
    job_type: JobType
    workplace_location: str
    workplace_sub_location: Optional[str] = None
    work_schedule: List[ScheduledActivity] = Field(default_factory=list)
    salary: Optional[float] = None
    job_satisfaction: float = Field(default=0.7, ge=0.0, le=1.0)
    skills: List[str] = Field(default_factory=list)
    
    def is_work_time(self, current_hour: int, day_of_week: int) -> bool:
        """Check if the current time is during work hours."""
        for schedule in self.work_schedule:
            if day_of_week in schedule.days_of_week:
                if schedule.start_hour <= schedule.end_hour:
                    if schedule.start_hour <= current_hour < schedule.end_hour:
                        return True
                else:  # Crosses midnight
                    if current_hour >= schedule.start_hour or current_hour < schedule.end_hour:
                        return True
        return False
    
    def get_work_activity(self, current_hour: int, day_of_week: int) -> Optional[ScheduledActivity]:
        """Get the current work activity if it's work time."""
        for schedule in self.work_schedule:
            if day_of_week in schedule.days_of_week:
                if schedule.start_hour <= schedule.end_hour:
                    if schedule.start_hour <= current_hour < schedule.end_hour:
                        return schedule
                else:
                    if current_hour >= schedule.start_hour or current_hour < schedule.end_hour:
                        return schedule
        return None


class EmotionType(str, Enum):
    """Types of emotions an agent can experience."""
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    ANXIOUS = "anxious"
    EXCITED = "excited"
    BORED = "bored"
    CONTENT = "content"
    STRESSED = "stressed"
    LONELY = "lonely"
    NEUTRAL = "neutral"


class MemoryType(str, Enum):
    """Types of memories agents can form."""
    OBSERVATION = "observation"
    SOCIAL = "social"
    EMOTION = "emotion"
    ACHIEVEMENT = "achievement"
    REFLECTION = "reflection"


class AgentMemoryEntry(BaseModel):
    """A single memory entry with emotional weight and decay."""
    id: str
    timestamp: datetime
    memory_type: MemoryType
    content: str
    importance: float = Field(ge=0.0, le=10.0, default=5.0)
    emotional_valence: float = Field(ge=-1.0, le=1.0, default=0.0)  # -1 = negative, 1 = positive
    related_agents: List[str] = Field(default_factory=list)
    related_location: Optional[str] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    def get_recency_weight(self, current_time: datetime) -> float:
        """Calculate memory weight based on recency (exponential decay)."""
        hours_ago = (current_time - self.timestamp).total_seconds() / 3600
        
        # Clamp to prevent overflow and handle edge cases
        if hours_ago < 0:
            # Memory from the future shouldn't happen, but treat as very recent
            return 1.0
        elif hours_ago > 700:  # Cap at ~1 month to prevent overflow
            return 0.0
        
        return math.exp(-0.1 * hours_ago)  # Decays over ~10 hours significantly

    def get_retrieval_score(self, current_time: datetime, query_relevance: float = 1.0) -> float:
        """Calculate overall retrieval score for memory search."""
        recency = self.get_recency_weight(current_time)
        importance_weight = self.importance / 10.0
        access_boost = min(0.3, self.access_count * 0.05)  # Frequently accessed memories are easier to recall
        return (recency * 0.4 + importance_weight * 0.4 + query_relevance * 0.2) + access_boost

class ActionDecision(BaseModel):
    """AI-generated action decision."""
    action_type: ActionType
    reasoning: str
    target_location: str
    target_sub_location: Optional[str] = None
    target_object: Optional[str] = None
    target_agent: Optional[str] = None
    priority: int = Field(ge=1, le=10, default=5)  # 1=low, 10=critical
    expected_duration_minutes: int = Field(ge=1, le=480, default=30)  # How long this action takes
    inner_thought: Optional[str] = None  # What the agent is thinking


class AgentEmotionalState(BaseModel):
    """Tracks the emotional state of an agent."""
    primary_emotion: EmotionType = EmotionType.NEUTRAL
    emotion_intensity: float = Field(ge=0.0, le=1.0, default=0.5)
    mood_baseline: float = Field(ge=-1.0, le=1.0, default=0.0)  # Long-term mood (-1 depressed, 1 elated)
    stress_level: float = Field(ge=0.0, le=1.0, default=0.2)
    recent_emotional_events: List[Tuple[EmotionType, float, str]] = Field(default_factory=list)  # (emotion, intensity, cause)
    
    def update_emotion(self, emotion: EmotionType, intensity: float, cause: str):
        """Update current emotional state based on an event."""
        # Blend with current emotion
        if intensity > self.emotion_intensity:
            self.primary_emotion = emotion
            self.emotion_intensity = intensity
        else:
            # Gradual decay towards new emotion
            self.emotion_intensity = self.emotion_intensity * 0.8 + intensity * 0.2
        
        # Track recent events
        self.recent_emotional_events.append((emotion, intensity, cause))
        self.recent_emotional_events = self.recent_emotional_events[-10:]  # Keep last 10
        
        # Update mood baseline slowly
        valence = 1.0 if emotion in [EmotionType.HAPPY, EmotionType.EXCITED, EmotionType.CONTENT] else \
                  -1.0 if emotion in [EmotionType.SAD, EmotionType.ANGRY, EmotionType.ANXIOUS] else 0.0
        self.mood_baseline = self.mood_baseline * 0.95 + valence * intensity * 0.05
        self.mood_baseline = max(-1.0, min(1.0, self.mood_baseline))
    
    def decay_over_time(self):
        """Natural decay of emotional intensity over time."""
        self.emotion_intensity *= 0.98
        if self.emotion_intensity < 0.2:
            self.primary_emotion = EmotionType.NEUTRAL if self.mood_baseline >= -0.3 else EmotionType.SAD
        self.stress_level = max(0.0, self.stress_level * 0.99 - 0.01)
    
    def get_mood_description(self) -> str:
        """Get a natural language description of current mood."""
        intensity_words = {
            (0.0, 0.3): "slightly",
            (0.3, 0.6): "somewhat",
            (0.6, 0.8): "quite",
            (0.8, 1.0): "very"
        }
        intensity_word = "slightly"
        for (low, high), word in intensity_words.items():
            if low <= self.emotion_intensity < high:
                intensity_word = word
                break
        
        emotion_descriptions = {
            EmotionType.HAPPY: "feeling good",
            EmotionType.SAD: "feeling down",
            EmotionType.ANGRY: "frustrated",
            EmotionType.ANXIOUS: "worried",
            EmotionType.EXCITED: "excited",
            EmotionType.BORED: "bored",
            EmotionType.CONTENT: "at peace",
            EmotionType.STRESSED: "stressed",
            EmotionType.LONELY: "lonely",
            EmotionType.NEUTRAL: "calm"
        }
        
        mood_add = ""
        if self.mood_baseline < -0.5:
            mood_add = " (going through a difficult time)"
        elif self.mood_baseline > 0.5:
            mood_add = " (in a good phase of life)"
        
        return f"{intensity_word} {emotion_descriptions.get(self.primary_emotion, 'neutral')}{mood_add}"

class AgentMemory(BaseModel):
    visited_locations: set[Point2D] = Field(default_factory=set)
    current_path: Optional[list[Point2D]] = None  # Immutable sequence of points
    known_object_positions: Dict[Tuple[str, str, str], List[Point2D]] = Field(default_factory=dict)  # (Sector Name, Sub Location Name, Object ID) to position mapping
    
    # Enhanced memory system
    episodic_memories: List[AgentMemoryEntry] = Field(default_factory=list)
    max_memories: int = 200
    daily_observations: List[str] = Field(default_factory=list)  # What happened today
    
    # Relationship memory
    relationship_memories: Dict[str, List[str]] = Field(default_factory=dict)  # agent_name -> list of interaction memories
    
    # Routine learning
    learned_routines: Dict[int, List[str]] = Field(default_factory=dict)  # hour -> typical actions
    
    # Goals and plans
    short_term_goals: List[str] = Field(default_factory=list)
    long_term_goals: List[str] = Field(default_factory=list)

    def recall_object(self, sector_name: str, sub_location_name: str, object_id: str) -> Optional[List[Point2D]]:
        """Recall known object by sector, sub-location, and ID."""
        # If sub_location_name is None, return all known positions for the object in the sector
        if sub_location_name is None:
            return [pos for (sec, sub, obj), pos in self.known_object_positions.items() if sec == sector_name and obj == object_id]
        return self.known_object_positions.get((sector_name, sub_location_name, object_id), None)

    def remember_object(self, sector_name: str, sub_location_name: str, object_id: str, position: Point2D):
        """Store known object by sector, sub-location, and ID."""
        if self.known_object_positions.get((sector_name, sub_location_name, object_id)) is None:
            self.known_object_positions[(sector_name, sub_location_name, object_id)] = []
        
        # Avoid duplicates
        if position not in self.known_object_positions[(sector_name, sub_location_name, object_id)]:
            self.known_object_positions[(sector_name, sub_location_name, object_id)].append(position)
    
    def add_memory(self, memory: AgentMemoryEntry):
        """Add a new episodic memory with importance-based retention."""
        self.episodic_memories.append(memory)
        
        # If over capacity, remove least important old memories
        if len(self.episodic_memories) > self.max_memories:
            # Sort by importance and recency, keep the most valuable
            current_time = datetime.utcnow()
            self.episodic_memories.sort(
                key=lambda m: m.get_retrieval_score(current_time),
                reverse=True
            )
            self.episodic_memories = self.episodic_memories[:self.max_memories]
    
    def recall_memories(self, current_time: datetime, query: str = "", limit: int = 10, 
                        memory_type: Optional[MemoryType] = None,
                        related_agent: Optional[str] = None) -> List[AgentMemoryEntry]:
        """Retrieve relevant memories based on recency, importance, and optional filters."""
        filtered = self.episodic_memories
        
        if memory_type:
            filtered = [m for m in filtered if m.memory_type == memory_type]
        
        if related_agent:
            filtered = [m for m in filtered if related_agent in m.related_agents]
        
        # Score and sort
        scored = [(m, m.get_retrieval_score(current_time)) for m in filtered]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Mark as accessed
        result = []
        for m, score in scored[:limit]:
            m.access_count += 1
            m.last_accessed = current_time
            result.append(m)
        
        return result
    
    def add_relationship_memory(self, agent_name: str, memory: str):
        """Add a memory about an interaction with another agent."""
        if agent_name not in self.relationship_memories:
            self.relationship_memories[agent_name] = []
        self.relationship_memories[agent_name].append(memory)
        # Keep last 20 memories per person
        self.relationship_memories[agent_name] = self.relationship_memories[agent_name][-20:]
    
    def get_relationship_history(self, agent_name: str) -> List[str]:
        """Get memory history with a specific agent."""
        return self.relationship_memories.get(agent_name, [])
    
    def record_routine_action(self, hour: int, action: str):
        """Learn what actions are typical at certain hours."""
        if hour not in self.learned_routines:
            self.learned_routines[hour] = []
        self.learned_routines[hour].append(action)
        # Keep last 7 days of patterns per hour
        self.learned_routines[hour] = self.learned_routines[hour][-7:]
    
    def get_typical_actions_for_hour(self, hour: int) -> List[str]:
        """Get commonly performed actions for a specific hour."""
        return self.learned_routines.get(hour, [])
    
    def summarize_day(self) -> str:
        """Create a summary of what happened today."""
        if not self.daily_observations:
            return "Nothing notable happened today."
        return "; ".join(self.daily_observations[-10:])
    
    def clear_daily_observations(self):
        """Reset daily observations (call at midnight)."""
        self.daily_observations = []

class AgentNeeds(BaseModel):
    energy: float = Field(default=100.0, ge=0.0, le=100.0)
    hunger: float = Field(default=0.0, ge=0.0, le=100.0)
    bladder: float = Field(default=0.0, ge=0.0, le=100.0)
    social: float = Field(default=100.0, ge=0.0, le=100.0)
    fun: float = Field(default=100.0, ge=0.0, le=100.0)
    hygiene: float = Field(default=100.0, ge=0.0, le=100.0)
    
    # New human-like needs
    comfort: float = Field(default=80.0, ge=0.0, le=100.0)  # Physical comfort
    fulfillment: float = Field(default=70.0, ge=0.0, le=100.0)  # Sense of purpose/achievement
    safety: float = Field(default=100.0, ge=0.0, le=100.0)  # Feeling of security

    # Decay rates per tick (adjusted for more natural pacing)
    ENERGY_DECAY_RATE: float = 0.4
    HUNGER_INCREASE_RATE: float = 0.25
    BLADDER_INCREASE_RATE: float = 0.2
    SOCIAL_DECAY_RATE: float = 0.15
    FUN_DECAY_RATE: float = 0.12
    HYGIENE_DECAY_RATE: float = 0.08
    COMFORT_DECAY_RATE: float = 0.1
    FULFILLMENT_DECAY_RATE: float = 0.05
    
    # Time-based modifiers (needs change differently based on time of day)
    def get_time_modifier(self, hour: int) -> Dict[str, float]:
        """Get need decay modifiers based on time of day."""
        modifiers = {
            "energy": 1.0,
            "hunger": 1.0,
            "social": 1.0,
            "fun": 1.0
        }
        
        # Late night: energy decays faster
        if hour >= 22 or hour <= 4:
            modifiers["energy"] = 1.5
            modifiers["fun"] = 0.5  # Less need for fun when tired
        
        # Meal times: hunger increases faster
        if hour in [7, 8, 12, 13, 18, 19]:
            modifiers["hunger"] = 1.5
        
        # Evening: social needs increase
        if 17 <= hour <= 21:
            modifiers["social"] = 1.3
        
        return modifiers

    def tick(self, current_hour: int = 12):
        """Update needs based on decay rates with time-of-day awareness."""
        modifiers = self.get_time_modifier(current_hour)
        
        self.energy = max(0.0, self.energy - self.ENERGY_DECAY_RATE * modifiers.get("energy", 1.0))
        self.hunger = min(100.0, self.hunger + self.HUNGER_INCREASE_RATE * modifiers.get("hunger", 1.0))
        self.bladder = min(100.0, self.bladder + self.BLADDER_INCREASE_RATE)
        self.social = max(0.0, self.social - self.SOCIAL_DECAY_RATE * modifiers.get("social", 1.0))
        self.fun = max(0.0, self.fun - self.FUN_DECAY_RATE * modifiers.get("fun", 1.0))
        self.hygiene = max(0.0, self.hygiene - self.HYGIENE_DECAY_RATE)
        self.comfort = max(0.0, self.comfort - self.COMFORT_DECAY_RATE)
        self.fulfillment = max(0.0, self.fulfillment - self.FULFILLMENT_DECAY_RATE)
    
    def get_critical_needs(self) -> List[str]:
        """Get list of needs that are critical (below 30 or above 70)."""
        critical = []
        if self.energy < 30:
            critical.append("energy (very tired)")
        if self.hunger > 70:
            critical.append("hunger (very hungry)")
        if self.bladder > 70:
            critical.append("bladder (need bathroom)")
        if self.social < 30:
            critical.append("social (lonely)")
        if self.fun < 30:
            critical.append("fun (bored)")
        if self.hygiene < 30:
            critical.append("hygiene (dirty)")
        if self.comfort < 25:
            critical.append("comfort (uncomfortable)")
        if self.fulfillment < 20:
            critical.append("fulfillment (unfulfilled)")
        return critical
    
    def get_priority_need(self) -> Optional[Tuple[str, float]]:
        """Get the single most urgent need and its urgency score."""
        needs_scores = [
            ("energy", 100 - self.energy),
            ("hunger", self.hunger),
            ("bladder", self.bladder),
            ("social", 100 - self.social),
            ("fun", 100 - self.fun),
            ("hygiene", 100 - self.hygiene),
            ("comfort", 100 - self.comfort),
            ("fulfillment", 100 - self.fulfillment),
        ]
        # Prioritize based on urgency
        needs_scores.sort(key=lambda x: x[1], reverse=True)
        if needs_scores[0][1] > 50:  # Only return if actually urgent
            return needs_scores[0]
        return None
    
    def to_description(self) -> str:
        """Get a natural language description of current needs."""
        status = []
        if self.energy < 30:
            status.append("exhausted")
        elif self.energy < 60:
            status.append("tired")
        
        if self.hunger > 70:
            status.append("very hungry")
        elif self.hunger > 40:
            status.append("hungry")
        
        if self.bladder > 70:
            status.append("urgently needs bathroom")
        elif self.bladder > 40:
            status.append("needs bathroom")
        
        if self.social < 30:
            status.append("lonely")
        elif self.social < 50:
            status.append("wanting company")
        
        if self.fun < 30:
            status.append("bored")
        elif self.fun < 50:
            status.append("seeking entertainment")
            
        if self.hygiene < 30:
            status.append("feeling dirty")
        
        if self.comfort < 30:
            status.append("uncomfortable")
        
        if self.fulfillment < 30:
            status.append("feeling unfulfilled")
        elif self.fulfillment < 50:
            status.append("seeking purpose")
        
        return ", ".join(status) if status else "feeling good"
    
    def get_wellbeing_score(self) -> float:
        """Calculate overall wellbeing from 0-100."""
        return (
            self.energy * 0.2 + 
            (100 - self.hunger) * 0.15 + 
            (100 - self.bladder) * 0.1 +
            self.social * 0.15 +
            self.fun * 0.1 +
            self.hygiene * 0.1 +
            self.comfort * 0.1 +
            self.fulfillment * 0.1
        )

class AgentPersonality(BaseModel):
    """Detailed personality profile that influences behavior."""
    # Big Five personality traits (0-1 scale)
    openness: float = Field(default=0.5, ge=0.0, le=1.0)  # Creativity, curiosity
    conscientiousness: float = Field(default=0.5, ge=0.0, le=1.0)  # Organization, discipline
    extraversion: float = Field(default=0.5, ge=0.0, le=1.0)  # Sociability, energy from others
    agreeableness: float = Field(default=0.5, ge=0.0, le=1.0)  # Cooperation, trust
    neuroticism: float = Field(default=0.5, ge=0.0, le=1.0)  # Emotional instability, anxiety
    
    # Specific behavioral tendencies
    morning_person: bool = True  # Prefers waking early vs staying up late
    preferred_wake_hour: int = Field(default=7, ge=0, le=23)
    preferred_sleep_hour: int = Field(default=22, ge=0, le=23)
    work_ethic: float = Field(default=0.6, ge=0.0, le=1.0)  # How much they prioritize work
    spontaneity: float = Field(default=0.4, ge=0.0, le=1.0)  # Likelihood to deviate from routine
    
    # Interests and preferences
    favorite_activities: List[str] = Field(default_factory=lambda: ["reading", "walking"])
    disliked_activities: List[str] = Field(default_factory=list)
    conversation_style: str = "friendly"  # friendly, reserved, talkative, brief
    
    # Life story that shaped this personality
    life_story: Optional['LifeStory'] = None
    
    def get_social_preference(self) -> float:
        """How much this agent prefers social interaction."""
        return self.extraversion * 0.7 + self.agreeableness * 0.3
    
    def get_routine_preference(self) -> float:
        """How much this agent prefers sticking to routines."""
        return self.conscientiousness * 0.6 + (1 - self.spontaneity) * 0.4
    
    def get_stress_susceptibility(self) -> float:
        """How easily this agent gets stressed."""
        return self.neuroticism
    
    def should_be_awake(self, hour: int) -> bool:
        """Check if the agent should be awake at this hour based on personality."""
        if self.preferred_wake_hour <= self.preferred_sleep_hour:
            return self.preferred_wake_hour <= hour < self.preferred_sleep_hour
        else:
            return hour >= self.preferred_wake_hour or hour < self.preferred_sleep_hour
    
    def get_description(self) -> str:
        """Get a natural language description of personality."""
        traits = []
        if self.extraversion > 0.7:
            traits.append("outgoing and social")
        elif self.extraversion < 0.3:
            traits.append("introverted and reserved")
        
        if self.conscientiousness > 0.7:
            traits.append("organized and disciplined")
        elif self.conscientiousness < 0.3:
            traits.append("spontaneous and flexible")
        
        if self.openness > 0.7:
            traits.append("creative and curious")
        
        if self.agreeableness > 0.7:
            traits.append("warm and cooperative")
        elif self.agreeableness < 0.3:
            traits.append("independent and competitive")
        
        if self.neuroticism > 0.7:
            traits.append("sensitive and emotional")
        elif self.neuroticism < 0.3:
            traits.append("calm and stable")
        
        return ", ".join(traits) if traits else "balanced personality"


class AgentState(BaseModel):
    """Current state of an agent."""
    agent_id: str
    name: str
    background: Optional[str] = None
    location: Point2D
    home_location_name: str
    current_action: Optional[ActionDecision] = None
    current_plan: Optional[Plan] = None
    status: str = Field(default="idle")
    last_reflection_time: Optional[datetime] = Field(default_factory=datetime.utcnow)
    relationship_scores: Dict[str, float] = Field(default_factory=dict)
    
    # Enhanced state tracking
    current_activity_description: str = Field(default="")  # What they're doing in natural language
    current_inner_thought: str = Field(default="")  # What they're thinking
    is_busy: bool = False
    is_sleeping: bool = False
    is_in_conversation: bool = False
    conversation_partner: Optional[str] = None
    action_start_time: Optional[datetime] = None
    action_end_time: Optional[datetime] = None
    
    # Detailed activity tracking
    current_activity_type: Optional[ActionType] = None
    current_location_name: Optional[str] = None
    current_sub_location: Optional[str] = None
    objects_in_use: List[str] = Field(default_factory=list)  # Currently interacting with these objects
    current_scheduled_activity: Optional[ScheduledActivity] = None  # The schedule being followed
    
    def update_relationship(self, other_agent_id: str, delta: float):
        """Update relationship score with another agent."""
        current = self.relationship_scores.get(other_agent_id, 0.0)
        self.relationship_scores[other_agent_id] = max(-1.0, min(1.0, current + delta))
    
    def get_relationship_description(self, other_agent_id: str) -> str:
        """Get a description of relationship with another agent."""
        score = self.relationship_scores.get(other_agent_id, 0.0)
        if score > 0.8:
            return "close friend"
        elif score > 0.5:
            return "friend"
        elif score > 0.2:
            return "acquaintance"
        elif score > -0.2:
            return "neutral"
        elif score > -0.5:
            return "disliked"
        else:
            return "enemy"
    
    def start_activity(self, description: str, duration_minutes: int, world_time: datetime):
        """Start a new activity with expected duration."""
        self.current_activity_description = description
        self.is_busy = True
        self.action_start_time = world_time
        self.action_end_time = world_time + timedelta(minutes=duration_minutes)
    
    def start_scheduled_activity(self, activity: ScheduledActivity, world_time: datetime):
        """Start a scheduled activity with detailed tracking."""
        self.current_scheduled_activity = activity
        self.current_activity_type = activity.activity_type
        self.current_activity_description = activity.description
        self.current_location_name = activity.location
        self.current_sub_location = activity.sub_location
        self.objects_in_use = activity.specific_objects.copy()
        self.is_busy = True
        self.action_start_time = world_time
        self.action_end_time = world_time + timedelta(minutes=activity.get_duration_minutes())
        
        # Update sleep status
        if activity.activity_type == ActionType.SLEEP:
            logger.info(f"😴 STARTED SLEEPING: {self.state.name} is now sleeping at {activity.location}/{activity.sub_location}")
            self.is_sleeping = True
            self.status = "sleeping"
        else:
            if self.is_sleeping:
                logger.info(f"⏰ STOPPED SLEEPING: {self.state.name} stopped sleeping to do {activity.activity_type.value}")
            self.is_sleeping = False
            self.status = activity.activity_type.value
    
    def get_detailed_status(self) -> str:
        """Get a detailed natural language description of what the agent is doing."""
        if not self.current_activity_description:
            return f"{self.name} is idle."
        
        status_parts = [f"{self.name} is {self.current_activity_description}"]
        
        if self.current_sub_location:
            status_parts.append(f"in the {self.current_sub_location}")
        elif self.current_location_name:
            status_parts.append(f"at {self.current_location_name}")
        
        if self.objects_in_use:
            if len(self.objects_in_use) == 1:
                status_parts.append(f"using the {self.objects_in_use[0]}")
            elif len(self.objects_in_use) == 2:
                status_parts.append(f"using the {self.objects_in_use[0]} and {self.objects_in_use[1]}")
            else:
                obj_list = ", ".join(self.objects_in_use[:-1]) + f", and {self.objects_in_use[-1]}"
                status_parts.append(f"using {obj_list}")
        
        return " ".join(status_parts) + "."
    
    def is_activity_complete(self, world_time: datetime) -> bool:
        """Check if the current activity should be complete."""
        if not self.action_end_time:
            return True
        return world_time >= self.action_end_time


class Agent(BaseModel):
    """
    Representation of an agent in the world.
    
    BEHAVIOR MODEL (Sims-like):
    - Needs decay over time and drive behavior deterministically
    - Agents choose actions based on urgency scoring and available objects
    - NO LLM is used for action selection
    
    LLM is ONLY used for:
    - Generating conversation dialogue with other agents
    - Creating rich internal thoughts (occasionally)
    - Memory formation and reflection
    """
    state: AgentState
    memory: AgentMemory = Field(default_factory=AgentMemory)
    needs: AgentNeeds = Field(default_factory=AgentNeeds)
    emotions: AgentEmotionalState = Field(default_factory=AgentEmotionalState)
    personality: AgentPersonality = Field(default_factory=AgentPersonality)
    job: Optional[AgentJob] = None  # Job/occupation information
    ollama_client: Any = Field(default_factory=OllamaClient)
    is_setup: bool = Field(default=False)
    last_decision_time: int = 0  # Tick number when last decision was made
    decision_cooldown: int = 3  # Ticks between decisions
    current_action_start: Optional[datetime] = None
    
    # Track behavioral patterns
    recent_actions: List[str] = Field(default_factory=list)
    
    # Current Sims-style action
    current_sims_action: Optional[SimsAction] = None
    
    # Daily schedule awareness
    daily_schedule: List[ScheduledActivity] = Field(default_factory=list)  # Scheduled activities
    
    # Inner life
    current_thoughts: List[str] = Field(default_factory=list)  # Recent inner thoughts
    pending_reflections: List[str] = Field(default_factory=list)  # Things to think about
    
    # Conversation history for context
    recent_conversations: List[Dict[str, Any]] = Field(default_factory=list)

    @classmethod
    def create(cls, agent_id: str, name: str, location: Point2D, home_location_name: str, 
               personality: Optional[AgentPersonality] = None) -> "Agent":
        """Create a new agent with initial state and randomized personality."""
        state = AgentState(
            agent_id=agent_id,
            name=name,
            location=location,
            home_location_name=home_location_name
        )
        
        # Generate personality if not provided
        if personality is None:
            personality = AgentPersonality(
                openness=random.uniform(0.3, 0.8),
                conscientiousness=random.uniform(0.3, 0.8),
                extraversion=random.uniform(0.2, 0.9),
                agreeableness=random.uniform(0.4, 0.9),
                neuroticism=random.uniform(0.2, 0.7),
                morning_person=random.choice([True, False]),
                preferred_wake_hour=random.randint(5, 9) if random.random() > 0.5 else random.randint(8, 11),
                preferred_sleep_hour=random.randint(21, 23) if random.random() > 0.5 else random.randint(23, 24) % 24,
                work_ethic=random.uniform(0.4, 0.9),
                spontaneity=random.uniform(0.2, 0.7),
                favorite_activities=random.sample(
                    ["reading", "walking", "cooking", "music", "games", "socializing", "exercise", "crafts"],
                    k=random.randint(2, 4)
                )
            )
        
        return cls(state=state, personality=personality)
    
    def generate_inner_thought(self, context: str = "") -> str:
        """Generate an inner thought based on current situation."""
        thoughts = []
        
        # Thoughts based on needs
        priority_need = self.needs.get_priority_need()
        if priority_need:
            need_name, urgency = priority_need
            if urgency > 70:
                thoughts.append(f"I really need to address my {need_name}...")
            elif urgency > 50:
                thoughts.append(f"I should probably take care of my {need_name} soon.")
        
        # Thoughts based on emotions
        if self.emotions.emotion_intensity > 0.6:
            if self.emotions.primary_emotion == EmotionType.HAPPY:
                thoughts.append("Today is going well!")
            elif self.emotions.primary_emotion == EmotionType.SAD:
                thoughts.append("I'm feeling a bit down...")
            elif self.emotions.primary_emotion == EmotionType.ANXIOUS:
                thoughts.append("Something feels off. I'm a bit worried.")
            elif self.emotions.primary_emotion == EmotionType.BORED:
                thoughts.append("I need to do something more interesting.")
        
        # Random personality-driven thoughts
        if random.random() < self.personality.openness * 0.3:
            thoughts.append("I wonder what new things I could try today...")
        
        if random.random() < self.personality.conscientiousness * 0.3:
            thoughts.append("Let me think about what I need to get done...")
        
        if thoughts:
            return random.choice(thoughts)
        return "Just going about my day..."
    
    def form_memory(self, content: str, memory_type: MemoryType, importance: float,
                    emotional_valence: float = 0.0, related_agents: List[str] = None,
                    location: Optional[str] = None):
        """Create and store a new memory."""
        memory = AgentMemoryEntry(
            id=f"mem_{self.state.agent_id}_{len(self.memory.episodic_memories)}",
            timestamp=datetime.utcnow(),
            memory_type=memory_type,
            content=content,
            importance=importance,
            emotional_valence=emotional_valence,
            related_agents=related_agents or [],
            related_location=location
        )
        self.memory.add_memory(memory)
        
        # Add to daily observations if significant
        if importance >= 5:
            self.memory.daily_observations.append(content)

    def get_current_location_name(self, simulation) -> Optional[str]:
        """Get the name of the current main location."""
        if simulation is None:
            return None
        # World stores locations in the map's arena_sub_locations_to_positions as
        # keys of (main_location, sub_location) -> list[Point2D]
        for (main_loc, sub_loc), positions in simulation.world.map.arena_sub_locations_to_positions.items():
            for pos in positions:
                if pos.x == self.state.location.x and pos.y == self.state.location.y:
                    return main_loc
        return None
    
    def get_nearby_agents(self, simulation, radius: float = 10.0) -> List[Dict[str, Any]]:
        """Get list of nearby agents within radius."""
        nearby = []
        for agent in simulation.agents.values():
            if agent.state.agent_id != self.state.agent_id:
                distance = self.state.location.distance_to(agent.state.location)
                if distance <= radius:
                    nearby.append({
                        "name": agent.state.name,
                        "distance": distance,
                        "location": agent.get_current_location_name(simulation)
                    })
        return nearby
    
    def get_nearby_objects(self, simulation) -> List[Dict[str, Any]]:
        """Get objects in the current location."""
        nearby = []
        current_loc = self.get_current_location_name(simulation)
        if not current_loc:
            return nearby

        # Use World API to get objects in the main location
        try:
            objs = simulation.world.get_objects_in_sector(current_loc)
        except Exception:
            objs = []

        for obj in objs:
            # obj may be structured per map.game_objects_to_arena_locations
            nearby.append({
                "id": obj.get("id") if isinstance(obj, dict) else str(obj),
                "sub_location": obj.get("sub_location") if isinstance(obj, dict) else None,
                "location": current_loc
            })
        return nearby[:10]  # Limit to 10 nearest objects

    def _choose_object_for_need(self, simulation, need: str, current_location: Optional[str]) -> Optional[Dict[str, Any]]:
        """Choose a nearby or home object that satisfies a given need using simple keyword matching."""
        keywords = {
            "sleep": ["bed", "mattress"],  # Only beds for sleeping
            "energy": ["bed", "sofa", "mattress", "couch"],  # Resting (not sleeping) can use sofas
            "hunger": ["refrigerator", "stove", "table", "kitchen", "fridge"],
            "bladder": ["toilet", "bathroom", "restroom"],
            "social": ["sofa", "park", "cafe", "bench"],
            "fun": ["tv", "game", "piano", "arcade", "computer"],
            "hygiene": ["shower", "bath", "bathtub", "sink"]
        }

        possible_objs = self.get_nearby_objects(simulation)
        # Prefer objects in current location
        for obj in possible_objs:
            name = (obj.get("id") or "").lower()
            for kw in keywords.get(need, []):
                if kw in name:
                    return obj

        # Fallback: check home location objects
        home_positions = simulation.world.get_map_of_sub_location_to_positions_for_main_location(self.state.home_location_name)
        if home_positions:
            for sub, pos_list in home_positions.items():
                for kw in keywords.get(need, []):
                    if kw in sub.lower():
                        # fabricate simple object dict
                        return {"id": kw, "sub_location": sub, "location": self.state.home_location_name}

        return None
    
    def get_current_scheduled_activity(self, simulation) -> Optional[ScheduledActivity]:
        """Get the scheduled activity for the current time, if any."""
        current_hour = simulation.world_time.hour
        day_of_week = simulation.world_time.weekday()
        
        # Check job schedule first
        if self.job:
            work_activity = self.job.get_work_activity(current_hour, day_of_week)
            if work_activity:
                return work_activity
        
        # Check personal schedule
        for activity in self.daily_schedule:
            if day_of_week in activity.days_of_week:
                if activity.start_hour <= activity.end_hour:
                    if activity.start_hour <= current_hour < activity.end_hour:
                        return activity
                else:  # Crosses midnight
                    if current_hour >= activity.start_hour or current_hour < activity.end_hour:
                        return activity
        
        return None

    def _rule_based_decision(self, simulation) -> Optional[ActionDecision]:
        """Quick deterministic rules to address urgent needs and common behavior.

        This handles the majority of routine choices so the LLM is reserved for
        planning, social scenes, or ambiguous situations. Now enhanced with
        personality-aware decision making and work/schedule awareness.
        """
        critical = self.needs.get_critical_needs()
        current_loc = self.get_current_location_name(simulation)
        current_hour = simulation.world_time.hour
        
        # Check if there's a scheduled activity (work, class, etc.)
        scheduled = self.get_current_scheduled_activity(simulation)
        if scheduled:
            # Only skip schedule for truly urgent needs
            if critical and any(need in ["bladder", "energy (very tired)"] for need in critical):
                # Handle urgent need first, then return to schedule
                pass
            else:
                # Follow the schedule
                return ActionDecision(
                    action_type=scheduled.activity_type,
                    reasoning=scheduled.description,
                    target_location=scheduled.location,
                    target_sub_location=scheduled.sub_location,
                    priority=scheduled.priority,
                    expected_duration_minutes=(scheduled.end_hour - current_hour) * 60 if scheduled.end_hour > current_hour else 60,
                    inner_thought=f"Time for {scheduled.description.lower()}..."
                )

        current_loc = self.get_current_location_name(simulation)
        current_hour = simulation.world_time.hour
        
        # Check if agent should be sleeping based on personality
        if not self.personality.should_be_awake(current_hour):
            logger.debug(f"🌙 SLEEP TIME CHECK: {self.state.name} | Current hour: {current_hour} | Should sleep | Currently sleeping: {self.state.is_sleeping}")
            if not self.state.is_sleeping:
                obj = self._choose_object_for_need(simulation, "sleep", current_loc)
                if obj:
                    logger.info(f"🛏️ SLEEP DECISION: {self.state.name} deciding to go to sleep | Target bed: {obj.get('id')} at {obj.get('location')}/{obj.get('sub_location')}")
                    # Don't set is_sleeping yet - wait until agent arrives at bed
                    # is_sleeping will be set in execute_action when the sleep action runs
                    return ActionDecision(
                        action_type=ActionType.SLEEP,
                        reasoning=f"It's {current_hour}:00 - time to sleep",
                        target_location=obj.get("location", self.state.home_location_name),
                        target_sub_location=obj.get("sub_location"),
                        target_object=obj.get("id"),
                        priority=10,
                        expected_duration_minutes=480,  # 8 hours of sleep
                        inner_thought="Time for bed... I need my rest."
                    )
                else:
                    logger.warning(f"⚠️ NO BED FOUND: {self.state.name} wants to sleep but couldn't find a bed")
            else:
                logger.debug(f"💤 ALREADY SLEEPING: {self.state.name} is already sleeping, continuing")
            return None  # Continue sleeping
        # Wake-up is now handled in tick() method, not here

        # Address the most critical need deterministically
        if critical:
            # Bladder is always urgent - personality doesn't change this!
            if any("bladder" in c for c in critical):
                obj = self._choose_object_for_need(simulation, "bladder", current_loc)
                if obj:
                    return ActionDecision(
                        action_type=ActionType.USE_TOILET,
                        reasoning="Need to use the bathroom urgently",
                        target_location=obj.get("location", self.state.home_location_name),
                        target_sub_location=obj.get("sub_location"),
                        target_object=obj.get("id"),
                        priority=10,
                        expected_duration_minutes=5,
                        inner_thought="I really need to go..."
                    )
            
            # Hygiene - when very dirty, take a bath
            if any("hygiene" in c for c in critical):
                obj = self._choose_object_for_need(simulation, "hygiene", current_loc)
                if obj:
                    return ActionDecision(
                        action_type=ActionType.SHOWER,
                        reasoning="Really need to take a bath",
                        target_location=obj.get("location", self.state.home_location_name),
                        target_sub_location=obj.get("sub_location"),
                        target_object=obj.get("id"),
                        priority=8,
                        expected_duration_minutes=20,
                        inner_thought="I feel so grimy... I need a shower."
                    )
            
            # Energy - affected by how disciplined the agent is
            if any("energy" in c for c in critical):
                # Conscientious agents might push through tiredness
                if random.random() > self.personality.conscientiousness * 0.3:
                    obj = self._choose_object_for_need(simulation, "energy", current_loc)
                    if obj:
                        return ActionDecision(
                            action_type=ActionType.USE_OBJECT,
                            reasoning="Feeling exhausted — need to rest",
                            target_location=obj.get("location", self.state.home_location_name),
                            target_sub_location=obj.get("sub_location"),
                            target_object=obj.get("id"),
                            priority=10,
                            expected_duration_minutes=60,
                            inner_thought="I can barely keep my eyes open..."
                        )

            if any("hunger" in c for c in critical):
                obj = self._choose_object_for_need(simulation, "hunger", current_loc)
                if obj:
                    # When very hungry, cook a proper meal
                    is_kitchen = "stove" in (obj.get("id") or "").lower() or "kitchen" in (obj.get("sub_location") or "").lower()
                    action_type = ActionType.COOK if is_kitchen else ActionType.EAT
                    meal_time = 30 if 11 <= current_hour <= 13 or 17 <= current_hour <= 20 else 20
                    return ActionDecision(
                        action_type=action_type,
                        reasoning="Very hungry — time to eat",
                        target_location=obj.get("location", self.state.home_location_name),
                        target_sub_location=obj.get("sub_location"),
                        target_object=obj.get("id"),
                        priority=9,
                        expected_duration_minutes=meal_time,
                        inner_thought="My stomach is growling... I need food."
                    )

        # Social rule: depends heavily on extraversion
        social_threshold = 40 + (self.personality.extraversion * 20)  # Extroverts need more social
        if self.needs.social < social_threshold:
            nearby = self.get_nearby_agents(simulation)
            if nearby:
                # Choose conversation partner based on relationship
                partner = self._choose_conversation_partner(nearby)
                if partner:
                    relationship = self.state.get_relationship_description(partner.get("name", ""))
                    return ActionDecision(
                        action_type=ActionType.CONVERSE,
                        reasoning=f"Feeling like chatting — {partner.get('name')} is nearby",
                        target_location=partner.get("location") or current_loc or self.state.home_location_name,
                        target_agent=partner.get("name"),
                        priority=7 if self.personality.extraversion > 0.6 else 5,
                        expected_duration_minutes=random.randint(10, 30),
                        inner_thought=f"It would be nice to talk to {partner.get('name')}... we're {relationship}s."
                    )
            elif self.personality.extraversion > 0.7:
                # Very extroverted agents might seek out others
                locations = simulation.world.get_all_main_locations()
                social_locations = [loc for loc in locations if any(kw in loc.lower() for kw in ["cafe", "park", "bar", "common"])]
                if social_locations:
                    return ActionDecision(
                        action_type=ActionType.MOVE,
                        reasoning="Looking for someone to talk to",
                        target_location=random.choice(social_locations),
                        priority=5,
                        expected_duration_minutes=15,
                        inner_thought="I should go somewhere I can meet people..."
                    )

        # Fun rule: depends on personality
        fun_threshold = 40 - (self.personality.conscientiousness * 15)  # Conscientious people tolerate boredom better
        if self.needs.fun < fun_threshold:
            # Choose activity based on personality and preferences
            activity = self._choose_fun_activity(simulation, current_loc)
            if activity:
                return activity

        # Work/fulfillment during work hours if conscientiousness is high
        if 9 <= current_hour <= 17 and self.personality.work_ethic > 0.5:
            if self.needs.fulfillment < 60 and random.random() < self.personality.work_ethic:
                return ActionDecision(
                    action_type=ActionType.WORK,
                    reasoning="Time to be productive",
                    target_location=current_loc or self.state.home_location_name,
                    priority=6,
                    expected_duration_minutes=random.randint(30, 120),
                    inner_thought="I should get some work done..."
                )
        
        # Even if needs aren't critical, agents should still do things proactively
        # Check for moderate needs and address them
        
        # Proactive bladder management - don't wait until urgent
        if self.needs.bladder > 40 and random.random() < 0.4:
            obj = self._choose_object_for_need(simulation, "bladder", current_loc)
            if obj:
                return ActionDecision(
                    action_type=ActionType.USE_TOILET,
                    reasoning="Using the bathroom",
                    target_location=obj.get("location", self.state.home_location_name),
                    target_sub_location=obj.get("sub_location"),
                    target_object=obj.get("id"),
                    priority=6,
                    expected_duration_minutes=5,
                    inner_thought="Better go now before it becomes urgent."
                )
        
        # Meal times - agents should cook and eat at appropriate hours
        is_meal_time = current_hour in [7, 8, 12, 13, 18, 19, 20]
        if self.needs.hunger > 40 and not critical:
            obj = self._choose_object_for_need(simulation, "hunger", current_loc)
            if obj and (is_meal_time or random.random() < 0.4):
                # Determine if cooking or just eating
                if "stove" in (obj.get("id") or "").lower() or "kitchen" in (obj.get("sub_location") or "").lower():
                    action_type = ActionType.COOK
                    reasoning = "Cooking a meal"
                    thought = "Time to make something to eat..."
                    duration = 25
                else:
                    action_type = ActionType.EAT
                    reasoning = "Getting something to eat"
                    thought = "I should eat something now while I'm thinking about it."
                    duration = 15
                return ActionDecision(
                    action_type=action_type,
                    reasoning=reasoning,
                    target_location=obj.get("location", self.state.home_location_name),
                    target_sub_location=obj.get("sub_location"),
                    target_object=obj.get("id"),
                    priority=6,
                    expected_duration_minutes=duration,
                    inner_thought=thought
                )
        
        if self.needs.energy < 60 and current_hour >= 14 and random.random() < 0.3:
            # Afternoon rest/nap
            obj = self._choose_object_for_need(simulation, "energy", current_loc)
            if obj:
                return ActionDecision(
                    action_type=ActionType.USE_OBJECT,
                    reasoning="Taking a short rest",
                    target_location=obj.get("location", self.state.home_location_name),
                    target_sub_location=obj.get("sub_location"),
                    target_object=obj.get("id"),
                    priority=5,
                    expected_duration_minutes=30,
                    inner_thought="I could use a little break..."
                )
        
        if self.needs.hygiene < 70 and random.random() < 0.5:
            obj = self._choose_object_for_need(simulation, "hygiene", current_loc)
            if obj:
                # Prefer bath/shower in morning or evening
                is_bath_time = 6 <= current_hour <= 9 or 20 <= current_hour <= 23
                action_type = ActionType.SHOWER if is_bath_time else ActionType.USE_OBJECT
                duration = 20 if is_bath_time else 10
                reasoning = "Taking a bath" if is_bath_time else "Freshening up"
                thought = "Time for a nice bath..." if is_bath_time else "Should probably clean up a bit."
                return ActionDecision(
                    action_type=action_type,
                    reasoning=reasoning,
                    target_location=obj.get("location", self.state.home_location_name),
                    target_sub_location=obj.get("sub_location"),
                    target_object=obj.get("id"),
                    priority=5,
                    expected_duration_minutes=duration,
                    inner_thought=thought
                )

        # Spontaneous actions based on personality
        if random.random() < self.personality.spontaneity * 0.1:
            locations = simulation.world.get_all_main_locations()
            if locations:
                chosen = random.choice(locations)
                activities = ["explore", "check out", "visit"]
                return ActionDecision(
                    action_type=ActionType.MOVE,
                    reasoning=f"Feeling like exploring — going to {random.choice(activities)} {chosen}",
                    target_location=chosen,
                    priority=4,
                    expected_duration_minutes=random.randint(10, 30),
                    inner_thought="Wonder what's going on over there..."
                )

        # Check learned routines for this hour
        typical_actions = self.memory.get_typical_actions_for_hour(current_hour)
        if typical_actions and random.random() < self.personality.get_routine_preference():
            # Follow routine
            most_common = max(set(typical_actions), key=typical_actions.count)
            # Don't just wander if that's the routine - do something meaningful
            if most_common != "move":
                return ActionDecision(
                    action_type=ActionType.WAIT,
                    reasoning=f"Following daily routine: usually {most_common} around this time",
                    target_location=current_loc or self.state.home_location_name,
                    priority=4,
                    expected_duration_minutes=30,
                    inner_thought=f"This is what I usually do at {current_hour}:00..."
                )

        # Default: do nothing special - let LLM decide or wait
        return None
    
    def _choose_conversation_partner(self, nearby_agents: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Choose the best conversation partner based on relationships and proximity."""
        if not nearby_agents:
            return None
        
        # Score each potential partner
        scored = []
        for agent in nearby_agents:
            name = agent.get("name", "")
            relationship_score = self.state.relationship_scores.get(name, 0.0)
            distance = agent.get("distance", 10)
            
            # Prefer closer agents and those we like
            score = (1.0 - distance / 20.0) + relationship_score * 0.5
            
            # Agreeable agents are more likely to approach anyone
            if self.personality.agreeableness > 0.6:
                score += 0.2
            
            scored.append((agent, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0] if scored else None
    
    def _choose_fun_activity(self, simulation, current_loc: Optional[str]) -> Optional[ActionDecision]:
        """Choose a fun activity based on personality and preferences."""
        obj = self._choose_object_for_need(simulation, "fun", current_loc)
        
        # Check if any favorite activities are available
        for activity in self.personality.favorite_activities:
            activity_obj = self._find_object_for_activity(simulation, activity, current_loc)
            if activity_obj:
                return ActionDecision(
                    action_type=ActionType.LEISURE,
                    reasoning=f"Doing something I enjoy: {activity}",
                    target_location=activity_obj.get("location", self.state.home_location_name),
                    target_sub_location=activity_obj.get("sub_location"),
                    target_object=activity_obj.get("id"),
                    priority=6,
                    expected_duration_minutes=random.randint(20, 60),
                    inner_thought=f"I could really go for some {activity} right now."
                )
        
        # Fallback to generic fun object
        if obj:
            return ActionDecision(
                action_type=ActionType.USE_OBJECT,
                reasoning="Looking for something fun to do",
                target_location=obj.get("location", self.state.home_location_name),
                target_sub_location=obj.get("sub_location"),
                target_object=obj.get("id"),
                priority=5,
                expected_duration_minutes=30,
                inner_thought="I need to do something more interesting..."
            )
        
        return None
    
    def _find_object_for_activity(self, simulation, activity: str, current_loc: Optional[str]) -> Optional[Dict[str, Any]]:
        """Find an object that matches a specific activity preference."""
        activity_keywords = {
            "reading": ["book", "library", "magazine"],
            "walking": ["park", "garden", "trail"],
            "cooking": ["kitchen", "stove", "refrigerator"],
            "music": ["piano", "guitar", "radio", "speaker"],
            "games": ["game", "arcade", "console", "computer"],
            "socializing": ["cafe", "bar", "lounge", "common"],
            "exercise": ["gym", "track", "weights"],
            "crafts": ["workshop", "studio", "craft"],
        }
        
        keywords = activity_keywords.get(activity, [activity])
        nearby_objects = self.get_nearby_objects(simulation)
        
        for obj in nearby_objects:
            name = (obj.get("id") or "").lower()
            for kw in keywords:
                if kw in name:
                    return obj
        
        return None

    # =========================================================================
    # DEPRECATED: Old LLM-based decision methods - NO LONGER USED
    # The Sims-style behavior system in decide_next_action() now handles all
    # action selection deterministically. LLM is only used for conversations.
    # =========================================================================

    def _generate_llm_thought(self, context: str = "") -> Optional[str]:
        """
        Generate a rich inner thought using LLM.
        
        This is one of the ONLY places LLM is now used - for internal thoughts,
        NOT for action selection.
        
        Called occasionally to give agents richer internal life.
        """
        if random.random() > 0.2:  # Only 20% chance to use LLM for thoughts
            return None
            
        system_prompt = f"""You are the inner voice of {self.state.name}.

CHARACTER:
- Personality: {self.personality.get_description()}
- Current mood: {self.emotions.get_mood_description()}
- Background: {self.state.background or 'Unknown'}

SITUATION: {context or self.state.current_activity_description or 'idle'}

Generate a single brief inner thought (10-20 words max).
Just output the thought itself, no quotes."""

        try:
            response = self.ollama_client.generate(
                system=system_prompt,
                prompt="What is on their mind?",
                temperature=0.9,
                max_tokens=40
            )
            return response.strip().strip('"\'')[:80]
        except Exception:
            return None

    def _deprecated_should_use_llm(self, simulation, rule_decision: Optional[ActionDecision]) -> bool:
        """DEPRECATED: LLM is no longer used for action selection.
        
        This method is kept for reference but should not be called.
        Action selection is now fully deterministic via SimsBehaviorManager.
        """
        return False  # Never use LLM for action selection

    def _deprecated_llm_decision(self, simulation) -> Optional[ActionDecision]:
        """Fallback to LLM for richer contextual decisions or planning."""
        needs_desc = self.needs.to_description()
        critical_needs = self.needs.get_critical_needs()
        nearby_agents = self.get_nearby_agents(simulation)
    def _deprecated_llm_decision(self, simulation) -> Optional[ActionDecision]:
        """
        DEPRECATED: LLM is no longer used for action selection.
        
        This method is kept for reference but should not be called.
        Action selection is now fully deterministic via SimsBehaviorManager.
        
        LLM is now ONLY used for:
        - Generating conversation dialogue (see conversation.py)
        - Occasional inner thoughts (see _generate_llm_thought)
        - Memory/reflection formation
        """
        return None  # Never use LLM for action selection

    def execute_action(self, decision: ActionDecision, simulation):
        """Execute the decided action with need effects (Sims-style)."""
        action_desc = f"{decision.action_type.value}: {decision.reasoning}"
        current_hour = simulation.world_time.hour
        
        logger.info(f"⚡ EXECUTE ACTION: {self.state.name} | Action: {decision.action_type.value} | Object: {decision.target_object} | Location: {decision.target_location}/{decision.target_sub_location}")
        
        # Record routine learning
        self.memory.record_routine_action(current_hour, decision.action_type.value)
        
        # Update activity state and track object usage
        self.state.start_activity(
            decision.reasoning, 
            decision.expected_duration_minutes,
            simulation.world_time
        )
        
        # Track which object is being used
        if decision.target_object:
            self.state.objects_in_use = [decision.target_object]
        else:
            self.state.objects_in_use = []
        
        # Apply Sims-style need effects if we have a sims_action
        if self.current_sims_action and decision.target_object:
            SimsBehaviorManager.apply_action_effects(self, self.current_sims_action)
            print(f"{self.state.name}: {decision.reasoning}")
            return
        
        # Fallback to manual need fulfillment (for compatibility)
        if decision.action_type == ActionType.USE_OBJECT:
            target_obj = (decision.target_object or "").lower()
            
            if "bed" in target_obj or "sofa" in target_obj or "couch" in target_obj:
                energy_gain = 20 if "bed" in target_obj else 10
                self.needs.energy = min(100.0, self.needs.energy + energy_gain)
                self.needs.comfort = min(100.0, self.needs.comfort + 15)
                self.emotions.decay_over_time()  # Rest helps emotional recovery
                print(f"{self.state.name} is resting (+{energy_gain} energy)")
                
            elif any(food in target_obj for food in ["refrigerator", "stove", "kitchen", "food", "table"]):
                self.needs.hunger = max(0.0, self.needs.hunger - 35)
                self.needs.comfort = min(100.0, self.needs.comfort + 5)
                # Eating can be enjoyable
                if random.random() < 0.3:
                    self.emotions.update_emotion(EmotionType.CONTENT, 0.3, "enjoyed a meal")
                print(f"{self.state.name} is eating (-35 hunger)")
                
            elif "toilet" in target_obj or "bathroom" in target_obj:
                self.needs.bladder = max(0.0, self.needs.bladder - 50)
                self.needs.comfort = min(100.0, self.needs.comfort + 10)
                print(f"{self.state.name} is using bathroom (-50 bladder)")
                
            elif any(clean in target_obj for clean in ["shower", "sink", "bath"]):
                self.needs.hygiene = min(100.0, self.needs.hygiene + 30)
                self.needs.comfort = min(100.0, self.needs.comfort + 10)
                self.needs.energy = min(100.0, self.needs.energy + 5)
                self.emotions.update_emotion(EmotionType.CONTENT, 0.2, "feeling fresh")
                print(f"{self.state.name} is cleaning up (+30 hygiene)")
                
            elif any(fun in target_obj for fun in ["tv", "game", "piano", "book", "computer", "music"]):
                self.needs.fun = min(100.0, self.needs.fun + 20)
                self.emotions.update_emotion(EmotionType.HAPPY, 0.3, "having fun")
                print(f"{self.state.name} is having fun (+20 fun)")
        
        elif decision.action_type == ActionType.CONVERSE:
            social_gain = 25 if self.personality.extraversion > 0.5 else 15
            self.needs.social = min(100.0, self.needs.social + social_gain)
            self.needs.fun = min(100.0, self.needs.fun + 5)
            
            # Update relationship
            if decision.target_agent:
                self.state.update_relationship(decision.target_agent, 0.05)
                self.state.is_in_conversation = True
                self.state.conversation_partner = decision.target_agent
                
                # Form a memory of the interaction
                self.form_memory(
                    f"Had a conversation with {decision.target_agent}",
                    MemoryType.SOCIAL,
                    importance=5.0,
                    emotional_valence=0.3,
                    related_agents=[decision.target_agent]
                )
                self.memory.add_relationship_memory(
                    decision.target_agent,
                    f"Talked at {simulation.world_time.strftime('%I:%M %p')}"
                )
                
            self.emotions.update_emotion(EmotionType.HAPPY, 0.25, "social interaction")
            print(f"{self.state.name} is socializing (+{social_gain} social)")
        
        elif decision.action_type == ActionType.WORK:
            self.needs.fulfillment = min(100.0, self.needs.fulfillment + 15)
            self.needs.energy = max(0.0, self.needs.energy - 5)
            self.needs.fun = max(0.0, self.needs.fun - 3)
            
            # Job satisfaction affects emotional response
            if self.job:
                if self.job.job_satisfaction > 0.7:
                    self.emotions.update_emotion(EmotionType.CONTENT, 0.3, "productive work")
                elif self.job.job_satisfaction < 0.4:
                    self.emotions.update_emotion(EmotionType.BORED, 0.2, "tedious work")
                    self.emotions.stress_level = min(1.0, self.emotions.stress_level + 0.05)
                else:
                    if random.random() < self.personality.conscientiousness:
                        self.emotions.update_emotion(EmotionType.CONTENT, 0.2, "getting work done")
            
            print(f"{self.state.name} is working (+15 fulfillment)")
        
        elif decision.action_type == ActionType.ATTEND_CLASS:
            self.needs.fulfillment = min(100.0, self.needs.fulfillment + 10)
            self.needs.energy = max(0.0, self.needs.energy - 8)
            self.needs.fun = max(0.0, self.needs.fun - 5)
            
            # Learning can be engaging or boring depending on personality
            if self.personality.openness > 0.6:
                self.emotions.update_emotion(EmotionType.CONTENT, 0.2, "learning new things")
                self.needs.fun = min(100.0, self.needs.fun + 5)  # Curious people enjoy learning
            else:
                self.emotions.update_emotion(EmotionType.BORED, 0.15, "sitting in class")
            
            print(f"{self.state.name} is attending class (+10 fulfillment)")
        
        elif decision.action_type == ActionType.STUDY:
            self.needs.fulfillment = min(100.0, self.needs.fulfillment + 8)
            self.needs.energy = max(0.0, self.needs.energy - 6)
            self.emotions.stress_level = min(1.0, self.emotions.stress_level + 0.03)
            
            if self.personality.conscientiousness > 0.6:
                self.emotions.update_emotion(EmotionType.CONTENT, 0.15, "being productive")
            
            print(f"{self.state.name} is studying (+8 fulfillment)")
        
        elif decision.action_type == ActionType.LEISURE:
            self.needs.fun = min(100.0, self.needs.fun + 25)
            self.needs.energy = min(100.0, self.needs.energy + 3)
            self.emotions.update_emotion(EmotionType.HAPPY, 0.35, "leisure activity")
            print(f"{self.state.name} is enjoying leisure time (+25 fun)")
        
        elif decision.action_type == ActionType.EXERCISE:
            self.needs.energy = max(0.0, self.needs.energy - 10)
            self.needs.fun = min(100.0, self.needs.fun + 10)
            self.needs.hygiene = max(0.0, self.needs.hygiene - 15)
            self.emotions.update_emotion(EmotionType.EXCITED, 0.3, "physical activity")
            # Long-term mood boost
            self.emotions.mood_baseline = min(1.0, self.emotions.mood_baseline + 0.02)
            print(f"{self.state.name} is exercising (+10 fun, -15 hygiene)")
        
        elif decision.action_type == ActionType.REFLECT:
            # Reflection helps process emotions and plan
            self.emotions.decay_over_time()
            self.emotions.stress_level = max(0.0, self.emotions.stress_level - 0.1)
            self.needs.fulfillment = min(100.0, self.needs.fulfillment + 5)
            
            # Generate a reflection thought
            thought = self.generate_inner_thought()
            self.current_thoughts.append(thought)
            self.pending_reflections = []  # Clear pending reflections
            print(f"{self.state.name} is reflecting: '{thought}'")
        
        elif decision.action_type == ActionType.WAIT:
            # Small passive recovery
            self.needs.energy = min(100.0, self.needs.energy + 2)
            self.needs.comfort = min(100.0, self.needs.comfort + 3)
            self.emotions.decay_over_time()
        
        elif decision.action_type == ActionType.SLEEP:
            # Agent has arrived at bed and is now sleeping
            logger.info(f"😴💤 EXECUTING SLEEP: {self.state.name} is now going to sleep | Clearing path and setting is_sleeping=True")
            self.state.is_sleeping = True
            self.state.status = "sleeping"
            # Clear path to prevent any movement
            if self.memory.current_path:
                logger.warning(f"🛑 CLEARING SLEEP PATH: {self.state.name} had {len(self.memory.current_path)} steps remaining in path, clearing it")
            self.memory.current_path = None
            # Good energy recovery from sleep
            self.needs.energy = min(100.0, self.needs.energy + 30)
            self.needs.comfort = min(100.0, self.needs.comfort + 20)
            self.emotions.decay_over_time()  # Sleep helps emotional recovery
            self.emotions.stress_level = max(0.0, self.emotions.stress_level - 0.1)
            logger.debug(f"💤 SLEEP RECOVERY: {self.state.name} | Energy: +30, Comfort: +20, Stress: -{0.1}")
            print(f"{self.state.name} is sleeping (+30 energy)")
        
        elif decision.action_type == ActionType.COOK:
            # Cooking takes energy but reduces hunger and can be satisfying
            self.needs.hunger = max(0.0, self.needs.hunger - 40)
            self.needs.energy = max(0.0, self.needs.energy - 5)
            self.needs.fulfillment = min(100.0, self.needs.fulfillment + 5)
            if random.random() < 0.4:
                self.emotions.update_emotion(EmotionType.CONTENT, 0.3, "made a nice meal")
            print(f"{self.state.name} is cooking (-40 hunger)")
        
        elif decision.action_type == ActionType.EAT:
            # Eating satisfies hunger
            self.needs.hunger = max(0.0, self.needs.hunger - 35)
            self.needs.comfort = min(100.0, self.needs.comfort + 10)
            if random.random() < 0.3:
                self.emotions.update_emotion(EmotionType.CONTENT, 0.25, "enjoyed a meal")
            print(f"{self.state.name} is eating (-35 hunger)")
        
        elif decision.action_type == ActionType.SHOWER:
            # Showering/bathing improves hygiene and comfort significantly
            self.needs.hygiene = min(100.0, self.needs.hygiene + 50)
            self.needs.comfort = min(100.0, self.needs.comfort + 15)
            self.needs.energy = min(100.0, self.needs.energy + 5)
            self.emotions.update_emotion(EmotionType.CONTENT, 0.25, "feeling fresh and clean")
            print(f"{self.state.name} is bathing (+50 hygiene)")
        
        elif decision.action_type == ActionType.USE_TOILET:
            # Using bathroom relieves bladder
            self.needs.bladder = max(0.0, self.needs.bladder - 60)
            self.needs.comfort = min(100.0, self.needs.comfort + 15)
            print(f"{self.state.name} is using the bathroom (-60 bladder)")
    
    def decide_next_action(self, simulation) -> Optional[ActionDecision]:
        """
        SIMS-STYLE DECISION: Pure deterministic behavior based on needs.
        
        NO LLM is used here. Actions are chosen based on:
        1. Scheduled activities (work, school)
        2. Sleep schedule based on personality
        3. Need urgency scoring
        4. Available objects to satisfy needs
        
        LLM is only used elsewhere for conversations and thoughts.
        """
        current_location = self.get_current_location_name(simulation)
        current_hour = simulation.world_time.hour
        nearby_objects = self.get_nearby_objects(simulation)
        nearby_agents = self.get_nearby_agents(simulation)
        
        # Use the Sims behavior manager for deterministic decision
        sims_action = SimsBehaviorManager.choose_action(
            agent=self,
            simulation=simulation,
            nearby_objects=nearby_objects,
            nearby_agents=nearby_agents,
            current_location=current_location,
            current_hour=current_hour
        )
        
        # Store the current sims action
        self.current_sims_action = sims_action
        
        # Convert SimsAction to ActionDecision for compatibility
        decision = self._sims_action_to_decision(sims_action)
        
        if decision:
            self.recent_actions.append(decision.action_type.value)
            self.recent_actions = self.recent_actions[-10:]
            
            # Generate a simple inner thought (no LLM, just based on state)
            decision.inner_thought = self._generate_simple_inner_thought(sims_action)
        
        return decision
    
    def _sims_action_to_decision(self, sims_action: SimsAction) -> ActionDecision:
        """Convert a SimsAction to an ActionDecision for compatibility."""
        # Map sims action types to ActionType
        action_type_map = {
            "use_object": ActionType.USE_OBJECT,
            "sleep": ActionType.SLEEP,
            "socialize": ActionType.CONVERSE,
            "move_to": ActionType.MOVE,
            "wait": ActionType.WAIT,
            "scheduled": ActionType.WORK,  # Default for scheduled
        }
        
        action_type = action_type_map.get(sims_action.action_type, ActionType.WAIT)
        
        return ActionDecision(
            action_type=action_type,
            reasoning=sims_action.description,
            target_location=sims_action.target_location,
            target_sub_location=sims_action.target_sub_location,
            target_object=sims_action.target_object,
            target_agent=sims_action.target_agent,
            priority=sims_action.priority,
            expected_duration_minutes=sims_action.duration_minutes,
            inner_thought=""
        )
    
    def _generate_simple_inner_thought(self, sims_action: SimsAction) -> str:
        """Generate a simple inner thought based on the action (NO LLM)."""
        if sims_action.need_being_addressed:
            need_thoughts = {
                "energy": ["So tired...", "Need some rest.", "Time to relax.", "Can barely keep my eyes open."],
                "hunger": ["Getting hungry.", "Time to eat.", "My stomach is growling.", "I could really go for some food."],
                "bladder": ["Need to find a bathroom.", "Can't wait much longer.", "Really need to go."],
                "hygiene": ["Should freshen up.", "Time for a shower.", "Need to clean up."],
                "social": ["Would be nice to chat.", "Feeling social.", "I should catch up with someone."],
                "fun": [
                    "I need to do something fun.",
                    "This should be entertaining.",
                    "Time to unwind!",
                    "Looking forward to this.",
                    "I've been so bored lately.",
                    "This will be a nice break."
                ],
            }
            thoughts = need_thoughts.get(sims_action.need_being_addressed, ["..."])
            return random.choice(thoughts)
        
        if sims_action.action_type == "socialize" and sims_action.target_agent:
            return f"I'll go talk to {sims_action.target_agent}."
        
        if sims_action.action_type == "sleep":
            return "Time for bed."
        
        if sims_action.action_type == "move_to":
            return random.choice([
                "Let's see what's going on there.",
                "Time for a change of scenery.",
                "Wonder what I'll find...",
                "Should be interesting."
            ])
        
        if sims_action.action_type == "wait":
            return random.choice(["Just relaxing.", "Taking it easy.", "Enjoying the moment.", "..."])
        
        return "..."
    
    def prime_memories(self, simulation):
        """Prime the agent's memory with relevant initial information."""
        # Add initial memory of waking up
        self.form_memory(
            f"Started my day at {self.state.home_location_name}",
            MemoryType.OBSERVATION,
            importance=3.0,
            emotional_valence=0.1,
            location=self.state.home_location_name
        )
        
        # Set initial goals based on background
        if self.state.background:
            if "work" in self.state.background.lower():
                self.memory.short_term_goals.append("Go to work")
            if "eat" in self.state.background.lower():
                self.memory.short_term_goals.append("Have a meal")
    
    def setup(self, simulation):
        """Initial setup for the agent within the simulation."""
        self.prime_memories(simulation)

        # Start at home
        home_positions = simulation.world.get_sub_locations_positions_for_main_location(
            self.state.home_location_name
        )
        if home_positions:
            chosen_position = random.choice(home_positions)
            self.state.location = Point2D(x=chosen_position.x, y=chosen_position.y)
            print(f"Agent {self.state.agent_id} starting at ({chosen_position.x}, {chosen_position.y}) in {self.state.home_location_name}")

        self.is_setup = True
        return self.state
    
    def check_and_execute_scheduled_activity(self, current_time: datetime) -> bool:
        """
        Check if there's a scheduled activity that should be active now and execute it.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            True if a scheduled activity was started, False otherwise
        """
        if not self.daily_schedule:
            return False
        
        current_hour = current_time.hour
        current_minute = current_time.minute
        day_of_week = current_time.weekday()
        
        # Check if current activity is still active
        if self.state.current_scheduled_activity:
            if self.state.current_scheduled_activity.is_active_at(current_hour, current_minute, day_of_week):
                return True  # Continue current activity
            else:
                # Activity ended
                self.state.current_scheduled_activity = None
                self.state.is_busy = False
                self.state.objects_in_use = []
        
        # Find matching scheduled activity
        for activity in self.daily_schedule:
            if activity.is_active_at(current_hour, current_minute, day_of_week):
                # Start this scheduled activity
                self.state.start_scheduled_activity(activity, current_time)
                logger.debug(f"{self.state.name}: Started scheduled activity - {activity.description}")
                return True
        
        return False

    def tick(self, simulation):
        """Per-tick update for the agent: needs, movement, emotions, and decision-making."""
        # Ensure initial setup
        if not self.is_setup:
            self.setup(simulation)

        current_hour = simulation.world_time.hour
        
        # COMPREHENSIVE LOGGING: Track tick start state
        logger.debug(f"🔄 TICK START: {self.state.name} | Hour: {current_hour}:00 | Location: ({self.state.location.x}, {self.state.location.y}) | Status: {self.state.status} | Sleeping: {self.state.is_sleeping} | Has Path: {self.memory.current_path is not None and len(self.memory.current_path) > 0}")
        
        # Update needs with time awareness
        self.needs.tick(current_hour)
        
        # Update emotional state (natural decay over time)
        self.emotions.decay_over_time()
        
        # Check for midnight to reset daily observations
        if current_hour == 0 and simulation.tick_count > 0:
            self.memory.clear_daily_observations()
        
        # Stress builds up from unmet critical needs
        critical_count = len(self.needs.get_critical_needs())
        if critical_count > 0:
            self.emotions.stress_level = min(1.0, self.emotions.stress_level + 0.02 * critical_count)
            if critical_count >= 3:
                self.emotions.update_emotion(EmotionType.STRESSED, 0.5, "too many unmet needs")
        
        # Low wellbeing affects mood
        wellbeing = self.needs.get_wellbeing_score()
        if wellbeing < 40:
            self.emotions.mood_baseline = max(-1.0, self.emotions.mood_baseline - 0.01)
        elif wellbeing > 70:
            self.emotions.mood_baseline = min(1.0, self.emotions.mood_baseline + 0.005)

        # Put current location in memory
        self.memory.visited_locations.add(self.state.location)
        
        # Check and execute scheduled activities (takes priority over other actions)
        if self.check_and_execute_scheduled_activity(simulation.world_time):
            # Agent is following their schedule
            return self.state

        # Handle sleeping state - check if agent should wake up
        if self.state.is_sleeping:
            logger.info(f"😴 SLEEPING: {self.state.name} | Should be awake at {current_hour}:00? {self.personality.should_be_awake(current_hour)} | Wake hour: {self.personality.preferred_wake_hour} | Sleep hour: {self.personality.preferred_sleep_hour}")
            
            # Check if it's time to wake up based on personality
            if self.personality.should_be_awake(current_hour):
                # Time to wake up!
                self.state.is_sleeping = False
                self.state.status = "idle"
                logger.info(f"⏰ WAKE UP: {self.state.name} is waking up at {current_hour}:00")
                print(f"{self.state.name} is waking up at {current_hour}:00")
            else:
                # Still sleeping - don't move, stay in place and recover
                if self.memory.current_path:
                    logger.warning(f"🛑 CLEARING PATH: {self.state.name} had a path while sleeping (length: {len(self.memory.current_path)}), clearing it now")
                    self.memory.current_path = None
                self.state.status = "sleeping"
                # Passive energy recovery while sleeping
                self.needs.energy = min(100.0, self.needs.energy + 0.5)
                self.needs.comfort = min(100.0, self.needs.comfort + 0.2)
                logger.debug(f"💤 STILL SLEEPING: {self.state.name} recovering (Energy: {self.needs.energy:.1f})")
                return self.state

        # Follow the current path if we have one
        if self.memory.current_path:
            path_length = len(self.memory.current_path)
            start_pos = (self.state.location.x, self.state.location.y)
            
            logger.debug(f"🚶 MOVING: {self.state.name} following path (length: {path_length}) from {start_pos}")
            
            # Move multiple steps per tick for faster movement (5 steps per tick)
            steps_per_tick = 5
            steps_taken = 0
            for _ in range(min(steps_per_tick, len(self.memory.current_path))):
                if self.memory.current_path:
                    next_point = self.memory.current_path.pop(0)
                    self.state.location = Point2D(x=next_point.x, y=next_point.y)
                    steps_taken += 1

            end_pos = (self.state.location.x, self.state.location.y)
            logger.debug(f"📍 MOVED: {self.state.name} moved {steps_taken} steps to {end_pos} | Remaining path: {len(self.memory.current_path) if self.memory.current_path else 0}")
            
            if not self.memory.current_path:
                logger.info(f"🎯 ARRIVED: {self.state.name} reached destination at {end_pos}")
                self.memory.current_path = None  # Clear path when done
                self.state.is_in_conversation = False
                self.state.conversation_partner = None
                
                # Execute action at destination
                if self.state.current_action:
                    logger.debug(f"⚡ EXECUTING ACTION: {self.state.name} executing {self.state.current_action.action_type.value}")
                    self.execute_action(self.state.current_action, simulation)

            self.state.status = "moving"
            return self.state
        
        # Check if current activity is complete
        if self.state.is_busy and self.state.is_activity_complete(simulation.world_time):
            self.state.is_busy = False
            self.state.current_activity_description = ""

        # Make decision every few ticks
        current_tick = getattr(simulation, 'tick_count', 0)
        ticks_since_decision = current_tick - self.last_decision_time
        
        if current_tick - self.last_decision_time >= self.decision_cooldown:
            logger.debug(f"🤔 DECISION TIME: {self.state.name} making new decision (last decision: {ticks_since_decision} ticks ago)")
            decision = self.decide_next_action(simulation)

            if decision:
                self.last_decision_time = current_tick
                
                logger.info(f"💡 DECISION MADE: {self.state.name} | Action: {decision.action_type.value} | Target: {decision.target_location}/{decision.target_sub_location} | Object: {decision.target_object} | Priority: {decision.priority}")
                logger.debug(f"   Reasoning: {decision.reasoning}")
                
                # Update inner thought display
                if decision.inner_thought:
                    self.state.current_inner_thought = decision.inner_thought
                    logger.debug(f"   💭 {decision.inner_thought}")
                
                print(f"{self.state.name}: {decision.reasoning} → {decision.action_type.value}")
                if decision.inner_thought:
                    print(f"  💭 \"{decision.inner_thought}\"")

                # SAFETY CHECK: Never create paths for sleeping agents
                if self.state.is_sleeping:
                    logger.error(f"🚨 SLEEPING BUG: {self.state.name} tried to make decision while sleeping! Decision: {decision.action_type.value}")
                    return self.state

                # Get target position
                target_positions = []
                if decision.target_sub_location:
                    positions_map = simulation.world.get_map_of_sub_location_to_positions_for_main_location(
                        decision.target_location
                    )
                    target_positions = positions_map.get(decision.target_sub_location, [])
                    
                    # If specific sub-location not found, fall back to any position in the main location
                    if not target_positions:
                        target_positions = simulation.world.get_sub_locations_positions_for_main_location(
                            decision.target_location
                        )
                else:
                    target_positions = simulation.world.get_sub_locations_positions_for_main_location(
                        decision.target_location
                    )

                if target_positions:
                    target_pos = random.choice(target_positions)
                    current_pos = (self.state.location.x, self.state.location.y)
                    target_coords = (target_pos.x, target_pos.y)

                    # Only create path if not already there
                    if self.state.location.x != target_pos.x or self.state.location.y != target_pos.y:
                        logger.debug(f"🗺️ PATHFINDING: {self.state.name} finding path from {current_pos} to {target_coords}")
                        path = simulation.world.find_path(self.state.location, target_pos)
                        if path and len(path) > 0:
                            logger.info(f"✅ PATH CREATED: {self.state.name} created path (length: {len(path)}) to {decision.target_location}/{decision.target_sub_location}")
                            self.memory.current_path = path
                            self.state.current_action = decision
                            self.state.status = "moving"
                        else:
                            logger.warning(f"⚠️ NO PATH: {self.state.name} couldn't find path, executing action in place")
                            # Can't path, execute action anyway
                            self.execute_action(decision, simulation)
                            self.state.status = "acting"
                    else:
                        logger.info(f"✅ AT DESTINATION: {self.state.name} already at target {target_coords}, executing immediately")
                        # Already at target, execute immediately
                        self.execute_action(decision, simulation)
                        self.state.status = "acting"
                else:
                    # Log the issue and execute action anyway (stay in place)
                    logger.warning(f"⚠️ NO POSITIONS: {self.state.name} - No positions found for {decision.target_location}/{decision.target_sub_location}, executing in place")
                    self.execute_action(decision, simulation)
                    self.state.status = "acting"
        else:
            self.state.status = "waiting"

        return self.state
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the agent's current state for display."""
        return {
            "name": self.state.name,
            "location": f"({self.state.location.x}, {self.state.location.y})",
            "status": self.state.status,
            "activity": self.state.current_activity_description or "idle",
            "inner_thought": self.state.current_inner_thought or "",
            "mood": self.emotions.get_mood_description(),
            "emotion": self.emotions.primary_emotion.value,
            "emotion_intensity": f"{self.emotions.emotion_intensity:.0%}",
            "wellbeing": f"{self.needs.get_wellbeing_score():.0f}/100",
            "needs": {
                "energy": f"{self.needs.energy:.0f}",
                "hunger": f"{self.needs.hunger:.0f}",
                "social": f"{self.needs.social:.0f}",
                "fun": f"{self.needs.fun:.0f}",
                "fulfillment": f"{self.needs.fulfillment:.0f}",
            },
            "critical_needs": self.needs.get_critical_needs(),
            "personality": self.personality.get_description(),
            "is_sleeping": self.state.is_sleeping,
            "is_in_conversation": self.state.is_in_conversation,
            "conversation_partner": self.state.conversation_partner,
        }
    
    def get_relationships_summary(self) -> List[Dict[str, str]]:
        """Get a summary of all relationships."""
        relationships = []
        for agent_name, score in self.state.relationship_scores.items():
            relationships.append({
                "agent": agent_name,
                "relationship": self.state.get_relationship_description(agent_name),
                "score": f"{score:.2f}",
                "recent_interactions": self.memory.get_relationship_history(agent_name)[-3:]
            })
        return relationships