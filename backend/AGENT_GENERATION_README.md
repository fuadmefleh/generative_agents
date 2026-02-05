# Agent Generation System with Sims-Like Behavior

This system creates agents with **emergent personalities** shaped by life stories, complete with **Sims-like autonomous behavior** driven by needs.

## 🎮 Behavior Model: The Sims Approach

### Key Principle: LLM is NOT used for action selection!

Agents behave like The Sims - their actions are driven **deterministically** by:
1. **Needs** that decay over time (hunger, energy, bladder, social, fun, hygiene)
2. **Available objects** that satisfy those needs
3. **Urgency scoring** - most urgent need gets addressed first
4. **Personality** - influences thresholds and preferences

### Where LLM IS Used (Limited)
- **Conversations** between agents - generating dialogue
- **Internal thoughts** - occasional rich inner monologue (20% of thoughts)
- **Memory/reflection** - processing experiences

## Overview

### Key Features

1. **Sims-Style Needs System**: Agents have decaying needs that drive autonomous behavior
2. **Object Interactions**: Objects satisfy specific needs (toilet → bladder, bed → energy, etc.)
3. **Life Story Generation**: Automatically generates life events that shape personality
4. **Emergent Personalities**: Personality traits influence behavior thresholds
5. **Detailed Daily Routines**: Scheduled activities for work, sleep, etc.

## 🎯 The Needs System

### Core Needs (like The Sims)

| Need | Decay Direction | Objects That Satisfy |
|------|----------------|---------------------|
| **Energy** | Decreases | bed, sofa, coffee_maker |
| **Hunger** | Increases | refrigerator, stove, kitchen_table |
| **Bladder** | Increases | toilet |
| **Hygiene** | Decreases | shower, bathtub, sink |
| **Social** | Decreases | Talk to other agents, computer (online) |
| **Fun** | Decreases | tv, computer, bookshelf, piano, stereo |
| **Comfort** | Decreases | sofa, bed, bathtub |

### How Decisions Work

```python
# Agents choose actions based on urgency scoring - NO LLM!
urgencies = SimsBehaviorManager.calculate_need_urgency(agent.needs)
# Returns: [(NeedType.BLADDER, 95.0), (NeedType.HUNGER, 72.0), ...]

# Agent finds object to satisfy most urgent need
action = SimsBehaviorManager.choose_action(
    agent=agent,
    simulation=simulation,
    nearby_objects=nearby_objects,
    nearby_agents=nearby_agents,
    current_location=current_location,
    current_hour=current_hour
)
# Returns: SimsAction(action_type="use_object", target_object="toilet", ...)
```

### Object Interaction Effects

Each object has defined effects on needs:

```python
# From sims_behavior.py
OBJECT_INTERACTIONS = {
    "toilet": ObjectInteraction(
        duration_minutes=5,
        need_effects={NeedType.BLADDER: -80},  # Greatly reduces bladder need
        action_name="Using the toilet"
    ),
    "shower": ObjectInteraction(
        duration_minutes=15,
        need_effects={
            NeedType.HYGIENE: 60,
            NeedType.COMFORT: 10,
            NeedType.ENERGY: 10
        },
        action_name="Taking a shower"
    ),
    "tv": ObjectInteraction(
        duration_minutes=60,
        need_effects={
            NeedType.FUN: 40,
            NeedType.COMFORT: 10,
        },
        action_name="Watching TV"
    ),
    # ... many more
}
```

## Quick Start

### Creating a Single Agent

```python
from app.factory.agent_factory import AgentFactory
from app.models.agent import JobType
from app.models.models import Point2D

factory = AgentFactory()

agent = factory.create_agent_with_life_story(
    agent_id="agent_001",
    name="Emma Rodriguez",
    location=Point2D(x=25.0, y=30.0),
    home_location_name="Emma's Home",
    age=32,
    occupation=JobType.ARTIST,
    cultural_background="Hispanic American"
)
```

### Running the Simulation

```python
# In simulation tick - agent decides action automatically
decision = agent.decide_next_action(simulation)
# This is 100% deterministic - NO LLM calls!

if decision:
    print(f"{agent.state.name}: {decision.reasoning}")
    # Example: "Emma Rodriguez: Using the toilet"
    
    agent.execute_action(decision, simulation)
    # Automatically applies need effects from sims_behavior.py
```

## Architecture

### Core Components

#### 1. Sims Behavior System (`app/models/sims_behavior.py`) ⭐ NEW

**Purpose:** Handles ALL action selection deterministically

**Key Classes:**
- `NeedType`: Enum of all needs (ENERGY, HUNGER, BLADDER, etc.)
- `ObjectInteraction`: Defines what objects do (duration, need effects)
- `SimsAction`: The action to take (use_object, socialize, sleep, wait)
- `SimsBehaviorManager`: Main decision engine

**Key Methods:**
```python
# Calculate urgency of all needs
urgencies = SimsBehaviorManager.calculate_need_urgency(agent.needs)

# Choose what to do (deterministic!)
action = SimsBehaviorManager.choose_action(agent, simulation, ...)

# Apply effects after action completes
SimsBehaviorManager.apply_action_effects(agent, action)
```

#### 2. Conversation System (`app/models/conversation.py`) ⭐ NEW

**Purpose:** LLM is ONLY used here - for generating dialogue

**Key Classes:**
- `ConversationManager`: Manages conversations between agents
- `InnerLifeManager`: Generates occasional rich thoughts using LLM

```python
# When agents socialize, LLM generates dialogue
conversation_mgr = ConversationManager(llm_client)
turn = conversation_mgr.generate_conversation_dialogue(
    speaker_agent, listener_agent, conversation
)
# Returns: ConversationTurn(speaker="Emma", dialogue="Hey, how's it going?", ...)
```

#### 3. Life Story System (`app/models/life_story.py`)

**Models:**
- `LifeEvent`: A single significant event with emotional impact and personality changes
- `LifeStory`: Complete life history with events, values, fears, and aspirations
- `PersonalityImpact`: How an event changes personality traits
- `EventCategory`: Types of events (FAMILY, EDUCATION, SOCIAL, ACHIEVEMENT, TRAUMA, etc.)

**Example Life Event:**
```python
LifeEvent(
    age=14,
    category=EventCategory.ACHIEVEMENT,
    description="Won regional art competition, discovered passion for creative expression",
    emotional_impact=0.7,  # Positive
    personality_impact=PersonalityImpact(
        openness_delta=0.15,
        neuroticism_delta=-0.10
    )
)
```

#### 2. Life Story Generator (`app/factory/life_story_generator.py`)

**Purpose:** Uses LLM to generate coherent life stories that shape personality

**Process:**
1. Calculate age periods (every 2 years from age 4)
2. Generate 2-3 events per period using LLM
3. Events build on each other naturally
4. Apply cumulative personality impacts

**Key Method:**
```python
life_story = generator.generate_life_story(
    agent_name="Emma Rodriguez",
    current_age=32,
    occupation="artist",
    cultural_background="Hispanic American"
)
```

#### 3. Routine Generator (`app/factory/routine_generator.py`)

**Purpose:** Creates detailed daily schedules based on personality and life story

**Generated Routines Include:**
- Morning routine (wake up, toilet, shower, brush teeth, get dressed)
- Meal preparation and eating (look in fridge, cook, eat, wash dishes)
- Work schedule (if employed)
- Leisure activities (based on personality and interests)
- Evening routine
- Bedtime routine (toilet, brush teeth, sleep)

**Example Activities:**
```python
ScheduledActivity(
    start_hour=7,
    start_minute=5,
    end_hour=7,
    end_minute=10,
    activity_type=ActionType.USE_TOILET,
    location="home",
    sub_location="bathroom",
    description="Using the bathroom",
    specific_objects=["toilet"],
    priority=9
)
```

#### 4. Agent Factory (`app/factory/agent_factory.py`)

**Purpose:** Orchestrates the creation of complete agents

**Process:**
1. Generate base personality
2. Generate life story using LLM
3. Apply life story impacts to personality
4. Create job (if applicable)
5. Generate daily routine based on personality
6. Assemble complete agent

## Detailed Activity Tracking

### Activity Types

The system tracks **40+ activity types** including:

**Basic needs:**
- `SLEEP`, `WAKE_UP`, `USE_TOILET`, `SHOWER`, `BRUSH_TEETH`, `GET_DRESSED`

**Kitchen/Food:**
- `COOK`, `EAT`, `DRINK`, `LOOK_IN_FRIDGE`, `LOOK_IN_CUPBOARD`, `WASH_DISHES`, `MAKE_COFFEE`

**Living:**
- `WATCH_TV`, `READ_BOOK`, `LISTEN_MUSIC`, `USE_COMPUTER`, `CLEAN_ROOM`, `DO_LAUNDRY`

**Social:**
- `CHAT`, `MAKE_PHONE_CALL`, `CONVERSE`

**Work:**
- `WORK`, `STUDY`, `ATTEND_CLASS`

**Outdoor:**
- `WALK`, `JOG`, `EXERCISE`, `SIT_ON_BENCH`

### Tracking What Agents Are Doing

```python
# Check scheduled activity
if agent.check_and_execute_scheduled_activity(simulation_time):
    # Get detailed status
    status = agent.state.get_detailed_status()
    # Example: "Emma Rodriguez is taking a shower in the bathroom using the shower."
    
    # Access specific details
    print(f"Activity: {agent.state.current_activity_type}")  # ActionType.SHOWER
    print(f"Location: {agent.state.current_location_name}")  # "home"
    print(f"Sub-location: {agent.state.current_sub_location}")  # "bathroom"
    print(f"Objects: {agent.state.objects_in_use}")  # ["shower"]
    print(f"Sleeping: {agent.state.is_sleeping}")  # False
    print(f"Busy: {agent.state.is_busy}")  # True
```

## Integration with Simulation

### In Your Simulation Tick/Update Loop

```python
def simulation_tick(agents, world_time):
    for agent in agents:
        # Priority 1: Check scheduled activities
        if agent.check_and_execute_scheduled_activity(world_time):
            # Agent is following their routine
            # You can access exactly what they're doing
            continue
        
        # Priority 2: Other agent logic (pathfinding, decisions, etc.)
        # ...
```

### Enhanced Agent State

The `AgentState` class now includes:

```python
class AgentState:
    # ... existing fields ...
    
    # Detailed activity tracking
    current_activity_type: Optional[ActionType]  # What action they're doing
    current_location_name: Optional[str]  # Where they are
    current_sub_location: Optional[str]  # Specific room/area
    objects_in_use: List[str]  # Objects being used (bed, fridge, toilet, etc.)
    current_scheduled_activity: Optional[ScheduledActivity]  # The schedule item
```

## Personality System

### Big Five Traits (0-1 scale)
- **Openness**: Creativity, curiosity, willingness to try new things
- **Conscientiousness**: Organization, discipline, reliability
- **Extraversion**: Sociability, energy from others, outgoingness
- **Agreeableness**: Cooperation, trust, warmth
- **Neuroticism**: Emotional instability, anxiety, stress

### How Life Events Shape Personality

Events have `PersonalityImpact` that modifies traits:

```python
# Example: Positive social event
personality_impact=PersonalityImpact(
    extraversion_delta=0.1,  # More outgoing
    agreeableness_delta=0.05  # More cooperative
)

# Example: Traumatic event
personality_impact=PersonalityImpact(
    neuroticism_delta=0.15,  # More anxious
    extraversion_delta=-0.1  # More withdrawn
)
```

Multiple events accumulate to shape the final personality.

## Testing

### Run the Test Suite

```bash
cd backend
python test_agent_generation.py
```

This will:
1. Generate a single agent with full life story
2. Generate multiple diverse agents
3. Test activity tracking system
4. Save results to JSON files

### Run the Examples

```bash
python example_agent_usage.py
```

Shows practical examples of:
- Creating agents
- Tracking activities
- Integrating with simulation

## Files Created/Modified

### New Files
- `backend/app/models/life_story.py` - Life story models
- `backend/app/factory/life_story_generator.py` - LLM-based story generation
- `backend/app/factory/routine_generator.py` - Daily routine creation
- `backend/app/factory/agent_factory.py` - Complete agent creation
- `backend/test_agent_generation.py` - Test suite
- `backend/example_agent_usage.py` - Usage examples
- `backend/AGENT_GENERATION_README.md` - This file

### Modified Files
- `backend/app/models/agent.py`:
  - Added 30+ new `ActionType` values for detailed activities
  - Enhanced `ScheduledActivity` with minute precision and object tracking
  - Added `life_story` field to `AgentPersonality`
  - Enhanced `AgentState` with detailed activity tracking
  - Added `check_and_execute_scheduled_activity()` method
  - Added `get_detailed_status()` method for natural language status

## API Reference

### AgentFactory

```python
class AgentFactory:
    def create_agent_with_life_story(
        agent_id: str,
        name: str,
        location: Point2D,
        home_location_name: str,
        age: int = None,  # Random 22-65 if not provided
        occupation: Optional[JobType] = None,
        cultural_background: Optional[str] = None,
        generate_life_story: bool = True  # Set False for quick testing
    ) -> Agent
    
    def create_multiple_agents(
        count: int,
        location_generator: callable,
        home_locations: List[str],
        occupations: Optional[List[JobType]] = None,
        cultural_backgrounds: Optional[List[str]] = None
    ) -> List[Agent]
```

### LifeStoryGenerator

```python
class LifeStoryGenerator:
    def generate_life_story(
        agent_name: str,
        current_age: int,
        occupation: Optional[str] = None,
        cultural_background: Optional[str] = None,
        personality_seed: Optional[dict] = None
    ) -> LifeStory
    
    def apply_life_story_to_personality(
        base_personality: dict,
        life_story: LifeStory
    ) -> dict
```

### RoutineGenerator

```python
class RoutineGenerator:
    @staticmethod
    def generate_daily_routine(
        personality: AgentPersonality,
        job: Optional[AgentJob] = None,
        life_story: Optional[LifeStory] = None,
        home_location: str = "home"
    ) -> List[ScheduledActivity]
```

## Performance Considerations

### LLM Generation
- Life story generation takes ~5-30 seconds per agent (depending on LLM)
- Set `generate_life_story=False` for quick testing/development
- Consider pre-generating agents and caching them

### Memory Usage
- Each agent with full life story: ~10-50 KB
- Daily schedule: ~1-5 KB per agent
- Scale tested up to 100+ agents

### Optimization Tips
1. Generate agents at simulation start, not runtime
2. Cache generated agents to disk
3. Use fallback stories if LLM fails
4. Consider simplified routines for background NPCs

## Future Enhancements

Potential improvements:
- [ ] Dynamic schedule adjustments based on personality changes
- [ ] Social routines (coordinated activities between agents)
- [ ] Seasonal/holiday schedule variations
- [ ] Memory of past activities influences future choices
- [ ] Routine learning from experience
- [ ] More granular sub-activities (e.g., brushing teeth -> get toothbrush, apply toothpaste, brush, rinse)

## Troubleshooting

### "No life story generated"
- Check LLM is running (Ollama on port 11434)
- Verify model name in `OllamaClient` initialization
- Check logs in `agent_generation_test.log`

### "Activity not found at expected time"
- Verify `ScheduledActivity.is_active_at()` logic
- Check day_of_week mapping (0=Monday, 6=Sunday)
- Ensure simulation time format matches

### "Agent not following schedule"
- Verify `check_and_execute_scheduled_activity()` is called in simulation tick
- Check that it's called before other decision-making logic
- Confirm `daily_schedule` list is populated

## Support

For questions or issues:
1. Check the example files
2. Review test output
3. Check logs in `agent_generation_test.log`
4. Verify LLM connection and model availability
