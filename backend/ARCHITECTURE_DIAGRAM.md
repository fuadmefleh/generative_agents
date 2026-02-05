# Agent Generation System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        AGENT FACTORY                                    │
│                   (agent_factory.py)                                    │
│                                                                         │
│  Creates complete agents with emergent personalities                    │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │
                            ├─────────────────┐
                            │                 │
                            ▼                 ▼
         ┌──────────────────────────┐   ┌─────────────────────────┐
         │  LIFE STORY GENERATOR    │   │   ROUTINE GENERATOR     │
         │  (life_story_generator)  │   │  (routine_generator)    │
         │                          │   │                         │
         │  Generates:              │   │  Generates:             │
         │  • Life events           │   │  • Morning routine      │
         │  • Personality impacts   │   │  • Work schedule        │
         │  • Core values           │   │  • Meal times           │
         │  • Fears & aspirations   │   │  • Evening activities   │
         │                          │   │  • Bedtime routine      │
         └────────┬─────────────────┘   └───────────┬─────────────┘
                  │                                 │
                  │                                 │
                  ▼                                 ▼
         ┌──────────────────────────┐   ┌─────────────────────────┐
         │    LIFE STORY            │   │  SCHEDULED ACTIVITIES   │
         │    (life_story.py)       │   │  (agent.py)             │
         │                          │   │                         │
         │  • LifeEvent             │   │  • Wake up              │
         │  • LifeStory             │   │  • Use toilet           │
         │  • PersonalityImpact     │   │  • Shower               │
         │  • EventCategory         │   │  • Cook/Eat             │
         │                          │   │  • Work                 │
         │  Age 4-Current:          │   │  • Leisure              │
         │  [Event] → [Event] → ... │   │  • Sleep                │
         │     ↓        ↓            │   │                         │
         │  Shapes personality       │   │  Each with:             │
         │                          │   │  • Time (hour:minute)   │
         └────────┬─────────────────┘   │  • Location             │
                  │                     │  • Objects used         │
                  │                     │  • Sub-activities       │
                  │                     └───────────┬─────────────┘
                  │                                 │
                  └──────────┬──────────────────────┘
                             │
                             ▼
                  ┌──────────────────────────┐
                  │       AGENT              │
                  │       (agent.py)         │
                  │                          │
                  │  Components:             │
                  │  ├─ State               │
                  │  ├─ Personality ◄────────┼─── Shaped by life story
                  │  │   (with life_story)  │
                  │  ├─ Memory              │
                  │  ├─ Emotions            │
                  │  ├─ Needs               │
                  │  ├─ Job                 │
                  │  └─ Daily Schedule ◄─────┼─── Generated routines
                  │                          │
                  └────────┬─────────────────┘
                           │
                           │ Runtime
                           │
                           ▼
         ┌──────────────────────────────────────┐
         │    SIMULATION INTEGRATION            │
         │                                      │
         │  simulation_tick():                  │
         │    for agent in agents:              │
         │      if agent.check_scheduled():     │
         │        # Agent follows routine       │
         │        status = agent.get_status()   │
         │        # Know exactly what they do   │
         └──────────────────────────────────────┘
```

## Data Flow Example

```
1. CREATE AGENT
   ├─ Name: "Emma Rodriguez"
   ├─ Age: 32
   └─ Occupation: Artist

2. GENERATE LIFE STORY (via LLM)
   ├─ Age 4-5: Started drawing, showed artistic talent
   │   └─ Impact: +0.15 openness
   ├─ Age 14-15: Won art competition, gained confidence  
   │   └─ Impact: +0.10 openness, -0.08 neuroticism
   ├─ Age 22-23: Graduated art school, first exhibition
   │   └─ Impact: +0.12 conscientiousness
   └─ ... (10+ more events)
   
   Result: Personality shaped by creative experiences

3. GENERATE DAILY ROUTINE
   Based on personality (high openness, artist):
   
   07:00 - Wake up (in bedroom, using bed)
   07:05 - Use toilet (in bathroom, using toilet)
   07:10 - Shower (in bathroom, using shower)
   07:25 - Get dressed (in bedroom, using closet)
   07:35 - Make coffee (in kitchen, using coffee_maker)
   08:00 - Eat breakfast (in kitchen, using table)
   09:00 - Work at studio (painting, using easel, canvas)
   12:00 - Lunch (cook, eat)
   14:00 - Continue work
   18:00 - Dinner prep (look in fridge, cook)
   20:00 - Leisure (based on interests)
   22:00 - Bedtime routine
   23:00 - Sleep (in bedroom, using bed)

4. RUNTIME TRACKING
   Time: 07:10 AM
   └─ check_scheduled_activity()
      └─ Found: SHOWER activity
         ├─ Activity type: ActionType.SHOWER
         ├─ Location: "home" > "bathroom"
         ├─ Objects: ["shower"]
         ├─ Status: "taking a shower"
         └─ get_detailed_status() →
             "Emma Rodriguez is taking a shower in 
              the bathroom using the shower."
```

## Activity Granularity

```
MORNING ROUTINE DETAIL:

07:00:00 ┌──────────────────────────────┐
         │ WAKE_UP                      │
         │ • Location: bedroom          │
07:05:00 │ • Object: bed                │
         ├──────────────────────────────┤
         │ USE_TOILET                   │
         │ • Location: bathroom         │
07:10:00 │ • Object: toilet             │
         ├──────────────────────────────┤
         │ SHOWER                       │
         │ • Location: bathroom         │
         │ • Object: shower             │
07:25:00 │ • Sub-activity: BRUSH_TEETH  │
         ├──────────────────────────────┤
         │ GET_DRESSED                  │
         │ • Location: bedroom          │
07:35:00 │ • Objects: closet, mirror    │
         ├──────────────────────────────┤
         │ LOOK_IN_FRIDGE               │
         │ • Location: kitchen          │
07:37:00 │ • Object: fridge             │
         ├──────────────────────────────┤
         │ MAKE_COFFEE + COOK           │
         │ • Location: kitchen          │
         │ • Objects: coffee_maker,     │
08:00:00 │   stove, cupboard            │
         ├──────────────────────────────┤
         │ EAT                          │
         │ • Location: kitchen          │
         │ • Objects: table, chair      │
08:20:00 │ • Sub-activity: DRINK        │
         └──────────────────────────────┘

You can see EXACTLY what the agent is doing at any moment!
```

## System Components

```
MODELS:
├─ life_story.py
│  ├─ LifeEvent (single event with impacts)
│  ├─ LifeStory (complete history)
│  ├─ PersonalityImpact (trait changes)
│  └─ EventCategory (event types)
│
├─ agent.py (ENHANCED)
│  ├─ ActionType (40+ detailed actions)
│  ├─ ScheduledActivity (with objects, timing)
│  ├─ AgentPersonality (with life_story)
│  └─ AgentState (detailed tracking)
│
└─ models.py
   ├─ Point2D
   └─ Plan

GENERATORS:
├─ life_story_generator.py
│  └─ LifeStoryGenerator
│     ├─ generate_life_story()
│     └─ apply_to_personality()
│
├─ routine_generator.py
│  └─ RoutineGenerator
│     ├─ generate_daily_routine()
│     ├─ _generate_morning_routine()
│     ├─ _generate_work_routine()
│     └─ _generate_evening_routine()
│
└─ agent_factory.py
   └─ AgentFactory
      ├─ create_agent_with_life_story()
      └─ create_multiple_agents()

TESTING:
├─ test_agent_generation.py
├─ example_agent_usage.py
└─ AGENT_GENERATION_README.md
```

## Key Methods

```python
# Create agent
agent = factory.create_agent_with_life_story(...)

# Check schedule (in simulation loop)
if agent.check_and_execute_scheduled_activity(time):
    # Agent is following their routine
    pass

# Get what they're doing
status = agent.state.get_detailed_status()
# Returns: natural language description

# Access details
agent.state.current_activity_type    # ActionType enum
agent.state.objects_in_use          # List of objects
agent.state.current_sub_location    # Exact location
agent.state.is_sleeping             # Boolean flags
agent.state.is_busy                 # Boolean flags
```

## Integration Flow

```
┌─────────────────┐
│   Start Sim     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Create Agents   │◄─── AgentFactory
│ with Stories    │     ├─ Generate life stories
└────────┬────────┘     ├─ Shape personalities
         │              └─ Create routines
         │
         ▼
┌─────────────────┐
│  Simulation     │
│  Loop/Tick      │
└────────┬────────┘
         │
         ├────────────────────────┐
         │                        │
         ▼                        ▼
┌─────────────────┐      ┌──────────────────┐
│ For each agent: │      │  Update world    │
│                 │      │  state, render   │
│ 1. Check        │      └──────────────────┘
│    scheduled    │
│    activity     │
│                 │
│ 2. If following │
│    schedule,    │
│    continue     │
│                 │
│ 3. Otherwise,   │
│    make         │
│    decisions    │
│                 │
│ 4. Update       │
│    needs,       │
│    emotions     │
└─────────────────┘
```
