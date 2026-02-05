from app.models.models import Point2D
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from enum import Enum
from pydantic import validator
from datetime import datetime
from typing import List, Tuple, Set
import random
import csv
import re
from app.models.world import WorldMap, WorldState, World, WorldMapTileType
from app.models.agent import (
    Agent, AgentPersonality, AgentJob, JobType, 
    ScheduledActivity, ActionType
)
from app.factory.world_factory import create_world


def create_job_for_agent(job_name: str, workplace: str, personality: AgentPersonality) -> Optional[AgentJob]:
    """Create a job with appropriate schedule based on job type."""
    job_type_map = {
        "barista": JobType.BARISTA,
        "artist": JobType.ARTIST,
        "student": JobType.STUDENT,
        "teacher": JobType.TEACHER,
        "chef": JobType.CHEF,
        "organizer": JobType.ORGANIZER,
        "writer": JobType.WRITER,
        "musician": JobType.MUSICIAN,
    }
    
    job_type = job_type_map.get(job_name.lower(), JobType.UNEMPLOYED)
    
    if job_type == JobType.UNEMPLOYED:
        return None
    
    work_schedule = []
    
    # Create schedules based on job type
    if job_type == JobType.BARISTA:
        # Morning/afternoon shifts
        if personality.morning_person:
            work_schedule.append(ScheduledActivity(
                start_hour=7,
                end_hour=15,
                activity_type=ActionType.WORK,
                location=workplace,
                description=f"Working shift at {workplace}",
                priority=9,
                days_of_week=[0, 1, 2, 3, 4]  # Mon-Fri
            ))
        else:
            work_schedule.append(ScheduledActivity(
                start_hour=14,
                end_hour=22,
                activity_type=ActionType.WORK,
                location=workplace,
                description=f"Working shift at {workplace}",
                priority=9,
                days_of_week=[0, 1, 2, 3, 4]  # Mon-Fri
            ))
    
    elif job_type == JobType.ORGANIZER:
        # Standard office hours with meetings
        work_schedule.append(ScheduledActivity(
            start_hour=9,
            end_hour=17,
            activity_type=ActionType.WORK,
            location=workplace,
            description="Community organizing work",
            priority=8,
            days_of_week=[0, 1, 2, 3, 4]  # Mon-Fri
        ))
    
    elif job_type == JobType.STUDENT:
        # Classes throughout the day
        work_schedule.extend([
            ScheduledActivity(
                start_hour=9,
                end_hour=11,
                activity_type=ActionType.ATTEND_CLASS,
                location=workplace,
                description="Attending morning class",
                priority=9,
                days_of_week=[0, 2, 4]  # Mon, Wed, Fri
            ),
            ScheduledActivity(
                start_hour=13,
                end_hour=15,
                activity_type=ActionType.ATTEND_CLASS,
                location=workplace,
                description="Attending afternoon class",
                priority=9,
                days_of_week=[1, 3]  # Tue, Thu
            ),
        ])
    
    elif job_type == JobType.ARTIST:
        # Flexible creative hours
        preferred_hour = personality.preferred_wake_hour + 2
        work_schedule.append(ScheduledActivity(
            start_hour=preferred_hour,
            end_hour=min(preferred_hour + 4, 20),
            activity_type=ActionType.WORK,
            location=workplace,
            description="Working on art",
            priority=7,
            days_of_week=[0, 1, 2, 3, 4, 5]  # Most days
        ))
    
    return AgentJob(
        job_type=job_type,
        workplace_location=workplace,
        work_schedule=work_schedule,
        job_satisfaction=random.uniform(0.5, 0.9)
    )


def parse_personality_from_background(background: str) -> AgentPersonality:
    """Parse agent background text to generate appropriate personality traits."""
    background_lower = background.lower()
    
    # Default values
    extraversion = 0.5
    conscientiousness = 0.5
    openness = 0.5
    agreeableness = 0.6
    neuroticism = 0.4
    morning_person = True
    preferred_wake_hour = 7
    preferred_sleep_hour = 22
    work_ethic = 0.6
    spontaneity = 0.4
    favorite_activities = ["walking", "reading"]
    
    # Parse extraversion
    if any(word in background_lower for word in ["extrovert", "social", "outgoing", "energetic", "loves meeting"]):
        extraversion = random.uniform(0.7, 0.95)
    elif any(word in background_lower for word in ["introvert", "quiet", "reserved", "shy", "prefers alone"]):
        extraversion = random.uniform(0.15, 0.4)
    
    # Parse openness/creativity
    if any(word in background_lower for word in ["creative", "artist", "curious", "imaginative", "dreams"]):
        openness = random.uniform(0.7, 0.95)
    
    # Parse conscientiousness
    if any(word in background_lower for word in ["organized", "disciplined", "punctual", "reliable"]):
        conscientiousness = random.uniform(0.7, 0.9)
    elif any(word in background_lower for word in ["spontaneous", "flexible", "casual"]):
        conscientiousness = random.uniform(0.3, 0.5)
    
    # Parse agreeableness
    if any(word in background_lower for word in ["warm", "friendly", "kind", "helpful", "caring"]):
        agreeableness = random.uniform(0.7, 0.95)
    elif any(word in background_lower for word in ["competitive", "independent", "assertive"]):
        agreeableness = random.uniform(0.3, 0.5)
    
    # Parse wake/sleep times from background
    import re
    wake_match = re.search(r'wakes?\s*up\s*(?:around|at)?\s*(\d{1,2})\s*(am|pm)?', background_lower)
    if wake_match:
        hour = int(wake_match.group(1))
        if wake_match.group(2) == 'pm' and hour != 12:
            hour += 12
        preferred_wake_hour = hour
        morning_person = hour < 8
    
    sleep_match = re.search(r'(?:goes?\s*to\s*)?sleep\s*(?:around|at|by)?\s*(\d{1,2})\s*(am|pm)?', background_lower)
    if sleep_match:
        hour = int(sleep_match.group(1))
        if sleep_match.group(2) == 'pm' and hour != 12:
            hour += 12
        elif sleep_match.group(2) == 'am':
            hour = hour  # Early morning sleep
        preferred_sleep_hour = hour
    
    # Parse favorite activities
    activities = []
    activity_keywords = {
        "coffee": "socializing",
        "reading": "reading",
        "art": "crafts",
        "painting": "crafts",
        "cooking": "cooking",
        "exercise": "exercise",
        "walk": "walking",
        "music": "music",
        "games": "games",
    }
    for keyword, activity in activity_keywords.items():
        if keyword in background_lower:
            activities.append(activity)
    
    if activities:
        favorite_activities = list(set(activities))[:4]
    
    return AgentPersonality(
        openness=openness,
        conscientiousness=conscientiousness,
        extraversion=extraversion,
        agreeableness=agreeableness,
        neuroticism=neuroticism,
        morning_person=morning_person,
        preferred_wake_hour=preferred_wake_hour,
        preferred_sleep_hour=preferred_sleep_hour,
        work_ethic=work_ethic,
        spontaneity=spontaneity,
        favorite_activities=favorite_activities
    )


class Simulation(BaseModel):
    
    """Representation of a simulation instance."""
    id: str
    name: str
    description: Optional[str] = None
    world_time: datetime = Field(default_factory=lambda: datetime(1990, 1, 1, 5, 0, 0))
    parameters: Dict[str, Any] = Field(default_factory=dict)
    world: World = Field(...)
    agents: Dict[str, Agent] = Field(default_factory=dict)
    tick_count: int = 0

    @staticmethod
    def create( logger):
        
        from app.models.agent import AgentState

        # Initialize World
        logger.info("Initializing world...")
        world_map = create_world(
            name="Demo Dungeon",
            width=75,
            height=75,
            num_rooms=random.randint(12, 18),
            seed=42
        )
        world_state = WorldState(time=datetime.utcnow())
        world = World(map=world_map, state=world_state)

        # Make some initial agents (optional)
        initial_agents = {}
        # load agents from csv at map/agents.csv
        logger.info("Spawning initial agents...")

        with open("./app/map/agents.csv", "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                agent_id = row["id"]
                agent_name = row["name"]
                agent_background = row["background"]
                agent_location = row["main_location"]
                agent_job_name = row.get("job", "unemployed")
                agent_workplace = row.get("workplace", agent_location)
                
                free_tiles = world.get_free_tiles_in_main_sector(agent_location)
                if free_tiles:
                    # Pick a random free tile in the specified main location
                    tile = random.choice(free_tiles)
                    agent_location_point = Point2D(x=tile.location.x, y=tile.location.y)
                    print(f"Spawning agent {agent_name} at ({agent_location_point.x}, {agent_location_point.y})")
                    
                    # Generate personality from background
                    personality = parse_personality_from_background(agent_background)
                    print(f"  Personality: {personality.get_description()}")
                    print(f"  Wake: {personality.preferred_wake_hour}:00, Sleep: {personality.preferred_sleep_hour}:00")
                    print(f"  Favorite activities: {', '.join(personality.favorite_activities)}")
                    
                    # Create agent
                    agent = Agent.create(
                        agent_id=agent_id,
                        name=agent_name,
                        location=agent_location_point,
                        home_location_name=row["main_location"],
                        personality=personality
                    )
                    agent.state.background = agent_background
                    
                    # Assign job
                    if agent_job_name and agent_job_name.lower() != "unemployed":
                        agent.job = create_job_for_agent(agent_job_name, agent_workplace, personality)
                        if agent.job:
                            print(f"  Job: {agent.job.job_type.value} at {agent.job.workplace_location}")
                            print(f"  Work schedule: {len(agent.job.work_schedule)} shifts")
                    
                    world.state.agents[agent_id] = agent.state
                    initial_agents[agent_id] = agent

        return Simulation(
            id="sim_001",
            name="Demo Simulation",
            description="A demo simulation with a dungeon world and initial agents.",
            world=world,
            agents=initial_agents,
            parameters={
                "tick_interval_seconds": 1,
                "max_ticks": 1000
            }
        )


    def tick(self):
        """Advance the simulation by one tick."""
        # Each tick is 6 minutes in world time
        from datetime import timedelta
        self.world_time += timedelta(minutes=6)
        self.tick_count += 1
        
        # Tick each agent
        for agent_id, agent in self.agents.items():
            agent.tick(self)