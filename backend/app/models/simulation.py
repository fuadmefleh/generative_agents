from app.models.models import Point2D
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from enum import Enum
from pydantic import validator
from datetime import datetime
from typing import List, Tuple, Set
import random
import csv
from app.models.world import WorldMap, WorldState, World, WorldMapTileType
from app.models.agent import Agent
from app.factory.world_factory import create_world


class Simulation(BaseModel):
    
    """Representation of a simulation instance."""
    id: str
    name: str
    description: Optional[str] = None
    world_time: datetime = Field(default_factory=lambda: datetime(1990, 1, 1, 0, 0, 0))
    parameters: Dict[str, Any] = Field(default_factory=dict)
    world: World = Field(...)
    agents: Dict[str, Agent] = Field(default_factory=dict)

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
                agent_location = row["main_location"]
                free_tiles = world.get_free_tiles_in_main_sector(agent_location)
                if free_tiles:
                    # Pick a random free tile in the specified main location
                    tile = random.choice(free_tiles)
                    agent_location = Point2D(x=tile.location.x, y=tile.location.y)
                    print( f"Spawning agent {agent_name} at ({agent_location.x}, {agent_location.y})" )
                    agent = Agent.create(
                        agent_id=agent_id,
                        name=agent_name,
                        location=agent_location,
                        home_location_name=row["main_location"]
                    )
                    agent.state.background = row["background"]
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
        # Additional logic to update world state and agents can be added here

        # Tick each agent
        for agent_id, agent in self.agents.items():
            # Placeholder for agent tick logic
            agent.tick(self)