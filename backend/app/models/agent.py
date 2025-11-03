from enum import Enum
from app.llm.client import OllamaClient, StructuredPromptBuilder
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pydantic import validator
import random
from app.models.models import Point2D, Plan


class ActionType(str, Enum):
    """Types of actions agents can perform."""
    MOVE = "move"
    INTERACT = "interact"
    CONVERSE = "converse"
    USE_OBJECT = "use_object"
    WAIT = "wait"
    REFLECT = "reflect"

class AgentAction(BaseModel):
    """Structured agent action."""
    agent_id: str
    action_type: ActionType
    description: str
    target_agent: str = None  # Target agent ID
    target_object: str = None  # Name of the target object, ex: "refrigerator", "bed", "kitchen sink"
    location: str # Location name
    sub_location: Optional[str] = None  # Sub-location name

    class Config:
        # Serialize Enums as their values and allow extra fields when constructing from external data
        use_enum_values = True
        extra = "allow"
    
    @validator('description')
    def description_format(cls, v):
        # Ensure description follows pattern: "Agent is [action]"
        if not v.strip():
            raise ValueError("Action description cannot be empty")
        return v
    
class AgentDailySchedule(BaseModel):
    hour_action: Dict[int, AgentAction] = Field(default_factory=dict)  # Hour (0-23) to action mapping

class AgentMemory(BaseModel):
    visited_locations: set[Point2D] = Field(default_factory=set)
    current_path: Optional[list[Point2D]] = None  # Immutable sequence of points
    known_object_positions: Dict[Tuple[str, str, str], List[Point2D]] = Field(default_factory=dict)  # (Sector Name, Sub Location Name, Object ID) to position mapping

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

class AgentNeeds(BaseModel):
    energy: float = Field(default=100.0, ge=0.0, le=100.0)
    hunger: float = Field(default=0.0, ge=0.0, le=100.0)
    bladder: float = Field(default=0.0, ge=0.0, le=100.0)
    social: float = Field(default=100.0, ge=0.0, le=100.0)
    fun: float = Field(default=100.0, ge=0.0, le=100.0)
    hygiene: float = Field(default=100.0, ge=0.0, le=100.0)

    # set contants for decay rates
    ENERGY_DECAY_RATE: float = 0.1  # per tick
    HUNGER_INCREASE_RATE: float = 0.2  # per tick
    BLADDER_INCREASE_RATE: float = 0.15  # per tick
    SOCIAL_DECAY_RATE: float = 0.05  # per tick
    FUN_DECAY_RATE: float = 0.07  # per tick
    HYGIENE_DECAY_RATE: float = 0.03  # per tick

    def tick(self):
        """Update needs based on decay rates."""
        self.energy = max(0.0, self.energy - self.ENERGY_DECAY_RATE)
        self.hunger = min(100.0, self.hunger + self.HUNGER_INCREASE_RATE)
        self.bladder = min(100.0, self.bladder + self.BLADDER_INCREASE_RATE)
        self.social = max(0.0, self.social - self.SOCIAL_DECAY_RATE)
        self.fun = max(0.0, self.fun - self.FUN_DECAY_RATE)
        self.hygiene = max(0.0, self.hygiene - self.HYGIENE_DECAY_RATE)

class AgentState(BaseModel):
    """Current state of an agent."""
    agent_id: str
    name: str
    background: Optional[str] = None
    location: Point2D
    home_location_name: str
    current_action: Optional[AgentAction] = None
    current_plan: Optional[Plan] = None
    status: str = Field(default="idle")
    last_reflection_time: Optional[datetime] = Field(default_factory=datetime.utcnow)
    relationship_scores: Dict[str, float] = Field(default_factory=dict)
    
    def update_relationship(self, other_agent_id: str, delta: float):
        """Update relationship score with another agent."""
        current = self.relationship_scores.get(other_agent_id, 0.0)
        self.relationship_scores[other_agent_id] = max(-1.0, min(1.0, current + delta))


class Agent(BaseModel):
    """Representation of an agent in the world."""
    state: AgentState
    memory: AgentMemory = Field(default_factory=AgentMemory)
    needs: AgentNeeds = Field(default_factory=AgentNeeds)
    ollama_client: Any = Field(default_factory=OllamaClient)
    is_setup: bool = Field(default=False)
    daily_schedule: AgentDailySchedule = Field(default_factory=AgentDailySchedule)

    def _find_nearest_food(self, simulation) -> Optional[Point2D]:
        """Find the nearest food object in the world."""
        food_sources = self.memory.recall_object(self.get_current_location_name(simulation), None, "refrigerator")
        if not food_sources:
            return None
        # Find the closest food source
        return min(food_sources, key=lambda food: food.distance_to(self.state.location))

    def pick_action(self, simulation) -> Optional[AgentAction]:
        """Decide on the next action based on current state and needs."""
        # Placeholder logic: if hunger > 80, look for food
        if self.needs.hunger > 80.0:
            food_location = self._find_nearest_food(simulation)
            if food_location:
                return AgentAction(
                    agent_id=self.state.agent_id,
                    action_type=ActionType.MOVE,
                    description=f"{self.state.name} is moving towards food.",
                    location=f"{food_location.x},{food_location.y}",
                    duration_seconds=5,
                    success_probability=0.8
                )
            # Find nearest food object in the world (not implemented)
            # For now, just return a wait action
            return AgentAction(
                agent_id=self.state.agent_id,
                action_type=ActionType.WAIT,
                description=f"{self.state.name} is waiting to find food.",
                location=f"{self.state.location.x},{self.state.location.y}",
                duration_seconds=10,
                success_probability=1.0
            )
        # Default action: move randomly
        return None  # Let tick handle movement for now
    
    def prime_memories(self, simulation):
        """Prime the agent's memory with relevant information."""
        # We should take our home location and remember objects there
        home_location_name = self.state.home_location_name
        world = simulation.world
        objects = world.get_objects_in_sector(home_location_name)

        for sub_location_name, objs in objects.items():
            for obj in objs:
                obj_id = obj.get("id")
                obj_position = Point2D(x=obj.get("location").get("x"), y=obj.get("location").get("y"))
                if obj_id:
                    self.memory.remember_object(home_location_name, sub_location_name, obj_id, obj_position)

    def get_current_location_name(self, simulation) -> Optional[str]:
        """Get the name of the current location based on agent's coordinates."""
        world = simulation.world
        tile_id = f"{int(self.state.location.x)},{int(self.state.location.y)}"
        if tile_id in world.map.sector_mappings:
            sector_info = world.map.sector_mappings[tile_id]
            return sector_info.get("main_location_name")
        return None
    
    def get_current_sub_location_name(self, simulation) -> Optional[str]:
        """Get the name of the current sub-location based on agent's coordinates."""
        world = simulation.world
        
        return world.get_sub_location_name_for_main_location_and_position(
            main_location_name=self.get_current_location_name(simulation),
            position=self.state.location
        )

    def setup(self, simulation):
        import json
        """Initial setup for the agent within the simulation."""

        # We need to make our daily schedule 
        # Get a range of hours in the day (0-23)
        hours = list(range(5))
        agent_background = f"I am agent {self.state.name}, with ID: {self.state.agent_id}. Background: {self.state.background or "An ordinary person."}"
        system_prompt = """You are a scheduling assistant that generates realistic hourly actions for simulation agents based on their background and current context.

        ## Time Format
        - Hour 0 = 12:00 AM (midnight)
        - Hour 23 = 11:00 PM
        - Generate contextually appropriate actions for each hour of the day

        ## Action Requirements
        1. Each action must specify an action_type (e.g., 'wait', 'move', 'interact')
        2. Include a descriptive explanation of what the agent is doing
        3. If using an object/furniture, specify it in the target_object field
        4. Specify the location and sub_location where the action occurs
        5. Consider typical human patterns (sleep at night, meals at regular times, work/leisure balance)

        ## Response Format
        Always respond using the provided JSON schema following the AgentAction structure.

        ## Available Objects
        Bedroom: bed, desk, closet, shelf, easel
        Bathroom: bathroom sink, shower, toilet
        Kitchen: kitchen sink, refrigerator, toaster, cooking area
        Common Areas: common room table, common room sofa, guitar, microphone, piano, game console, computer desk, computer
        Bar/Cafe: bar customer seating, behind the bar counter, behind the cafe counter, cafe customer seating
        Library: library sofa, bookshelf, library table
        Classroom: classroom student seating, classroom podium, blackboard
        Stores: behind the pharmacy counter, behind the grocery counter, pharmacy store shelf, grocery store shelf, pharmacy store counter, grocery store counter, supply store product shelf, behind the supply store counter, supply store counter
        Outdoor: dorm garden, house garden, garden chair, park garden
        Recreation: harp, lifting weight, pool table"""

        # Add dynamic location information
        locations = simulation.world.get_all_main_locations()
        system_prompt += f"\n\n## Available Main Locations\n{', '.join(locations)}"

        # Add sub-location mapping
        system_prompt += "\n\n## Location Hierarchy\n"
        sub_locations = simulation.world.map.arena_main_locations_to_sub_locations
        sub_locations = {k: list(v) for k, v in sub_locations.items()}
        system_prompt += json.dumps(sub_locations, indent=2)

        # Add more detailed examples
        system_prompt += """

        ## Schedule Examples

        ### Early Morning (0-5)
        Hour 0: {"agent_id": "2", "action_type": "wait", "description": "Sleeping peacefully in bed.", "target_agent": null, "target_object": "bed", "location": "artist's co-living space", "sub_location": "Latoya Williams's room"}

        ### Morning (6-11)
        Hour 7: {"agent_id": "2", "action_type": "wait", "description": "Taking a morning shower.", "target_agent": null, "target_object": "shower", "location": "artist's co-living space", "sub_location": "bathroom"}
        Hour 8: {"agent_id": "2", "action_type": "wait", "description": "Preparing and eating breakfast.", "target_agent": null, "target_object": "refrigerator", "location": "artist's co-living space", "sub_location": "kitchen"}

        ### Afternoon (12-17)
        Hour 14: {"agent_id": "2", "action_type": "wait", "description": "Working at desk on creative projects.", "target_agent": null, "target_object": "desk", "location": "artist's co-living space", "sub_location": "Latoya Williams's room"}

        ### Evening (18-23)
        Hour 19: {"agent_id": "2", "action_type": "wait", "description": "Socializing in the common room.", "target_agent": null, "target_object": "common room sofa", "location": "artist's co-living space", "sub_location": "common room"}

        ## Guidelines
        - Ensure activities align with the agent's background and personality
        - Create realistic daily patterns (sleep, meals, work, leisure)
        - Vary activities to make the schedule interesting but believable
        - Consider social interactions during appropriate hours
        - Match activities to appropriate locations and objects"""

        # if there is a saved daily schedule, load it
        schedule_file = f"./agent_{self.state.agent_id}_daily_schedule.json"
        try:
            with open(schedule_file, "r") as f:
                saved_schedule = json.load(f)
                saved_schedule = saved_schedule["hour_action"]
                for hour_str, action_data in saved_schedule.items():
                    hour = int(hour_str)
                    action = AgentAction(**action_data)
                    self.daily_schedule.hour_action[hour] = action
            print( f"Loaded saved daily schedule for agent {self.state.agent_id} from {schedule_file}" )
            self.is_setup = True
            
        except FileNotFoundError:
            print( f"No saved daily schedule found for agent {self.state.agent_id}, generating new schedule." )

            # For each hour, generate an action
            for hour in hours:
                output = self.ollama_client.generate_structured(
                        system=system_prompt,
                        prompt=StructuredPromptBuilder.build_planning_prompt(
                            agent_name=self.state.name,
                            agent_description=agent_background,
                            current_time=hour,
                            context=f"The agent lives in {self.state.home_location_name}."
                        ),
                        response_model=AgentAction,
                        temperature=0.1
                )

                print( output )

                self.daily_schedule.hour_action[hour] = output
            
            #save the daily schedule to a json file for debugging
            import json
            with open(f"./agent_{self.state.agent_id}_daily_schedule.json", "w") as f:
                json.dump(self.daily_schedule.dict(), f, indent=2)

        self.prime_memories(simulation)

        # Get our hour 0 action and set it as our current action
        if 0 in self.daily_schedule.hour_action:
            self.state.current_action = self.daily_schedule.hour_action[0]
            # Now move to that location, take the location and look up coordinates
            location = self.state.current_action.location
            # if there is a sublocation, look up its coordinates as well
            sublocation = self.state.current_action.sub_location
            world = simulation.world
            print( f"Agent {self.state.agent_id} initial action set to hour 0 action: {self.state.current_action.action_type} at location {location}, sublocation {sublocation}" )
            positions = []
            if sublocation:
                positions = world.get_map_of_sub_location_to_positions_for_main_location(location).get(sublocation, [])
            else:
                positions = world.get_sub_locations_positions_for_main_location(location)
            if positions:
                chosen_position = random.choice(positions)
                print( f"Agent {self.state.agent_id} initial position set to ({chosen_position.x}, {chosen_position.y}) in location {location}, sublocation {sublocation}" )
                self.state.location = Point2D(x=chosen_position.x, y=chosen_position.y)

        self.is_setup = True
        


    def tick(self, simulation):
        """Advance the agent's state within the simulation."""
        # Check if our memory is empty and needs priming
        print( self.state.background )

        if not self.is_setup:
            self.setup(simulation)

        # print( f"Agent {self.state.agent_id} at location ({self.state.location.x}, {self.state.location.y})" )
        # print( self.get_current_location_name(simulation) )
        # print( self.get_current_sub_location_name(simulation) )
        # print( self._find_nearest_food(simulation) )

        self.state.last_reflection_time = datetime.utcnow()
        # Additional logic for the agent's actions can be added here

        # Put our current location in memory
        self.memory.visited_locations.add(self.state.location)

        # Tick needs
        self.needs.tick()

        # Use our needs to decide on an action



        # Find a place to move (randomly for now)
        simulation_world = simulation.world

        # Follow the current path if we have one
        if self.memory.current_path:
            # Follow the current path
            next_point = self.memory.current_path.pop(0)
            self.state.location = Point2D(x=next_point.x, y=next_point.y)
            if not self.memory.current_path:
                self.memory.current_path = None  # Clear path when done
            self.state.status = "moved"
            return self.state
            
        # No current path, check if we need to pick a new action from our daily schedule
        current_hour = simulation.world_time.hour
        if current_hour in self.daily_schedule.hour_action:
            scheduled_action = self.daily_schedule.hour_action[current_hour]
            self.state.current_action = scheduled_action
            if scheduled_action.action_type == ActionType.MOVE:
                # Parse target location
                loc_parts = scheduled_action.location.split(",")
                if len(loc_parts) == 2:
                    target_x = float(loc_parts[0])
                    target_y = float(loc_parts[1])
                    path = simulation_world.find_path(self.state.location, Point2D(x=target_x, y=target_y))
                    if path:
                        self.memory.current_path = path
                        # Move along the path (for simplicity, just move to the next point)
                        next_point = path[0]
                        self.state.location = Point2D(x=next_point.x, y=next_point.y)
                        self.state.status = "moved"
                        print( f"Agent {self.state.agent_id} moved to ({self.state.location.x}, {self.state.location.y}) as per schedule." )
                        return self.state


        # free_tiles = simulation_world.get_free_tiles()

        # if not free_tiles:
        #     print( f"Agent {self.state.agent_id} found no free tiles to move to." )
        #     return self.state  # No free tiles to move to
        
        # new_tile = random.choice(free_tiles)
        # new_x, new_y = new_tile.location.x, new_tile.location.y

        # # Find the path to the new location        
        # path = simulation_world.find_path(self.state.location, Point2D(x=new_x, y=new_y))

        # if path:
        #     self.memory.current_path = path
        #     # Move along the path (for simplicity, just move to the next point)
        #     next_point = path[0]
        #     new_x, new_y = next_point.x, next_point.y

        #     self.state.location = Point2D(x=new_x, y=new_y)

        # # Check if the new location is valid within the world map
        # world_map = simulation.world.map
        # #print( world_map )
        # if world_map:
        #     tile_id = f"{int(new_x)},{int(new_y)}"
        #     if tile_id in world_map.tiles:
        #         tile = world_map.tiles[tile_id]
        #         from app.models.world import WorldMapTileType

        #         if tile.type != WorldMapTileType.OBSTACLE:
        #             self.state.location = Point2D(x=new_x, y=new_y)
        
        # Update status
        print( f"Agent {self.state.agent_id} waited at ({self.state.location.x}, {self.state.location.y})" )
        #self.state.status = "moved"

        return self.state

    @staticmethod
    def create(agent_id: str, name: str, location: Point2D, home_location_name: str) -> 'Agent':
        """Factory method to create a new Agent."""
        agent_state = AgentState(
            agent_id=agent_id,
            name=name,
            location=location,
            status="idle",
            home_location_name=home_location_name
        )
        return Agent(state=agent_state)
