"""
Sims-like deterministic behavior system.

This module handles autonomous agent behavior without LLM involvement.
Agents act like The Sims: needs decay over time, and agents choose 
actions based on urgency scoring and available objects.

LLM is reserved ONLY for:
- Generating conversation content between agents
- Internal thoughts and reflections  
- Memory formation and recall
"""
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field
import random
from loguru import logger


class NeedType(str, Enum):
    """The core needs that drive agent behavior (like The Sims)."""
    ENERGY = "energy"         # Sleep, rest
    HUNGER = "hunger"         # Food, eating
    BLADDER = "bladder"       # Bathroom
    HYGIENE = "hygiene"       # Shower, bath
    SOCIAL = "social"         # Conversation, being with others
    FUN = "fun"               # Entertainment, hobbies
    COMFORT = "comfort"       # Physical comfort
    ENVIRONMENT = "environment"  # Clean surroundings


class ObjectType(str, Enum):
    """Types of interactable objects that satisfy needs."""
    # Bathroom objects
    TOILET = "toilet"
    SHOWER = "shower"
    SINK = "sink"
    BATHTUB = "bathtub"
    
    # Bedroom objects
    BED = "bed"
    DRESSER = "dresser"
    
    # Kitchen objects
    REFRIGERATOR = "refrigerator"
    STOVE = "stove"
    MICROWAVE = "microwave"
    COFFEE_MAKER = "coffee_maker"
    KITCHEN_TABLE = "kitchen_table"
    
    # Living room objects
    SOFA = "sofa"
    TV = "tv"
    BOOKSHELF = "bookshelf"
    COMPUTER = "computer"
    STEREO = "stereo"
    
    # Outdoor/public objects
    PARK_BENCH = "park_bench"
    CAFE_TABLE = "cafe_table"
    
    # Work objects
    DESK = "desk"
    EASEL = "easel"
    PIANO = "piano"


@dataclass
class ObjectInteraction:
    """Defines what happens when an agent uses an object."""
    object_type: ObjectType
    duration_minutes: int
    need_effects: Dict[NeedType, float]  # Positive = satisfy, Negative = drain
    action_name: str
    animation_state: str = "using"
    can_be_interrupted: bool = True
    requires_standing: bool = False


# Define all object interactions and their effects on needs
OBJECT_INTERACTIONS: Dict[str, ObjectInteraction] = {
    # Bathroom
    "toilet": ObjectInteraction(
        object_type=ObjectType.TOILET,
        duration_minutes=5,
        need_effects={NeedType.BLADDER: -80},  # Greatly reduces bladder need
        action_name="Using the toilet",
        animation_state="sitting",
        can_be_interrupted=False
    ),
    "shower": ObjectInteraction(
        object_type=ObjectType.SHOWER,
        duration_minutes=15,
        need_effects={
            NeedType.HYGIENE: 60,
            NeedType.COMFORT: 10,
            NeedType.ENERGY: 10
        },
        action_name="Taking a shower",
        animation_state="showering",
        can_be_interrupted=True
    ),
    "bathtub": ObjectInteraction(
        object_type=ObjectType.BATHTUB,
        duration_minutes=30,
        need_effects={
            NeedType.HYGIENE: 70,
            NeedType.COMFORT: 25,
            NeedType.FUN: 10
        },
        action_name="Taking a bath",
        animation_state="bathing"
    ),
    "sink": ObjectInteraction(
        object_type=ObjectType.SINK,
        duration_minutes=3,
        need_effects={NeedType.HYGIENE: 15},
        action_name="Washing hands",
        animation_state="washing",
        requires_standing=True
    ),
    
    # Bedroom
    "bed": ObjectInteraction(
        object_type=ObjectType.BED,
        duration_minutes=480,  # 8 hours default
        need_effects={
            NeedType.ENERGY: 100,
            NeedType.COMFORT: 50
        },
        action_name="Sleeping",
        animation_state="sleeping",
        can_be_interrupted=True
    ),
    
    # Kitchen
    "refrigerator": ObjectInteraction(
        object_type=ObjectType.REFRIGERATOR,
        duration_minutes=20,
        need_effects={
            NeedType.HUNGER: -50,
            NeedType.ENERGY: 5
        },
        action_name="Getting food from fridge",
        animation_state="eating"
    ),
    "stove": ObjectInteraction(
        object_type=ObjectType.STOVE,
        duration_minutes=30,
        need_effects={
            NeedType.HUNGER: -70,
            NeedType.FUN: 15  # Cooking can be fun
        },
        action_name="Cooking a meal",
        animation_state="cooking"
    ),
    "microwave": ObjectInteraction(
        object_type=ObjectType.MICROWAVE,
        duration_minutes=10,
        need_effects={NeedType.HUNGER: -40},
        action_name="Microwaving food",
        animation_state="waiting"
    ),
    "coffee_maker": ObjectInteraction(
        object_type=ObjectType.COFFEE_MAKER,
        duration_minutes=10,
        need_effects={
            NeedType.ENERGY: 25,
            NeedType.HUNGER: -10
        },
        action_name="Making coffee",
        animation_state="drinking"
    ),
    "kitchen_table": ObjectInteraction(
        object_type=ObjectType.KITCHEN_TABLE,
        duration_minutes=25,
        need_effects={
            NeedType.HUNGER: -60,
            NeedType.COMFORT: 10
        },
        action_name="Eating at the table",
        animation_state="eating"
    ),
    
    # Living room
    "sofa": ObjectInteraction(
        object_type=ObjectType.SOFA,
        duration_minutes=60,
        need_effects={
            NeedType.COMFORT: 40,
            NeedType.ENERGY: 20
        },
        action_name="Relaxing on the sofa",
        animation_state="sitting"
    ),
    "tv": ObjectInteraction(
        object_type=ObjectType.TV,
        duration_minutes=60,
        need_effects={
            NeedType.FUN: 40,
            NeedType.COMFORT: 10,
            NeedType.SOCIAL: 5  # Parasocial
        },
        action_name="Watching TV",
        animation_state="watching"
    ),
    "bookshelf": ObjectInteraction(
        object_type=ObjectType.BOOKSHELF,
        duration_minutes=45,
        need_effects={
            NeedType.FUN: 30,
            NeedType.COMFORT: 5
        },
        action_name="Reading a book",
        animation_state="reading"
    ),
    "computer": ObjectInteraction(
        object_type=ObjectType.COMPUTER,
        duration_minutes=60,
        need_effects={
            NeedType.FUN: 35,
            NeedType.SOCIAL: 15  # Online social
        },
        action_name="Using the computer",
        animation_state="typing"
    ),
    "stereo": ObjectInteraction(
        object_type=ObjectType.STEREO,
        duration_minutes=30,
        need_effects={
            NeedType.FUN: 25,
            NeedType.COMFORT: 10
        },
        action_name="Listening to music",
        animation_state="listening"
    ),
    
    # Public objects
    "park_bench": ObjectInteraction(
        object_type=ObjectType.PARK_BENCH,
        duration_minutes=20,
        need_effects={
            NeedType.COMFORT: 15,
            NeedType.ENVIRONMENT: 20
        },
        action_name="Sitting on a park bench",
        animation_state="sitting"
    ),
    "cafe_table": ObjectInteraction(
        object_type=ObjectType.CAFE_TABLE,
        duration_minutes=30,
        need_effects={
            NeedType.HUNGER: -30,
            NeedType.SOCIAL: 20,
            NeedType.COMFORT: 15
        },
        action_name="Having coffee at the cafe",
        animation_state="drinking"
    ),
    
    # Work objects
    "desk": ObjectInteraction(
        object_type=ObjectType.DESK,
        duration_minutes=120,
        need_effects={
            NeedType.FUN: -10,
            NeedType.ENERGY: -15
        },
        action_name="Working at desk",
        animation_state="working"
    ),
    "easel": ObjectInteraction(
        object_type=ObjectType.EASEL,
        duration_minutes=90,
        need_effects={
            NeedType.FUN: 40,
            NeedType.ENERGY: -10
        },
        action_name="Painting",
        animation_state="painting"
    ),
    "piano": ObjectInteraction(
        object_type=ObjectType.PIANO,
        duration_minutes=45,
        need_effects={
            NeedType.FUN: 50,
            NeedType.SOCIAL: 10  # Others can listen
        },
        action_name="Playing piano",
        animation_state="playing"
    ),
}


# Map keywords in object IDs to interaction definitions
OBJECT_KEYWORD_MAP: Dict[str, str] = {
    "toilet": "toilet",
    "bathroom": "toilet",
    "shower": "shower",
    "bath": "bathtub",
    "bathtub": "bathtub",
    "sink": "sink",
    "bed": "bed",
    "mattress": "bed",
    "dresser": "dresser",
    "refrigerator": "refrigerator",
    "fridge": "refrigerator",
    "stove": "stove",
    "oven": "stove",
    "microwave": "microwave",
    "coffee": "coffee_maker",
    "table": "kitchen_table",
    "sofa": "sofa",
    "couch": "sofa",
    "tv": "tv",
    "television": "tv",
    "book": "bookshelf",
    "bookshelf": "bookshelf",
    "computer": "computer",
    "laptop": "computer",
    "stereo": "stereo",
    "radio": "stereo",
    "speaker": "stereo",
    "bench": "park_bench",
    "cafe": "cafe_table",
    "desk": "desk",
    "easel": "easel",
    "piano": "piano",
    "keyboard": "piano",
}


class SimsAction(BaseModel):
    """A deterministic action chosen by the behavior system."""
    action_type: str  # "use_object", "move_to", "socialize", "sleep", "wait"
    target_object: Optional[str] = None
    target_location: str
    target_sub_location: Optional[str] = None
    target_agent: Optional[str] = None
    duration_minutes: int = 30
    need_being_addressed: Optional[str] = None
    priority: int = 5  # 1-10 scale
    description: str = ""


class SimsBehaviorManager:
    """
    Manages deterministic Sims-like behavior.
    
    This system works purely on need priorities and available objects.
    No LLM is involved in action selection.
    """
    
    # Need urgency thresholds
    CRITICAL_THRESHOLD = 80  # Need value at which it becomes critical
    HIGH_THRESHOLD = 60      # High priority
    MODERATE_THRESHOLD = 40  # Moderate attention needed
    
    # Priority weights for different needs (bladder always highest)
    NEED_PRIORITY_WEIGHTS = {
        NeedType.BLADDER: 1.5,    # Most urgent - can't be ignored
        NeedType.ENERGY: 1.2,     # Very important
        NeedType.HUNGER: 1.1,     # Important
        NeedType.HYGIENE: 0.8,
        NeedType.COMFORT: 0.6,
        NeedType.SOCIAL: 0.7,
        NeedType.FUN: 0.9,        # Fun is important for wellbeing!
        NeedType.ENVIRONMENT: 0.4,
    }
    
    @staticmethod
    def get_interaction_for_object(object_id: str) -> Optional[ObjectInteraction]:
        """Find the interaction definition for an object based on its ID."""
        object_lower = object_id.lower()
        for keyword, interaction_key in OBJECT_KEYWORD_MAP.items():
            if keyword in object_lower:
                return OBJECT_INTERACTIONS.get(interaction_key)
        return None
    
    @staticmethod
    def calculate_need_urgency(agent_needs) -> List[Tuple[NeedType, float]]:
        """
        Calculate urgency scores for all needs.
        
        Returns list of (need_type, urgency_score) sorted by urgency (highest first).
        Urgency is 0-100 where 100 is most urgent.
        """
        urgencies = []
        
        # For needs where LOW is bad (energy, social, fun, hygiene, comfort)
        low_is_bad = {
            NeedType.ENERGY: agent_needs.energy,
            NeedType.SOCIAL: agent_needs.social,
            NeedType.FUN: agent_needs.fun,
            NeedType.HYGIENE: agent_needs.hygiene,
            NeedType.COMFORT: agent_needs.comfort,
        }
        
        # For needs where HIGH is bad (hunger, bladder)
        high_is_bad = {
            NeedType.HUNGER: agent_needs.hunger,
            NeedType.BLADDER: agent_needs.bladder,
        }
        
        for need_type, value in low_is_bad.items():
            # Urgency = 100 - value (so low value = high urgency)
            urgency = (100 - value) * SimsBehaviorManager.NEED_PRIORITY_WEIGHTS.get(need_type, 1.0)
            urgencies.append((need_type, urgency))
        
        for need_type, value in high_is_bad.items():
            # Urgency = value (so high value = high urgency)
            urgency = value * SimsBehaviorManager.NEED_PRIORITY_WEIGHTS.get(need_type, 1.0)
            urgencies.append((need_type, urgency))
        
        # Sort by urgency (highest first)
        urgencies.sort(key=lambda x: x[1], reverse=True)
        return urgencies
    
    @staticmethod
    def get_objects_that_satisfy_need(need_type: NeedType) -> List[str]:
        """Get list of object keywords that can satisfy a given need."""
        satisfying_objects = []
        
        for obj_key, interaction in OBJECT_INTERACTIONS.items():
            for effect_need, effect_value in interaction.need_effects.items():
                # For needs where we want to increase (energy, social, fun, etc.)
                if need_type in [NeedType.ENERGY, NeedType.SOCIAL, NeedType.FUN, 
                                NeedType.HYGIENE, NeedType.COMFORT, NeedType.ENVIRONMENT]:
                    if effect_need == need_type and effect_value > 0:
                        satisfying_objects.append(obj_key)
                # For needs where we want to decrease (hunger, bladder)
                elif need_type in [NeedType.HUNGER, NeedType.BLADDER]:
                    if effect_need == need_type and effect_value < 0:
                        satisfying_objects.append(obj_key)
        
        return satisfying_objects
    
    @staticmethod
    def find_best_object_for_need(
        need_type: NeedType,
        available_objects: List[Dict[str, Any]],
        home_location: str,
        simulation
    ) -> Optional[Dict[str, Any]]:
        """Find the best available object to satisfy a need."""
        satisfying_keywords = SimsBehaviorManager.get_objects_that_satisfy_need(need_type)
        logger.debug(f"🔍 SEARCHING OBJECTS: Need={need_type.value} | Keywords={satisfying_keywords} | Available objects: {len(available_objects)}")
        
        best_object = None
        best_effect = 0
        
        # Check nearby objects first
        for obj in available_objects:
            obj_id = (obj.get("id") or "").lower()
            for keyword in satisfying_keywords:
                if keyword in obj_id:
                    interaction = OBJECT_INTERACTIONS.get(keyword)
                    if interaction:
                        effect = abs(interaction.need_effects.get(need_type, 0))
                        logger.debug(f"   ✓ Match: {obj.get('id')} at {obj.get('location')}/{obj.get('sub_location')} (effect: {effect})")
                        if effect > best_effect:
                            best_effect = effect
                            best_object = obj
                            break
        
        if best_object:
            logger.info(f"✅ FOUND OBJECT: {best_object.get('id')} for {need_type.value} (effect: {best_effect})")
            return best_object
        
        logger.debug(f"   No nearby objects found, searching home...")
        
        # If nothing nearby, check home
        if simulation:
            home_positions = simulation.world.get_map_of_sub_location_to_positions_for_main_location(home_location)
            if home_positions:
                logger.debug(f"   Home has {len(home_positions)} sub-locations: {list(home_positions.keys())}")
                for sub_loc, positions in home_positions.items():
                    sub_loc_lower = sub_loc.lower()
                    for keyword in satisfying_keywords:
                        if keyword in sub_loc_lower:
                            logger.info(f"✅ FOUND AT HOME: {keyword} in {sub_loc} at {home_location}")
                            return {
                                "id": keyword,
                                "sub_location": sub_loc,
                                "location": home_location
                            }
        
        logger.warning(f"❌ NO OBJECT FOUND: Could not find any object for {need_type.value}")
        return None
    
    @staticmethod
    def find_agent_room(agent, simulation) -> Optional[str]:
        """Find the agent's personal room in their home location."""
        if not simulation:
            return None
        
        home_positions = simulation.world.get_map_of_sub_location_to_positions_for_main_location(
            agent.state.home_location_name
        )
        if not home_positions:
            return None
        
        agent_name = agent.state.name
        # Look for a room with the agent's name
        for sub_loc in home_positions.keys():
            sub_loc_lower = sub_loc.lower()
            # Check for patterns like "Francisco Lopez's room" or "bedroom"
            if agent_name.lower() in sub_loc_lower and "room" in sub_loc_lower:
                return sub_loc
        
        # Fallback: look for any room
        for sub_loc in home_positions.keys():
            if "room" in sub_loc.lower() and "bathroom" not in sub_loc.lower():
                return sub_loc
        
        return None
    
    @staticmethod
    def find_agent_bathroom(agent, simulation) -> Optional[str]:
        """Find the agent's personal bathroom in their home location."""
        if not simulation:
            return None
        
        home_positions = simulation.world.get_map_of_sub_location_to_positions_for_main_location(
            agent.state.home_location_name
        )
        if not home_positions:
            return None
        
        agent_name = agent.state.name
        # Look for a bathroom with the agent's name
        for sub_loc in home_positions.keys():
            sub_loc_lower = sub_loc.lower()
            if agent_name.lower() in sub_loc_lower and "bathroom" in sub_loc_lower:
                return sub_loc
        
        # Fallback: look for any bathroom
        for sub_loc in home_positions.keys():
            if "bathroom" in sub_loc.lower():
                return sub_loc
        
        return None
    
    @staticmethod
    def choose_action(
        agent,
        simulation,
        nearby_objects: List[Dict[str, Any]],
        nearby_agents: List[Dict[str, Any]],
        current_location: Optional[str],
        current_hour: int
    ) -> SimsAction:
        """
        Choose the next action based purely on needs (Sims-style).
        
        This is the main decision function - NO LLM involved.
        """
        logger.debug(f"🎮 SIMS BEHAVIOR: {agent.state.name} choosing action | Hour: {current_hour} | Location: {current_location} | Sleeping: {agent.state.is_sleeping}")
        logger.debug(f"   Needs - Energy: {agent.needs.energy:.1f}, Hunger: {agent.needs.hunger:.1f}, Bladder: {agent.needs.bladder:.1f}, Social: {agent.needs.social:.1f}, Fun: {agent.needs.fun:.1f}, Hygiene: {agent.needs.hygiene:.1f}")
        
        # 1. Check for scheduled activities (work, school, etc.)
        scheduled = agent.get_current_scheduled_activity(simulation) if hasattr(agent, 'get_current_scheduled_activity') else None
        if scheduled:
            logger.debug(f"📅 SCHEDULED ACTIVITY: {agent.state.name} has scheduled activity: {scheduled.description}")
            # Only break schedule for truly critical needs
            urgencies = SimsBehaviorManager.calculate_need_urgency(agent.needs)
            top_urgency = urgencies[0] if urgencies else None
            
            # Bladder > 90 or Energy < 10 can interrupt schedule
            if not (top_urgency and 
                    ((top_urgency[0] == NeedType.BLADDER and top_urgency[1] > 90) or
                     (top_urgency[0] == NeedType.ENERGY and agent.needs.energy < 10))):
                logger.info(f"✅ FOLLOWING SCHEDULE: {agent.state.name} continuing with {scheduled.description}")
                return SimsAction(
                    action_type="scheduled",
                    target_location=scheduled.location,
                    target_sub_location=scheduled.sub_location,
                    duration_minutes=scheduled.get_duration_minutes(),
                    description=scheduled.description,
                    priority=scheduled.priority
                )
            else:
                logger.warning(f"⚠️ INTERRUPTING SCHEDULE: {agent.state.name} breaking schedule due to critical need: {top_urgency[0].value} ({top_urgency[1]:.1f})")
        
        # 2. Check if agent should be sleeping
        if hasattr(agent, 'personality') and not agent.personality.should_be_awake(current_hour):
            logger.info(f"🌙 SLEEP ACTION: {agent.state.name} should be sleeping at {current_hour}:00 (wake hour: {agent.personality.preferred_wake_hour}, sleep hour: {agent.personality.preferred_sleep_hour})")
            # Find the agent's personal room
            agent_room = SimsBehaviorManager.find_agent_room(agent, simulation)
            logger.debug(f"   Target: {agent.state.home_location_name}/{agent_room}")
            return SimsAction(
                action_type="sleep",
                target_object="bed",
                target_location=agent.state.home_location_name,
                target_sub_location=agent_room,  # Use agent's personal room, not generic "bedroom"
                duration_minutes=480,
                need_being_addressed="energy",
                priority=10,
                description="Going to sleep"
            )
        
        # 3. Calculate need urgencies
        urgencies = SimsBehaviorManager.calculate_need_urgency(agent.needs)
        logger.debug(f"📊 URGENCIES: {agent.state.name} | " + ", ".join([f"{need.value}: {urgency:.1f}" for need, urgency in urgencies[:5]]))
        
        # 4. Address needs in order of urgency
        for need_type, urgency in urgencies:
            # Skip if not urgent enough (lowered threshold so agents are more proactive)
            if urgency < 20:
                logger.debug(f"   ⏭️ Skipping {need_type.value} (urgency {urgency:.1f} < 20)")
                continue
            
            logger.debug(f"   🔍 Looking for object to satisfy {need_type.value} (urgency: {urgency:.1f})")
            
            # Find object to satisfy this need
            obj = SimsBehaviorManager.find_best_object_for_need(
                need_type,
                nearby_objects,
                agent.state.home_location_name,
                simulation
            )
            
            if obj:
                logger.info(f"✅ OBJECT FOUND: {agent.state.name} found {obj.get('id')} for {need_type.value} at {obj.get('location')}/{obj.get('sub_location')}")
                interaction = SimsBehaviorManager.get_interaction_for_object(obj.get("id", ""))
                duration = interaction.duration_minutes if interaction else 30
                action_name = interaction.action_name if interaction else f"Addressing {need_type.value}"
                
                # Calculate priority based on urgency
                if urgency >= 80:
                    priority = 10
                elif urgency >= 60:
                    priority = 8
                elif urgency >= 40:
                    priority = 6
                else:
                    priority = 4
                
                return SimsAction(
                    action_type="use_object",
                    target_object=obj.get("id"),
                    target_location=obj.get("location", agent.state.home_location_name),
                    target_sub_location=obj.get("sub_location"),
                    duration_minutes=duration,
                    need_being_addressed=need_type.value,
                    priority=priority,
                    description=action_name
                )
            else:
                logger.debug(f"   ❌ No object found for {need_type.value}")
            
            # Special case: social need - look for agents to talk to
            if need_type == NeedType.SOCIAL and nearby_agents:
                logger.info(f"💬 SOCIALIZING: {agent.state.name} choosing to socialize (urgency: {urgency:.1f}) | {len(nearby_agents)} agents nearby")
                # Pick a random nearby agent
                target = random.choice(nearby_agents)
                return SimsAction(
                    action_type="socialize",
                    target_agent=target.get("name"),
                    target_location=current_location or agent.state.home_location_name,
                    duration_minutes=random.randint(10, 30),
                    need_being_addressed="social",
                    priority=6 if urgency >= 50 else 4,
                    description=f"Chatting with {target.get('name')}"
                )
            
            # Special case: fun need - if no object found nearby, try to go home
            if need_type == NeedType.FUN and obj is None:
                logger.warning(f"🎮 NO FUN OBJECT: {agent.state.name} couldn't find fun object nearby, trying home")
                # Try to find fun objects at home
                fun_rooms = ["living room", "bedroom", "common room"]
                home_positions = simulation.world.get_map_of_sub_location_to_positions_for_main_location(
                    agent.state.home_location_name
                ) if simulation else {}
                
                for room_name in home_positions.keys():
                    room_lower = room_name.lower()
                    if any(fun_room in room_lower for fun_room in fun_rooms):
                        logger.info(f"🏠 GOING HOME FOR FUN: {agent.state.name} going to {room_name} for entertainment")
                        return SimsAction(
                            action_type="use_object",
                            target_object="tv",  # Generic fun activity
                            target_location=agent.state.home_location_name,
                            target_sub_location=room_name,
                            duration_minutes=45,
                            need_being_addressed="fun",
                            priority=6 if urgency >= 50 else 4,
                            description="Going home for entertainment"
                        )
        
        # 5. No urgent needs - autonomous behavior
        logger.debug(f"😌 NO URGENT NEEDS: {agent.state.name} choosing autonomous behavior")
        # Look for fun activities or socializing
        roll = random.random()
        
        if roll < 0.3 and nearby_agents and agent.needs.social < 70:
            # Spontaneous socializing
            target = random.choice(nearby_agents)
            return SimsAction(
                action_type="socialize",
                target_agent=target.get("name"),
                target_location=current_location or agent.state.home_location_name,
                duration_minutes=random.randint(5, 20),
                need_being_addressed="social",
                priority=3,
                description=f"Chatting with {target.get('name')}"
            )
        
        # Proactively seek fun even if not critical
        if agent.needs.fun < 70:
            logger.debug(f"🎮 SEEKING FUN: {agent.state.name} fun is {agent.needs.fun:.1f}, looking for entertainment")
            # Look for fun objects nearby
            fun_objects = ["tv", "computer", "bookshelf", "piano", "stereo", "game", "radio"]
            for obj in nearby_objects:
                obj_id = (obj.get("id") or "").lower()
                for fun_kw in fun_objects:
                    if fun_kw in obj_id:
                        interaction = OBJECT_INTERACTIONS.get(fun_kw)
                        logger.info(f"🎯 FUN FOUND: {agent.state.name} found {obj.get('id')} for fun")
                        return SimsAction(
                            action_type="use_object",
                            target_object=obj.get("id"),
                            target_location=obj.get("location", agent.state.home_location_name),
                            target_sub_location=obj.get("sub_location"),
                            duration_minutes=interaction.duration_minutes if interaction else 30,
                            need_being_addressed="fun",
                            priority=3,
                            description=interaction.action_name if interaction else "Having fun"
                        )
            
            # If no fun objects nearby, go to a fun location
            logger.debug(f"🚶 NO FUN NEARBY: {agent.state.name} going to look for entertainment elsewhere")
            fun_locations = ["cafe", "park", "common", "arcade", "library"]
            all_locations = simulation.world.get_all_main_locations() if simulation else []
            for loc in all_locations:
                loc_lower = loc.lower()
                if any(fun_loc in loc_lower for fun_loc in fun_locations):
                    logger.info(f"🎪 GOING TO FUN PLACE: {agent.state.name} going to {loc}")
                    return SimsAction(
                        action_type="move_to",
                        target_location=loc,
                        duration_minutes=15,
                        need_being_addressed="fun",
                        priority=3,
                        description=f"Going to {loc} for fun"
                    )
        
        # 6. Default: idle/wait
        return SimsAction(
            action_type="wait",
            target_location=current_location or agent.state.home_location_name,
            duration_minutes=15,
            priority=1,
            description="Relaxing"
        )
    
    @staticmethod
    def apply_action_effects(agent, action: SimsAction):
        """
        Apply the effects of an action to agent needs.
        Called when an action completes.
        """
        logger.debug(f"⚙️ APPLYING EFFECTS: {agent.state.name} | Action: {action.action_type} | Object: {action.target_object}")
        
        if action.action_type == "use_object" and action.target_object:
            interaction = SimsBehaviorManager.get_interaction_for_object(action.target_object)
            if interaction:
                effects_applied = []
                for need_type, effect in interaction.need_effects.items():
                    old_value = None
                    if need_type == NeedType.ENERGY:
                        old_value = agent.needs.energy
                        agent.needs.energy = max(0, min(100, agent.needs.energy + effect))
                        effects_applied.append(f"Energy: {old_value:.1f} → {agent.needs.energy:.1f} ({effect:+.1f})")
                    elif need_type == NeedType.HUNGER:
                        old_value = agent.needs.hunger
                        agent.needs.hunger = max(0, min(100, agent.needs.hunger + effect))
                        effects_applied.append(f"Hunger: {old_value:.1f} → {agent.needs.hunger:.1f} ({effect:+.1f})")
                    elif need_type == NeedType.BLADDER:
                        old_value = agent.needs.bladder
                        agent.needs.bladder = max(0, min(100, agent.needs.bladder + effect))
                        effects_applied.append(f"Bladder: {old_value:.1f} → {agent.needs.bladder:.1f} ({effect:+.1f})")
                    elif need_type == NeedType.HYGIENE:
                        old_value = agent.needs.hygiene
                        agent.needs.hygiene = max(0, min(100, agent.needs.hygiene + effect))
                        effects_applied.append(f"Hygiene: {old_value:.1f} → {agent.needs.hygiene:.1f} ({effect:+.1f})")
                    elif need_type == NeedType.SOCIAL:
                        old_value = agent.needs.social
                        agent.needs.social = max(0, min(100, agent.needs.social + effect))
                        effects_applied.append(f"Social: {old_value:.1f} → {agent.needs.social:.1f} ({effect:+.1f})")
                    elif need_type == NeedType.FUN:
                        old_value = agent.needs.fun
                        agent.needs.fun = max(0, min(100, agent.needs.fun + effect))
                        effects_applied.append(f"Fun: {old_value:.1f} → {agent.needs.fun:.1f} ({effect:+.1f})")
                    elif need_type == NeedType.COMFORT:
                        old_value = agent.needs.comfort
                        agent.needs.comfort = max(0, min(100, agent.needs.comfort + effect))
                        effects_applied.append(f"Comfort: {old_value:.1f} → {agent.needs.comfort:.1f} ({effect:+.1f})")
                
                if effects_applied:
                    logger.info(f"✨ EFFECTS APPLIED: {agent.state.name} | " + ", ".join(effects_applied))
        
        elif action.action_type == "socialize":
            # Socializing always boosts social need
            old_social = agent.needs.social
            old_fun = agent.needs.fun
            agent.needs.social = min(100, agent.needs.social + 25)
            agent.needs.fun = min(100, agent.needs.fun + 10)
            logger.info(f"✨ SOCIALIZING EFFECTS: {agent.state.name} | Social: {old_social:.1f} → {agent.needs.social:.1f}, Fun: {old_fun:.1f} → {agent.needs.fun:.1f}")
        
        elif action.action_type == "sleep":
            old_energy = agent.needs.energy
            old_comfort = agent.needs.comfort
            agent.needs.energy = 100
            agent.needs.comfort = min(100, agent.needs.comfort + 30)
            logger.info(f"✨ SLEEP EFFECTS: {agent.state.name} | Energy: {old_energy:.1f} → 100.0, Comfort: {old_comfort:.1f} → {agent.needs.comfort:.1f}")
        
        elif action.action_type == "wait":
            # Small passive recovery
            agent.needs.energy = min(100, agent.needs.energy + 2)
            agent.needs.comfort = min(100, agent.needs.comfort + 3)
            logger.debug(f"✨ WAIT EFFECTS: {agent.state.name} | Small passive recovery")
