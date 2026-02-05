"""Factory for creating agents with emergent personalities and life stories."""
import random
from typing import Optional, List
from app.models.agent import Agent, AgentPersonality, AgentState, AgentJob, JobType
from app.models.models import Point2D
from app.models.life_story import LifeStory
from app.factory.life_story_generator import LifeStoryGenerator
from app.factory.routine_generator import RoutineGenerator
from app.llm.client import OllamaClient
from loguru import logger


class AgentFactory:
    """Factory for creating agents with generated life stories and emergent personalities."""
    
    def __init__(self, llm_client: Optional[OllamaClient] = None):
        self.llm_client = llm_client or OllamaClient()
        self.life_story_generator = LifeStoryGenerator(self.llm_client)
    
    def create_agent_with_life_story(
        self,
        agent_id: str,
        name: str,
        location: Point2D,
        home_location_name: str,
        age: int = None,
        occupation: Optional[JobType] = None,
        cultural_background: Optional[str] = None,
        generate_life_story: bool = True
    ) -> Agent:
        """
        Create an agent with an emergent personality shaped by a generated life story.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Agent's name
            location: Starting location
            home_location_name: Name of agent's home location
            age: Agent's age (random between 22-65 if not provided)
            occupation: Optional job type
            cultural_background: Optional cultural context for life story
            generate_life_story: Whether to generate life story (set False for quick testing)
            
        Returns:
            Fully initialized Agent with personality shaped by life events
        """
        # Determine age
        if age is None:
            age = random.randint(22, 65)
        
        logger.info(f"Creating agent {name} (age {age}) with life story generation")
        
        # Start with a base personality (will be modified by life story)
        base_personality = {
            'openness': random.uniform(0.4, 0.6),
            'conscientiousness': random.uniform(0.4, 0.6),
            'extraversion': random.uniform(0.3, 0.7),
            'agreeableness': random.uniform(0.4, 0.7),
            'neuroticism': random.uniform(0.3, 0.6),
            'work_ethic': random.uniform(0.4, 0.7),
            'spontaneity': random.uniform(0.3, 0.6),
        }
        
        life_story = None
        if generate_life_story:
            try:
                # Generate life story
                life_story = self.life_story_generator.generate_life_story(
                    agent_name=name,
                    current_age=age,
                    occupation=occupation.value if occupation else None,
                    cultural_background=cultural_background,
                    personality_seed=base_personality
                )
                
                # Apply life story impacts to personality
                final_personality = self.life_story_generator.apply_life_story_to_personality(
                    base_personality, life_story
                )
                
                logger.info(f"Life story shaped {name}'s personality with {len(life_story.events)} events")
            except Exception as e:
                logger.error(f"Failed to generate life story for {name}: {e}")
                final_personality = base_personality
        else:
            final_personality = base_personality
            logger.info(f"Skipping life story generation for {name}")
        
        # Create personality object
        personality = AgentPersonality(
            openness=final_personality['openness'],
            conscientiousness=final_personality['conscientiousness'],
            extraversion=final_personality['extraversion'],
            agreeableness=final_personality['agreeableness'],
            neuroticism=final_personality['neuroticism'],
            work_ethic=final_personality['work_ethic'],
            spontaneity=final_personality['spontaneity'],
            morning_person=random.choice([True, False]),
            preferred_wake_hour=random.randint(5, 8) if final_personality['conscientiousness'] > 0.6 else random.randint(7, 10),
            preferred_sleep_hour=random.randint(21, 23) if final_personality['conscientiousness'] > 0.6 else random.randint(22, 24) % 24,
            favorite_activities=self._generate_favorite_activities(final_personality, life_story),
            conversation_style=self._determine_conversation_style(final_personality),
            life_story=life_story
        )
        
        # Create job if occupation provided
        job = None
        if occupation:
            job = self._create_job(occupation, personality)
        
        # Generate daily routine based on personality and life story
        daily_schedule = RoutineGenerator.generate_daily_routine(
            personality=personality,
            job=job,
            life_story=life_story,
            home_location=home_location_name
        )
        
        # Create agent state
        state = AgentState(
            agent_id=agent_id,
            name=name,
            location=location,
            home_location_name=home_location_name,
            background=life_story.get_narrative_summary() if life_story else None
        )
        
        # Create complete agent
        agent = Agent(
            state=state,
            personality=personality,
            job=job,
            daily_schedule=daily_schedule,
            ollama_client=self.llm_client
        )
        
        logger.info(f"Created agent {name} with {len(daily_schedule)} scheduled activities")
        return agent
    
    def create_multiple_agents(
        self,
        count: int,
        location_generator: callable,
        home_locations: List[str],
        occupations: Optional[List[JobType]] = None,
        cultural_backgrounds: Optional[List[str]] = None
    ) -> List[Agent]:
        """
        Create multiple agents with diverse backgrounds.
        
        Args:
            count: Number of agents to create
            location_generator: Function that returns a Point2D for agent starting location
            home_locations: List of possible home location names
            occupations: Optional list of job types to assign
            cultural_backgrounds: Optional list of cultural backgrounds
            
        Returns:
            List of created agents
        """
        agents = []
        names = self._generate_diverse_names(count)
        
        for i in range(count):
            name = names[i]
            location = location_generator()
            home = random.choice(home_locations)
            age = random.randint(22, 65)
            
            occupation = None
            if occupations:
                occupation = random.choice(occupations)
            
            cultural_bg = None
            if cultural_backgrounds:
                cultural_bg = random.choice(cultural_backgrounds)
            
            try:
                agent = self.create_agent_with_life_story(
                    agent_id=f"agent_{i}",
                    name=name,
                    location=location,
                    home_location_name=home,
                    age=age,
                    occupation=occupation,
                    cultural_background=cultural_bg
                )
                agents.append(agent)
                logger.info(f"Successfully created agent {i+1}/{count}: {name}")
            except Exception as e:
                logger.error(f"Failed to create agent {name}: {e}")
        
        return agents
    
    @staticmethod
    def _generate_favorite_activities(personality_dict: dict, life_story: Optional[LifeStory]) -> List[str]:
        """Generate favorite activities based on personality and life story."""
        activities = []
        
        # Personality-based activities
        if personality_dict['openness'] > 0.6:
            activities.extend(["reading", "art", "exploring"])
        
        if personality_dict['extraversion'] > 0.6:
            activities.extend(["socializing", "parties", "group activities"])
        else:
            activities.extend(["reading", "quiet time", "solitude"])
        
        if personality_dict['conscientiousness'] > 0.6:
            activities.extend(["organizing", "planning", "projects"])
        
        # Life story based activities
        if life_story:
            for event in life_story.events:
                if "music" in event.description.lower():
                    activities.append("music")
                if "sport" in event.description.lower() or "athletic" in event.description.lower():
                    activities.append("exercise")
                if "art" in event.description.lower() or "creative" in event.description.lower():
                    activities.append("art")
        
        # Remove duplicates and limit
        activities = list(set(activities))
        return activities[:4] if len(activities) > 4 else activities
    
    @staticmethod
    def _determine_conversation_style(personality_dict: dict) -> str:
        """Determine conversation style from personality."""
        if personality_dict['extraversion'] > 0.7:
            return "talkative"
        elif personality_dict['extraversion'] < 0.3:
            return "reserved"
        elif personality_dict['agreeableness'] > 0.7:
            return "friendly"
        else:
            return "brief"
    
    @staticmethod
    def _create_job(job_type: JobType, personality: AgentPersonality) -> AgentJob:
        """Create a job object with appropriate workplace."""
        workplace_map = {
            JobType.BARISTA: ("The Ville Cafe", "counter"),
            JobType.ARTIST: ("Art Studio", "studio"),
            JobType.STUDENT: ("University", "classroom"),
            JobType.TEACHER: ("School", "classroom"),
            JobType.CHEF: ("Restaurant", "kitchen"),
            JobType.ORGANIZER: ("Community Center", "office"),
            JobType.WRITER: ("Library", "study room"),
            JobType.MUSICIAN: ("Music Hall", "stage"),
        }
        
        workplace, sub_location = workplace_map.get(job_type, ("Office", "desk"))
        
        # Job satisfaction influenced by personality fit
        satisfaction = 0.7
        if job_type == JobType.ARTIST and personality.openness > 0.7:
            satisfaction = 0.9
        elif job_type == JobType.TEACHER and personality.agreeableness > 0.7:
            satisfaction = 0.85
        elif job_type == JobType.BARISTA and personality.extraversion > 0.7:
            satisfaction = 0.85
        
        return AgentJob(
            job_type=job_type,
            workplace_location=workplace,
            workplace_sub_location=sub_location,
            job_satisfaction=satisfaction,
            skills=[]
        )
    
    @staticmethod
    def _generate_diverse_names(count: int) -> List[str]:
        """Generate diverse names for agents."""
        first_names = [
            "Emma", "Liam", "Olivia", "Noah", "Ava", "Ethan", "Sophia", "Mason",
            "Isabella", "William", "Mia", "James", "Charlotte", "Benjamin", "Amelia",
            "Lucas", "Harper", "Henry", "Evelyn", "Alexander", "Abigail", "Michael",
            "Emily", "Daniel", "Ella", "Matthew", "Scarlett", "Jackson", "Grace",
            "Sebastian", "Chloe", "Jack", "Victoria", "Aiden", "Riley", "Owen",
            "Aria", "Samuel", "Lily", "Joseph", "Aubrey", "John", "Zoey", "David",
            "Penelope", "Wyatt", "Lillian", "Carter", "Addison", "Jayden"
        ]
        
        last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
            "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
            "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
            "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
            "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
            "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
            "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
            "Carter", "Roberts"
        ]
        
        names = []
        used_names = set()
        
        while len(names) < count:
            first = random.choice(first_names)
            last = random.choice(last_names)
            full_name = f"{first} {last}"
            
            if full_name not in used_names:
                names.append(full_name)
                used_names.add(full_name)
        
        return names
