"""Generator for creating agent life stories using LLM."""
import random
from typing import Optional, List
from app.llm.client import OllamaClient
from app.models.life_story import (
    LifeStory, LifeEvent, LifeStoryPrompt, GeneratedLifeStoryResponse,
    EventCategory, PersonalityImpact
)
from loguru import logger


class LifeStoryGenerator:
    """Generates detailed life stories for agents using LLM."""
    
    def __init__(self, llm_client: Optional[OllamaClient] = None):
        self.llm_client = llm_client or OllamaClient()
    
    def generate_life_story(
        self,
        agent_name: str,
        current_age: int,
        occupation: Optional[str] = None,
        cultural_background: Optional[str] = None,
        personality_seed: Optional[dict] = None
    ) -> LifeStory:
        """
        Generate a complete life story with 2-3 shaping events per age period.
        
        Args:
            agent_name: Name of the agent
            current_age: Current age of the agent
            occupation: Optional occupation to influence story
            cultural_background: Optional cultural context
            personality_seed: Optional initial personality traits to guide story
            
        Returns:
            Complete LifeStory with generated events
        """
        logger.info(f"Generating life story for {agent_name}, age {current_age}")
        
        # Calculate age periods (every 2 years starting at age 4)
        age_periods = self._calculate_age_periods(current_age)
        
        # Build the prompt for LLM
        prompt = self._build_generation_prompt(
            agent_name, current_age, age_periods, occupation, 
            cultural_background, personality_seed
        )
        
        system_prompt = """You are a creative writer specializing in character development and life histories. 
Generate realistic, coherent life stories with events that naturally shape personality development. 
Events should be diverse, emotionally impactful, and interconnected. Consider how early events influence later ones.
Make the story feel authentic with a mix of positive and negative experiences."""
        
        try:
            # Generate using structured output
            response = self.llm_client.generate_structured(
                prompt=prompt,
                response_model=GeneratedLifeStoryResponse,
                system=system_prompt,
                temperature=0.8  # Higher temperature for more creative stories
            )
            
            # Build complete life story
            life_story = LifeStory(
                agent_name=agent_name,
                current_age=current_age,
                birthplace=response.birthplace,
                events=response.events,
                core_values=response.core_values,
                fears=response.fears,
                aspirations=response.aspirations,
                important_people=response.important_people
            )
            
            logger.info(f"Successfully generated {len(life_story.events)} life events for {agent_name}")
            return life_story
            
        except Exception as e:
            logger.error(f"Failed to generate life story for {agent_name}: {e}")
            # Return a minimal life story as fallback
            return self._generate_fallback_story(agent_name, current_age)
    
    def _calculate_age_periods(self, current_age: int) -> List[tuple[int, int]]:
        """Calculate age periods (every 2 years from age 4)."""
        periods = []
        start_age = 4
        
        while start_age < current_age:
            end_age = min(start_age + 1, current_age)  # 2-year periods
            periods.append((start_age, end_age))
            start_age += 2
        
        return periods
    
    def _build_generation_prompt(
        self,
        agent_name: str,
        current_age: int,
        age_periods: List[tuple[int, int]],
        occupation: Optional[str],
        cultural_background: Optional[str],
        personality_seed: Optional[dict]
    ) -> str:
        """Build the LLM prompt for life story generation."""
        
        prompt_parts = [
            f"Generate a detailed life story for {agent_name}, who is currently {current_age} years old.",
            ""
        ]
        
        if occupation:
            prompt_parts.append(f"Current occupation: {occupation}")
        
        if cultural_background:
            prompt_parts.append(f"Cultural background: {cultural_background}")
        
        if personality_seed:
            traits = []
            if personality_seed.get('extraversion', 0.5) > 0.7:
                traits.append("naturally outgoing")
            elif personality_seed.get('extraversion', 0.5) < 0.3:
                traits.append("naturally introverted")
            if personality_seed.get('conscientiousness', 0.5) > 0.7:
                traits.append("organized")
            if personality_seed.get('openness', 0.5) > 0.7:
                traits.append("creative and curious")
            if traits:
                prompt_parts.append(f"Personality tendencies: {', '.join(traits)}")
        
        prompt_parts.extend([
            "",
            "Generate 2-3 significant life events for each of these age periods:",
        ])
        
        for start, end in age_periods:
            if end == start:
                prompt_parts.append(f"  - Age {start}")
            else:
                prompt_parts.append(f"  - Ages {start}-{end}")
        
        prompt_parts.extend([
            "",
            "Requirements:",
            "- Each event should be specific and meaningful",
            "- Events should connect and influence each other naturally",
            "- Include a mix of positive and negative experiences",
            "- Events should shape personality traits (use personality_impact field)",
            "- Consider family, education, relationships, achievements, and challenges",
            "- Make emotional_impact realistic (-1.0 to 1.0, where -1 is very negative, 1 is very positive)",
            "",
            "For personality_impact, consider how each event would realistically change traits:",
            "- Positive social events: increase extraversion, agreeableness",
            "- Achievements: increase conscientiousness, reduce neuroticism",
            "- Traumas/losses: increase neuroticism, may decrease extraversion",
            "- Creative experiences: increase openness",
            "- Failures: may decrease conscientiousness, increase neuroticism",
            "- Adventures: increase openness, spontaneity",
            "",
            "Also include:",
            "- Birthplace (city or region)",
            "- 2-5 core values developed through life",
            "- 1-4 deep fears or anxieties",
            "- 2-4 aspirations or life goals",
            "- Important people in their life (name -> relationship, e.g., 'Sarah' -> 'childhood best friend')",
        ])
        
        return "\n".join(prompt_parts)
    
    def _generate_fallback_story(self, agent_name: str, current_age: int) -> LifeStory:
        """Generate a basic fallback story if LLM fails."""
        logger.warning(f"Using fallback story generation for {agent_name}")
        
        events = []
        
        # Add a few basic events
        if current_age >= 6:
            events.append(LifeEvent(
                age=6,
                category=EventCategory.EDUCATION,
                description=f"Started elementary school and made first friends",
                emotional_impact=0.4,
                personality_impact=PersonalityImpact(extraversion_delta=0.1)
            ))
        
        if current_age >= 14:
            events.append(LifeEvent(
                age=14,
                category=EventCategory.SOCIAL,
                description=f"Joined a club or sports team in middle school",
                emotional_impact=0.3,
                personality_impact=PersonalityImpact(conscientiousness_delta=0.1)
            ))
        
        if current_age >= 18:
            events.append(LifeEvent(
                age=18,
                category=EventCategory.ACHIEVEMENT,
                description=f"Graduated high school and started planning for the future",
                emotional_impact=0.5,
                personality_impact=PersonalityImpact(openness_delta=0.15)
            ))
        
        if current_age >= 22:
            events.append(LifeEvent(
                age=22,
                category=EventCategory.CAREER,
                description=f"Started first job and learned professional skills",
                emotional_impact=0.3,
                personality_impact=PersonalityImpact(work_ethic_delta=0.15)
            ))
        
        return LifeStory(
            agent_name=agent_name,
            current_age=current_age,
            birthplace="Unknown Town",
            events=events,
            core_values=["growth", "independence"],
            fears=["failure"],
            aspirations=["success", "happiness"]
        )
    
    def apply_life_story_to_personality(
        self,
        base_personality: dict,
        life_story: LifeStory
    ) -> dict:
        """
        Apply cumulative personality impacts from life story to base personality.
        
        Args:
            base_personality: Dict with base personality traits
            life_story: Complete life story with events
            
        Returns:
            Updated personality dict with traits shaped by life events
        """
        # Start with base personality (or defaults)
        personality = {
            'openness': base_personality.get('openness', 0.5),
            'conscientiousness': base_personality.get('conscientiousness', 0.5),
            'extraversion': base_personality.get('extraversion', 0.5),
            'agreeableness': base_personality.get('agreeableness', 0.5),
            'neuroticism': base_personality.get('neuroticism', 0.5),
            'work_ethic': base_personality.get('work_ethic', 0.5),
            'spontaneity': base_personality.get('spontaneity', 0.5),
        }
        
        # Apply cumulative impact from all events
        total_impact = life_story.get_total_personality_impact()
        
        personality['openness'] = self._clamp(
            personality['openness'] + total_impact.openness_delta, 0.0, 1.0
        )
        personality['conscientiousness'] = self._clamp(
            personality['conscientiousness'] + total_impact.conscientiousness_delta, 0.0, 1.0
        )
        personality['extraversion'] = self._clamp(
            personality['extraversion'] + total_impact.extraversion_delta, 0.0, 1.0
        )
        personality['agreeableness'] = self._clamp(
            personality['agreeableness'] + total_impact.agreeableness_delta, 0.0, 1.0
        )
        personality['neuroticism'] = self._clamp(
            personality['neuroticism'] + total_impact.neuroticism_delta, 0.0, 1.0
        )
        personality['work_ethic'] = self._clamp(
            personality['work_ethic'] + total_impact.work_ethic_delta, 0.0, 1.0
        )
        personality['spontaneity'] = self._clamp(
            personality['spontaneity'] + total_impact.spontaneity_delta, 0.0, 1.0
        )
        
        logger.info(f"Applied life story impacts to {life_story.agent_name}'s personality")
        return personality
    
    @staticmethod
    def _clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamp value between min and max."""
        return max(min_val, min(max_val, value))
