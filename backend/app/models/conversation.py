"""
LLM-powered conversation and inner life system.

This module handles the ONLY parts of agent behavior that use the LLM:
1. Generating conversation dialogue between agents
2. Creating internal thoughts and reflections
3. Memory formation and emotional processing

All action decisions are handled by the SimsBehaviorManager - NO LLM for that.
"""
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pydantic import BaseModel, Field
import random

from app.models.agent import MemoryType, EmotionType


class ConversationTurn(BaseModel):
    """A single turn in a conversation."""
    speaker: str
    dialogue: str
    inner_thought: Optional[str] = None
    emotion: Optional[str] = None


class Conversation(BaseModel):
    """A conversation between two or more agents."""
    participants: List[str]
    turns: List[ConversationTurn] = Field(default_factory=list)
    topic: Optional[str] = None
    location: str
    start_time: datetime
    is_active: bool = True
    

class ConversationManager:
    """
    Manages LLM-powered conversations between agents.
    
    This is one of the ONLY places the LLM is used.
    """
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.active_conversations: Dict[str, Conversation] = {}
    
    def start_conversation(
        self,
        initiator_agent,
        target_agent,
        location: str,
        current_time: datetime
    ) -> Conversation:
        """Start a new conversation between two agents."""
        conversation_id = f"{initiator_agent.state.agent_id}_{target_agent.state.agent_id}_{current_time.isoformat()}"
        
        conversation = Conversation(
            participants=[initiator_agent.state.name, target_agent.state.name],
            location=location,
            start_time=current_time
        )
        
        self.active_conversations[conversation_id] = conversation
        return conversation
    
    def generate_conversation_dialogue(
        self,
        speaker_agent,
        listener_agent,
        conversation: Conversation,
        context: Optional[str] = None
    ) -> ConversationTurn:
        """
        Generate a single dialogue turn using the LLM.
        
        This is where the LLM is used for rich, contextual conversation.
        """
        # Build context about the speaker
        speaker_context = self._build_agent_context(speaker_agent)
        listener_context = self._build_agent_context(listener_agent, brief=True)
        
        # Get relationship history
        relationship = speaker_agent.state.get_relationship_description(listener_agent.state.name)
        history = speaker_agent.memory.get_relationship_history(listener_agent.state.name)
        recent_history = history[-3:] if history else []
        
        # Get conversation history
        recent_turns = conversation.turns[-4:] if conversation.turns else []
        conversation_history = "\n".join([
            f"{turn.speaker}: \"{turn.dialogue}\""
            for turn in recent_turns
        ])
        
        system_prompt = f"""You are roleplaying as {speaker_agent.state.name} in a conversation.

CHARACTER PROFILE:
{speaker_context}

TALKING TO: {listener_agent.state.name}
- Relationship: {relationship}
- Recent interactions: {'; '.join(recent_history) if recent_history else 'None'}
{listener_context}

CONVERSATION SO FAR:
{conversation_history if conversation_history else "(Just starting)"}

GUIDELINES:
- Speak naturally as {speaker_agent.state.name} would
- Keep responses concise (1-3 sentences max)
- Reflect the character's personality and current mood
- Reference shared history if relevant
- Be authentic to the relationship dynamics

Generate what {speaker_agent.state.name} says next.
Also include their inner thought (what they're really thinking but not saying).
"""

        prompt = f"What does {speaker_agent.state.name} say to {listener_agent.state.name}?"
        
        try:
            # Use structured output for dialogue
            response = self.llm_client.generate_structured(
                system=system_prompt,
                prompt=prompt,
                response_model=ConversationTurn,
                temperature=0.8,
                max_retries=2
            )
            response.speaker = speaker_agent.state.name
            return response
        except Exception as e:
            # Fallback to simple dialogue
            greetings = [
                "Hey, how's it going?",
                "Nice to see you!",
                "What's up?",
                "How are you doing?",
            ]
            return ConversationTurn(
                speaker=speaker_agent.state.name,
                dialogue=random.choice(greetings),
                inner_thought="Just making small talk..."
            )
    
    def _build_agent_context(self, agent, brief: bool = False) -> str:
        """Build context string about an agent for LLM prompts."""
        if brief:
            return f"""- Personality: {agent.personality.get_description()}
- Current mood: {agent.emotions.get_mood_description()}"""
        
        needs_desc = agent.needs.to_description()
        
        return f"""- Name: {agent.state.name}
- Background: {agent.state.background or 'Unknown'}
- Personality: {agent.personality.get_description()}
- Current mood: {agent.emotions.get_mood_description()}
- Physical state: {needs_desc if needs_desc else 'feeling fine'}
- Currently: {agent.state.current_activity_description or 'idle'}"""
    
    def end_conversation(
        self, 
        conversation: Conversation,
        agents: List[Any]
    ) -> Dict[str, str]:
        """
        End a conversation and generate memories for participants.
        
        Returns dict of agent_name -> memory summary
        """
        if len(conversation.turns) == 0:
            return {}
        
        memories = {}
        summary = self._summarize_conversation(conversation)
        
        for agent in agents:
            if agent.state.name in conversation.participants:
                # Form a memory of the conversation
                other_participants = [p for p in conversation.participants if p != agent.state.name]
                memory_content = f"Had a conversation with {', '.join(other_participants)}: {summary}"
                
                agent.form_memory(
                    content=memory_content,
                    memory_type=MemoryType.SOCIAL,
                    importance=5.0,
                    emotional_valence=0.3,  # Conversations are generally positive
                    related_agents=other_participants,
                    location=conversation.location
                )
                
                # Update relationship memory
                for other in other_participants:
                    agent.memory.add_relationship_memory(
                        other,
                        f"Talked at {conversation.start_time.strftime('%I:%M %p')}: {summary[:50]}..."
                    )
                
                memories[agent.state.name] = memory_content
        
        conversation.is_active = False
        return memories
    
    def _summarize_conversation(self, conversation: Conversation) -> str:
        """Create a brief summary of the conversation."""
        if not conversation.turns:
            return "Brief exchange"
        
        if len(conversation.turns) <= 2:
            return "Quick chat"
        
        # Simple keyword extraction for topics
        all_dialogue = " ".join([t.dialogue.lower() for t in conversation.turns])
        
        topic_keywords = {
            "work": ["work", "job", "office", "boss", "meeting"],
            "weather": ["weather", "rain", "sun", "cold", "hot"],
            "plans": ["plan", "going to", "tomorrow", "later", "weekend"],
            "feelings": ["feel", "happy", "sad", "tired", "excited"],
            "food": ["eat", "hungry", "lunch", "dinner", "food"],
        }
        
        detected_topics = []
        for topic, keywords in topic_keywords.items():
            if any(kw in all_dialogue for kw in keywords):
                detected_topics.append(topic)
        
        if detected_topics:
            return f"Discussed {', '.join(detected_topics[:2])}"
        
        return "General conversation"


class InnerLifeManager:
    """
    Manages LLM-powered inner thoughts and reflections.
    
    This handles the rich internal life of agents using LLM.
    """
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    def generate_inner_thought(
        self,
        agent,
        situation: str = "",
        current_time: Optional[datetime] = None
    ) -> str:
        """
        Generate an inner thought for the agent based on their situation.
        
        Used for rich internal monologue during activities.
        """
        # Most of the time, use simple deterministic thoughts
        # Only use LLM occasionally for richer thoughts
        if random.random() > 0.3:  # 70% deterministic
            return self._generate_simple_thought(agent)
        
        # Use LLM for richer thought
        system_prompt = f"""You are the inner voice of {agent.state.name}.

CHARACTER:
- Personality: {agent.personality.get_description()}
- Current mood: {agent.emotions.get_mood_description()}
- Background: {agent.state.background or 'Unknown'}

CURRENT SITUATION: {situation or agent.state.current_activity_description or 'idle'}

Generate a single brief inner thought (10-20 words max) that {agent.state.name} might be thinking right now.
It should feel authentic to their personality and current state.
Just output the thought itself, no quotes or attribution."""

        try:
            response = self.llm_client.generate(
                system=system_prompt,
                prompt="What is on their mind right now?",
                temperature=0.9,
                max_tokens=50
            )
            return response.strip().strip('"\'')[:100]  # Truncate to keep brief
        except Exception:
            return self._generate_simple_thought(agent)
    
    def _generate_simple_thought(self, agent) -> str:
        """Generate a simple deterministic thought based on state."""
        # Thoughts based on needs
        if agent.needs.hunger > 70:
            return random.choice([
                "I'm getting hungry...",
                "Should probably eat soon.",
                "My stomach is growling.",
            ])
        
        if agent.needs.energy < 30:
            return random.choice([
                "I'm so tired...",
                "Could really use some rest.",
                "Getting sleepy.",
            ])
        
        if agent.needs.bladder > 70:
            return "Need to find a bathroom..."
        
        if agent.needs.social < 30:
            return random.choice([
                "Would be nice to talk to someone.",
                "Feeling a bit lonely.",
                "Haven't seen anyone in a while.",
            ])
        
        if agent.needs.fun < 30:
            return random.choice([
                "This is getting boring.",
                "Need to do something fun.",
                "I'm so bored...",
            ])
        
        # Thoughts based on emotions
        if agent.emotions.emotion_intensity > 0.6:
            emotion_thoughts = {
                EmotionType.HAPPY: ["Today is going well!", "Feeling good."],
                EmotionType.SAD: ["Feeling a bit down.", "Not the best day."],
                EmotionType.ANXIOUS: ["Something feels off.", "A bit worried."],
                EmotionType.BORED: ["So bored.", "Need something to do."],
                EmotionType.EXCITED: ["This is exciting!", "Can't wait!"],
                EmotionType.STRESSED: ["So much going on.", "Need a break."],
            }
            thoughts = emotion_thoughts.get(agent.emotions.primary_emotion, ["Just thinking..."])
            return random.choice(thoughts)
        
        # Default thoughts
        return random.choice([
            "Just going about my day.",
            "...",
            "Hmm.",
            "Things are okay.",
        ])
    
    def generate_reflection(
        self,
        agent,
        recent_events: List[str],
        current_time: datetime
    ) -> Optional[str]:
        """
        Generate a deeper reflection on recent events.
        
        Used periodically for memory consolidation and emotional processing.
        """
        if not recent_events:
            return None
        
        # Get recent memories for context
        recent_memories = agent.memory.recall_memories(current_time, limit=5)
        memory_context = "\n".join([f"- {m.content}" for m in recent_memories])
        
        system_prompt = f"""You are helping {agent.state.name} reflect on their day.

CHARACTER:
- Personality: {agent.personality.get_description()}
- Background: {agent.state.background or 'Unknown'}
- Current mood: {agent.emotions.get_mood_description()}

RECENT EVENTS:
{chr(10).join(['- ' + e for e in recent_events[-5:]])}

MEMORIES:
{memory_context if memory_context else 'No significant memories'}

Generate a brief reflection (1-2 sentences) that {agent.state.name} might have about their recent experiences.
Focus on emotional insight or personal meaning.
Just output the reflection itself."""

        try:
            response = self.llm_client.generate(
                system=system_prompt,
                prompt="What insight does this person have about their recent experiences?",
                temperature=0.8,
                max_tokens=100
            )
            return response.strip().strip('"\'')[:200]
        except Exception:
            return None
    
    def process_emotional_event(
        self,
        agent,
        event: str,
        event_type: str = "neutral"
    ) -> Tuple[EmotionType, float, str]:
        """
        Process an event emotionally and return the emotional response.
        
        Returns: (emotion_type, intensity, inner_reaction)
        """
        # Simple deterministic emotional processing based on event type
        event_emotions = {
            "positive": (EmotionType.HAPPY, random.uniform(0.4, 0.7), "That's nice."),
            "social": (EmotionType.CONTENT, random.uniform(0.3, 0.5), "Good to connect."),
            "achievement": (EmotionType.EXCITED, random.uniform(0.5, 0.8), "I did it!"),
            "negative": (EmotionType.SAD, random.uniform(0.3, 0.6), "That's unfortunate."),
            "stressful": (EmotionType.STRESSED, random.uniform(0.4, 0.7), "This is overwhelming."),
            "boring": (EmotionType.BORED, random.uniform(0.3, 0.5), "Nothing interesting."),
            "neutral": (EmotionType.NEUTRAL, random.uniform(0.1, 0.3), "Okay then."),
        }
        
        emotion, intensity, reaction = event_emotions.get(
            event_type, 
            (EmotionType.NEUTRAL, 0.2, "...")
        )
        
        # Adjust based on personality
        if agent.personality.neuroticism > 0.7:
            intensity *= 1.3  # More emotionally reactive
        elif agent.personality.neuroticism < 0.3:
            intensity *= 0.7  # More emotionally stable
        
        intensity = min(1.0, intensity)
        
        return emotion, intensity, reaction
