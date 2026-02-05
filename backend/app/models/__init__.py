"""Models package initialization.

This module ensures all models are properly initialized and resolves
forward references between models.

BEHAVIOR SYSTEM:
- Agents use Sims-like autonomous behavior (sims_behavior.py)
- Action selection is DETERMINISTIC - NO LLM for decisions
- LLM is only used for conversations and occasional thoughts (conversation.py)
"""

# Import all models to ensure they're loaded
from app.models.life_story import LifeStory  # noqa: F401
from app.models.agent import AgentPersonality  # noqa: F401

# Import Sims behavior system
from app.models.sims_behavior import (  # noqa: F401
    SimsBehaviorManager, 
    SimsAction, 
    NeedType,
    ObjectInteraction,
    OBJECT_INTERACTIONS
)

# Import conversation system (for LLM-based dialogue)
from app.models.conversation import (  # noqa: F401
    ConversationManager,
    InnerLifeManager,
    Conversation,
    ConversationTurn
)

# Rebuild models with forward references after all models are loaded
AgentPersonality.model_rebuild()
