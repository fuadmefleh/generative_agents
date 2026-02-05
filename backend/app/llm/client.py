"""Ollama client for LLM operations with structured output support."""
import json
from typing import Type, TypeVar, Optional, Dict, Any
import httpx
from pydantic import BaseModel, ValidationError
from loguru import logger
from ollama import chat

T = TypeVar('T', bound=BaseModel)


class OllamaClient:
    """Client for interacting with Ollama API with structured output support."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "gen-agent-model"):
        """Initialize Ollama client.
        
        Args:
            base_url: Base URL for Ollama API
            model: Model name to use
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        #self.client = httpx.Client(timeout=120.0)

    def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.7) -> str:
        """Generate text completion.
        
        Args:
            prompt: User prompt
            system: Optional system prompt
            
        Returns:
            Generated text
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "temperature": temperature,
                    "extra_body": {"reasoning_effort": "low"}
                }
            )
            response.raise_for_status()
            result = response.json()
            return result["message"]["content"]
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise
    
    def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_retries: int = 3
    ) -> T:
        schema = response_model.model_json_schema()
        
        # Simplified prompt - don't include schema since format param handles it
        structured_prompt = prompt
        structured_system = system or "You are a helpful assistant that always responds with valid JSON."
        
        for attempt in range(max_retries):
            try:
                response = chat(
                    messages=[
                        {"role": "system", "content": structured_system},
                        {"role": "user", "content": structured_prompt}
                    ],
                    model=self.model,
                    format=schema,  # This enforces the schema
                    options={
                        "temperature": temperature,
                        "num_predict": 2048  # Increased token limit for complete JSON responses
                    }
                )
                
                # Extract content string from response
                content = response['message']['content']

                # Log empty responses
                if not content or content.strip() == '':
                    logger.warning(f"Empty response from LLM on attempt {attempt + 1}")
                    if attempt == max_retries - 1:
                        raise ValueError("LLM returned empty response after all retries")
                    continue

                logger.debug(f"Generated content on attempt {attempt + 1} ({len(content)} chars)")
                
                # Try to clean up common JSON issues
                # Remove any text before the first { or [
                content = content.strip()
                if '{' in content:
                    content = content[content.find('{'):]
                if content.endswith('}'):
                    # Find the last valid closing brace
                    brace_count = 0
                    for i, char in enumerate(content):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                content = content[:i+1]
                                break
                
                # Parse the JSON string to a Python dict
                data = json.loads(content)
                
                # Create and return the Pydantic model instance
                return response_model(**data)
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error on attempt {attempt + 1}: {e}")
                logger.debug(f"Failed to parse: {content}")
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed to generate valid JSON after {max_retries} attempts")
                    
            except ValidationError as e:
                logger.warning(f"Validation error on attempt {attempt + 1}: {e}")
                logger.debug(f"Data that failed validation: {data}")
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed to generate valid response after {max_retries} attempts: {e}")
    
    def embed(self, text: str) -> list[float]:
        """Generate embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            response = self.client.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                }
            )
            response.raise_for_status()
            result = response.json()
            return result["embedding"]
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return a placeholder embedding if error
            return [0.0] * 768  # Common embedding dimension
    
    def generate_batch(
        self,
        prompts: list[str],
        system: Optional[str] = None
    ) -> list[str]:
        """Generate responses for multiple prompts.
        
        Args:
            prompts: List of prompts
            system: Optional system prompt
            
        Returns:
            List of generated responses
        """
        import asyncio
        tasks = [self.generate(prompt, system) for prompt in prompts]
        return asyncio.gather(*tasks)
    
    def list_models(self) -> list[str]:
        """List available models.
        
        Returns:
            List of model names
        """
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            result = response.json()
            return [model["name"] for model in result.get("models", [])]
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def check_model_availability(self) -> bool:
        """Check if the configured model is available.
        
        Returns:
            True if model is available
        """
        models = self.list_models()
        return self.model in models
    
    def close(self):
        """Close the HTTP client."""
        self.client.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        import asyncio
        asyncio.create_task(self.close())


class StructuredPromptBuilder:
    """Helper class for building structured prompts."""
    
    @staticmethod
    def build_importance_prompt(observation: str, agent_name: str) -> str:
        """Build prompt for scoring importance."""
        return f"""Rate the importance of this observation for {agent_name}.
        
Observation: {observation}

Scale:
1-2: Mundane daily activities (brushing teeth, making bed)
3-4: Routine tasks (eating meals, commuting)
5-6: Notable events (meeting someone new, completing a task)
7-8: Significant events (important conversations, achievements)
9-10: Life-changing events (breakup, job offer, major discovery)"""
    
    @staticmethod
    def build_reflection_prompt(memories: list[str], agent_name: str) -> str:
        """Build prompt for generating reflections."""
        memories_text = "\n".join([f"{i+1}. {m}" for i, m in enumerate(memories)])
        
        return f"""Based on these recent experiences of {agent_name}, what are 3 high-level insights?

Recent experiences:
{memories_text}

Generate 3 insights that synthesize patterns or important realizations from these experiences."""
    
    @staticmethod
    def build_planning_prompt(
        agent_name: str,
        agent_description: str,
        current_time: str,
        context: str
    ) -> str:
    
        """Build prompt for generating plans."""
        # Convert the 0-23 current time to AM/PM format
        import datetime
        # Accept int or str for current_time; normalize to a zero-padded hour string
        try:
            current_time_str = f"{int(current_time):02d}"
        except (TypeError, ValueError):
            current_time_str = str(current_time)
        current_time_am_pm = datetime.datetime.strptime(current_time_str, "%H").strftime("%I %p")
        return f"""Create the action for the CURRENT TIME we are generating the action for: {current_time_am_pm} for {agent_name}.

        
        
Character: {agent_description}
Current Time we are generating the action for: {    current_time_am_pm  }
Context: {context}

The location must be from the list of known locations for the agent. The sublocation is optional but if supplied must also be from the known sublocations for the agent.

An target object or target agent must be included if the action involves interaction with something or someone.

Generate a realistic action with specific activity that fit this character's personality and situation."""

    @staticmethod
    def build_dialogue_prompt(
        speaker_name: str,
        listener_name: str,
        context: str,
        last_message: str
    ) -> str:
        """Build prompt for generating dialogue."""
        return f"""{speaker_name} is in conversation with {listener_name}.

Context: {context}
{listener_name} just said: "{last_message}"

How would {speaker_name} naturally respond? Keep it brief and conversational."""
