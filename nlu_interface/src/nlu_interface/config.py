import os
import logging
from dataclasses import dataclass
from typing import Tuple

logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    model: str
    prompt_mode: str
    prompt_type: str
    num_incontext_examples: int
    temperature: float = 0.0
    debug: bool = False

    def validate(self, valid_model_names: Tuple, valid_prompt_modes: Tuple):
        if self.model not in valid_model_names:
            raise ValueError(f"Invalid model name: {self.model}")
        if self.prompt_mode not in valid_prompt_modes:
            raise ValueError(f"Invalid prompt mode: {self.prompt_mode}")

@dataclass
class OpenAIConfig(LLMConfig):
    api_key_env_var: str = ""
    api_timeout: float = 120.0
    seed: int = 100

    def resolve_api_key(self) -> str:
        if self.api_key_env_var:
            api_key = os.getenv(self.api_key_env_var)
        else:
            self.api_key_env_var = "OPENAI_API_KEY"
            logger.warning(f"No API key env var provided; defaulting to \"{self.api_key_env_var}\".")
            api_key = os.getenv(self.api_key_env_var)
        if not api_key:
            raise ValueError(f"API Key not found for \"{self.api_key_env_var}\"")
        return api_key
    
@dataclass
class OllamaConfig(LLMConfig):
    ollama_url: str = "http://localhost:11434"


