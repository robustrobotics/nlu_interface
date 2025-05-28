from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Tuple
import logging
logger = logging.getLogger(__name__)

from nlu_interface.prompt import Prompt, DefaultPrompt
from nlu_interface.config import LLMConfig, OpenAIConfig

import openai
from openai import APIResponse

P = TypeVar("P", bound=Prompt) # Prompt Type
C = TypeVar("C", bound=LLMConfig) # Config Type
R = TypeVar("R") # Response Type

class LLMInterface(ABC, Generic[P, C]):
    valid_model_names: Tuple = ()
    valid_prompt_names: Tuple = ()

    def __init__(self, config: C, prompt: P):
        self.config = config
        self.prompt = prompt
        self.model = config.model
        self.prompt_mode = config.prompt_mode
        self.prompt_type = config.prompt_type
        self.num_incontext_examples = (
            len(self.prompt.incontext_examples)
            if config.num_incontext_examples == -1
            else config.num_incontext_examples
        )
        self.temperature = config.temperature
        self.debug = config.debug
        self.response_history: List = []

    @abstractmethod
    def _query(self) -> R:
        raise NotImplementedError('Subclasses must implement "_query()".')

    @abstractmethod
    def _create_client(self):
        raise NotImplementedError('Subclasses must implement "_create_client()".')

class OpenAIWrapper(LLMInterface[OpenAIConfig, Prompt]):
    valid_model_names = ("gpt-4o", "gpt-4o-mini")
    valid_prompt_modes = ("default", "chain-of-thought")

    def __init__(
        self,
        config: OpenAIConfig,
        prompt: P,
    ):
        super().__init__(config, prompt)
        self.api_key = config.resolve_api_key()
        self.api_timeout = config.api_timeout
        self.seed = config.seed

        # Validate the configuration
        config.validate(
            valid_model_names=self.valid_model_names,
            valid_prompt_modes=self.valid_prompt_modes
        )

        # Create the OpenAI client
        self._create_client()

    def _query(self) -> APIResponse:
        if not self.prompt:
            raise ValueError("No prompt available in _query().")
        instructions, user = self.prompt.to_openai(self.num_incontext_examples)
        if self.debug:
            logger.debug(
                f"Querying OpenAI Cloud API ({self.model}).\nCurrent Prompt: {self.prompt.render(self.num_incontext_examples)}"
            )

        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=user,
        )
        self.response_history.append(response)
        return response

    def _create_client(self):
        self.client = openai.OpenAI(
            api_key = self.api_key,
            timeout = self.api_timeout,
        )
        if self.debug:
            logger.debug(f"Created an OpenAI client for model {self.model}.")
