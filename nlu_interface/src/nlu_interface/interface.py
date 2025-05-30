from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Tuple
import logging
logger = logging.getLogger(__name__)

from nlu_interface.prompt import Prompt, DefaultPrompt
from nlu_interface.config import LLMConfig, OpenAIConfig, OllamaConfig, AnthropicBedrockConfig

import ollama
import anthropic
from anthropic.types import Message
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

    def _query(self) -> Tuple[str, APIResponse]:
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
            temperature=self.temperature,
        )
        self.response_history.append(response)
        output_text = response.output[0].content[0].text
        return output_text, response

    def _create_client(self):
        self.client = openai.OpenAI(
            api_key = self.api_key,
            timeout = self.api_timeout,
        )
        if self.debug:
            logger.debug(f"Created an OpenAI client for model {self.model}.")

class OllamaWrapper(LLMInterface[OllamaConfig, Prompt]):
    
    def __init__(
        self,
        config: OllamaConfig,
        prompt: P,
    ):
        super().__init__(config, prompt)
        self.ollama_url = config.ollama_url
        
        self._create_client()
        self.valid_model_names = self._get_model_names()
        print(f"Valid model names: {self.valid_model_names}")
        self.valid_prompt_modes = ("default")

        # Validate the configuration
        config.validate(
            valid_model_names=self.valid_model_names,
            valid_prompt_modes=self.valid_prompt_modes
        )

        # Create the Ollama client
        self._create_client()

    def _create_client(self):
        self.client = ollama.Client(
            host=self.ollama_url,
        )
        if self.debug:
            logger.debug(f"Created an Ollama client for url {self.ollama_url}.")
            
    def _get_model_names(self) -> List[str]:
        model_list = self.client.list()
        if model_list['models']:
            return [model['model'] for model in model_list['models']]
        else:
            raise ValueError("No models found in Ollama client. Please check your Ollama server connection.")
        
    def _query(self) -> APIResponse:
        if not self.prompt:
            raise ValueError("No prompt available in _query().")
        prompt = self.prompt.to_ollama(self.num_incontext_examples)
        if self.debug:
            logger.debug(
                f"Querying Ollama API ({self.model}).\nCurrent Prompt: {self.prompt.render(self.num_incontext_examples)}"
            )

        response = self.client.generate(
            model=self.model,
            prompt=prompt
        )
        self.response_history.append(response)
        return response
        
class AnthropicBedrockWrapper(LLMInterface[AnthropicBedrockConfig, Prompt]):
    valid_model_names = (
        "anthropic.claude-3-5-haiku-20241022-v1:0",
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "anthropic.claude-sonnet-4-20250514-v1:0",
        "anthropic.claude-opus-4-20250514-v1:0",
    )
    valid_prompt_modes = ("default", "chain-of-thought")

    def __init__(
        self,
        config: AnthropicBedrockConfig,
        prompt: P,
    ):
        super().__init__(config, prompt)
        self.aws_region = config.aws_region
        self.access_key = config.resolve_access_key()
        self.secret_key = config.resolve_secret_key()

        # Create the AnthropicBedrock client
        self._create_client()

    def _query(self) -> Tuple[str, Message]:
        if not self.prompt:
            raise ValueError("No prompt available in _query().")
        system, user = self.prompt.to_anthropic(self.num_incontext_examples)
        if self.debug:
            logger.debug(
                f"Querying AWS Bedrock API ({self.model}).\nCurrent Prompt: {self.prompt.render(self.num_incontext_examples)}"
            )

        response = self.client.messages.create(
            model=self.model,
            system=system,
            messages=user,
            temperature=self.temperature,
            max_tokens=4096,
        )
        self.response_history.append(response)
        output_text = response.content[0].text
        return output_text, response

    def _create_client(self):
        self.client = anthropic.AnthropicBedrock(
            aws_access_key=self.access_key,
            aws_secret_key=self.secret_key,
            aws_region=self.aws_region,
        )
        if self.debug:
            logger.debug(f"Created an AnthropicBedrock client for model {self.model}.")
