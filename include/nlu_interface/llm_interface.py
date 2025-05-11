# LLM Imports
import tiktoken
import requests

""" Exceptions
"""
class ConstructionFailure(Exception):
    pass

class PromptingFailure(Exception):
    pass

class ParsingFailure(Exception):
    pass

""" Globals
"""
from types import MappingProxyType
models_map = MappingProxyType(
    {
        "gpt-4o" : "gpt-4o-2024-08-06",
        "gpt-4o-mini" : "gpt-4o-mini-2024-07-18",
    }
)

prompt_modes_set = (
    "naive",
    "chain-of-thought",
)

""" Classes
"""
class Prompt:
    def __init__(
        self,
        system=None,
        incontext_examples_preamble=None,
        incontext_examples=None,
        new_instruction_preamble=None,
        new_instruction=None,
        new_scene_graph=None,
        response_format=None,
    ):
        self.system = system
        self.incontext_examples_preamble = incontext_examples_preamble
        self.incontext_examples = incontext_examples
        self.new_instruction_preamble = new_instruction_preamble
        self.new_instruction = new_instruction
        self.new_scene_graph = new_scene_graph
        self.response_format = response_format

    def __repr__(self):
        return repr( self.to_dict() )

    @classmethod
    def from_dict(cls, d, new_instruction=None, new_scene_graph=None):
        system = d["system"]
        incontext_examples_preamble = d["incontext_examples_preamble"]
        incontext_examples = [ IncontextExample.from_dict(e) for e in d["incontext_examples"] ]
        new_instruction_preamble = d["new_instruction_preamble"]
        response_format = d["response_format"]
        
        # Only load the "new instruction" from the dictionary if no function arg provided
        if not new_instruction and "new_instruction" in d
            new_instruction = d["new_instruction"]
        if not new_scene_graph and "new_scene_graph" in d
            new_scene_graph = d["new_scene_graph"]

        return cls(
            system,
            incontext_examples_preamble,
            incontext_examples,
            new_instruction_preamble,
            new_instruction,
            new_scene_graph,
            response_format,
        )

    def to_dict(self):
        d = {}
        if self.system:
            d["system"] = self.system
        if self.incontext_examples_preamble:
            d["incontext_examples_preamble"] = self.incontext_examples_preamble
        if self.incontext_examples:
            d["incontext_examples"] = self.incontext_examples
        if self.new_instruction_preamble:
            d["new_instruction_preamble"] = self.new_instruction_preamble
        if self.new_instruction:
            d["new_instruction"] = self.new_instruction
        if self.new_scene_graph:
            d["new_scene_graph"] = self.new_scene_graph
        if self.response_format:
            d["response_format"] = self.response_format
        return d

    def to_openai_messages(self, num_incontext_examples):
        if num_incontext_examples > len(self.incontext_examples):
            raise PromptingFailure(f"More requested incontext examples ({num_incontext_examples}) than available ({len(self.incontext_examples)}).")
        messages = []
        # Add the system message
        system_message_content = self.system.format(response_format=self.response_format)
        messages.append({"role" : "system", "content" : system_message_content})
        # Compose & Add the user message
        user_message_content = self.incontext_examples_preamble
        for i in range(num_incontext_examples):
            user_message_content += "\n" + self.incontext_example[i].to_prompt()
        user_message_content += self.new_instruction_preamble.format(response_format=self.response_format)
        user_message_content += self.new_scene_graph
        user_message_content += self.new_instruction
        messages.append({"role" : "user", "content" : user_message_content})
        return messages

class IncontextExample:
    def __init__(self, example_input, example_output):
        self.example_input = example_input
        self.example_output = example_output
        return

    def __repr__(self):
        return repr( self.to_dict() )

    @classmethod
    def from_dict(cls, d):
        example_input = d["example_input"]
        example_output = d["example_output"]
        return cls(example_input, example_output)

    def to_dict(self):
        d = {}
        if self.example_input:
            d["example_input"] = self.example_input
        if self.example_output:
            d["example_output"] = self.example_output
        return d

    def to_prompt(self):
        return self.example_input + "\n" + self.example_output

class LLMInterface:
    def __init__(self, model_name, prompt_mode, prompt, num_incontext_examples, temperature, seed, api_timeout, debug=False):
        if num_incontext_examples == -1:
            num_incontext_examples = len(prompt["incontext-examples"])

        # Initialize the provided member variables
        self.model_name = model_name
        self.prompt_mode = prompt_mode
        self.prompt = Prompt.from_dict( prompt )
        self.num_incontext_examples = num_incontext_examples
        self.temperature = temperature
        self.seed = seed
        self.api_timeout = api_timeout
        self.response_history = []
        self.debug = debug

        # Validate the initialization
        self.__validate_initialization()

        # Create the LLM client
        self.__create_client()

    def __repr__(self):
        return repr( self.to_dict() )

    @classmethod
    def from_dict(cls, d):
        # Load from the dict
        self.model_name = d["model_name"]
        self.prompt_mode = d["prompt_mode"]
        self.prompt = Prompt.from_dict( d["prompt"] )
        self.num_incontext_examples = d["num_incontext_examples"]
        self.temperature = d["temperature"]
        self.seed = d["seed"]
        self.api_timeout = d["api_timeout"]
        self.response_history = d["response_history"]

        # Validate the initialization
        self.__validate_initialization()

        # Create the LLM client
        self.__create_client()

    def __validate_initialization(self):
        if not self.model_name in models_map.keys():
            raise ConstructionFailure(f"The model \"{self.model_name}\" was not recognized.")
        if not self.prompt_mode in prompt_modes_set:
            raise ConstructionFailure(f"The prompt mode \"{self.prompt_mode}\" was not recognized.")
        if self.num_incontext_examples < len(self.prompt.incontext_examples):
            raise ConstructionFailure(f"The requested number of incontext examples ({self.num_incontext_examples}) is greater than the number available ({len(self.prompt.incontext_examples)}).")
        if self.debug:
            print("Successfully validated the initialization.")
        return

    def __create_client(self):
        # Create LLM clients per the model
        if "gpt" in self.model_name:
            import openai
            self.openai_client = openai.OpenAI(
                api_key = os.getenv("OPENAI_API_KEY"),
                timeout = self.api_timeout,
            )
        elif "anthropic" in self.model_name:
            import anthropic
            aws_region = "us-west-2"
            self.anthropic_client = anthropic.AnthropicBedrock(
                aws_access_key = os.getenv("AWS_ACCESS_KEY"),
                aws_secret_key = os.getenv("AWS_SECRET_KEY"),
                aws_region = aws_region,
            )
        elif "ollama" in self.model_name or "local" in self.model_name:
            import ollama
            self.url = os.getenv("LOCAL_LLM_URL", "http://localhost:11434")
            self.ollama_client = ollama.Client(self.url)
        else:
            raise ConstructionFailure(f"Unable to create a client for the model \"{self.model_name}\".")
        if self.debug:
            if hasattr(self, "openai_client"):
                print(f"Successfully created an OpenAI client for model \"{self.model_name}\"".)
            if hasattr(self, "anthropic_client"):
                print(f"Successfully created an Anthropic client for model \"{self.model_name}\"".)
            if hasattr(self, "ollama_client"):
                print(f"Successfully created an Ollama client for model \"{self.model_name}\"".)
        return

    def to_dict(self):
        d = {}
        if self.model_name:
            d["model_name"] = self.model_name
        if self.prompt_mode:
            d["prompt_mode"] = self.prompt_mode
        if self.prompt:
            d["model"] = self.prompt.to_dict()
        if self.num_incontext_examples:
            d["num_incontext_examples"] = self.num_incontext_examples
        if self.temperature:
            d["temperature"] = self.temperature
        if self.seed:
            d["seed"] = self.seed
        if self.api_timeout:
            d["api_timeout"] = self.api_timeout
        if self.response_history:
            d["response_history"] = self.response_history
        return d
        
    def query_llm(self, instruction, scene_graph):
        if not self.prompt:
            raise PromptingFailure("No prompt available in query_llm().")
        # Set the instruction & scene graph in the prompt
        self.prompt.new_instruction = instruction
        self.prompt.new_scene_graph = scene_graph
        if hasattr(self, "openai_client"):
            response = self.__query_openai()
        elif hasattr(self, "anthropic_client"):
            response = self.__query_anthropic()
        elif hasattr(self, "ollama_client"):
            response = self.__query_ollama()
        else:
            raise PromptingFailure("Unexpected lack of language model client in query_llm()")
        if self.debug:
            print(f"Response: {response}")
        self.response_history.append(response)
        return response

    def __query_openai(self):
        if self.debug:
            print(f"Querying OpenAI Cloud API ({self.model_name}).\nCurrent Prompt: {self.prompt.to_openai()}", flush=True)
        response = self.openai_client.chat.completions.create(
            model = self.model_name,
            messages = self.prompt.to_openai_messages(),
            temperature = self.temperature,
            seed = self.seed,
        )
        return response

    def __query_anthropic(self):
        if self.debug:
            print(f"Querying Anthropic Cloud API ({self.model_name}).\nCurrent Prompt: {self.prompt.to_anthropic()}", flush=True)
        response = self.anthropic_client.messages.create(
            model = self.model_name,
            system = self.prompt.system,
            messages = self.prompt.to_anthropic(),
            max_tokens = 4096,
            temperature = self.temperature,
        )
        return response

    def __query_ollama(self):
        if self.debug:
            print(f"Querying Ollama API ({self.model_name}).\n Current Prompt: {self.prompt.to_ollama()}", flush=True)
        response = self.ollama_client.chat(
            model = models_map.get( self.model_name ),
            message = self.prompt.to_ollama(),
        )
        return response
