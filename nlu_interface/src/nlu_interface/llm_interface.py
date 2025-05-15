import os
import re
from abc import ABC, abstractmethod
from types import MappingProxyType

# Scene Graphs
import spark_dsg

# Used as a hook into ROS logging
import logging
logger = logging.getLogger(__name__)


# The imports below are triggered conditionally according to the required client. Leaving them commented here for visibility.
# import openai
# import ollama
# import anthropic

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
models_map = MappingProxyType(
    {
        "gpt-4o" : "gpt-4o-2024-08-06",
        "gpt-4o-mini" : "gpt-4o-mini-2024-07-18",
    }
)

prompt_modes_set = (
    "default",
    "chain-of-thought",
)

""" Helper Methods
"""
def parse_response( response_string ):
    response_match = re.search(r"<Answer>(.*?)</Answer>", response_string)
    if response_match:
        parsed_response = response_match.group(1)
    else:
        raise ParsingFailure(f"Unable to parse the answer from the response: {response_string}")
    return parsed_response

def get_region_parent_of_object(object_node, scene_graph):
    # Get the parent of the object (Place)
    parent_place_id = object_node.get_parent()
    if not parent_place_id:
        return "none"
    parent_place_node = scene_graph.get_node( parent_place_id )
    parent_region_id = parent_place_node.get_parent()
    if not parent_region_id:
        return "none"
    parent_region_node = scene_graph.get_node( parent_region_id )
    return str(parent_region_node.id)

def object_to_prompt(object_node, scene_graph):
    attrs = object_node.attributes
    symbol = str(object_node.id)
    object_labelspace = scene_graph.get_labelspace(2,0)
    if not object_labelspace:
        raise PromptingFailure(f"No available object labelspace") 
    semantic_type = object_labelspace.get_category( attrs.semantic_label )
    position = f"({attrs.position[0]},{attrs.position[1]})"
    parent_region = get_region_parent_of_object(object_node, scene_graph)
    object_prompt = f"\n-\t(id={symbol}, type={semantic_type}, pos={position}, parent_region={parent_region})"
    return object_prompt

def region_to_prompt(region_node, scene_graph):
    attrs = region_node.attributes
    symbol = str(region_node.id)
    region_labelspace = scene_graph.get_labelspace(4,0)
    if not region_labelspace:
        raise PromptingFailure("No available region labelspace")
    semantic_type = region_labelspace.get_category( attrs.semantic_label )
    region_prompt = f"\n-\t(id={symbol}, type={semantic_type})"
    return region_prompt

def scene_graph_to_prompt(scene_graph):
    # Add the objects
    objects_prompt = ""
    for object_node in scene_graph.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes:
        objects_prompt += object_to_prompt(object_node, scene_graph)
    # Add the regions
    regions_prompt = ""
    for region_node in scene_graph.get_layer(spark_dsg.DsgLayers.ROOMS).nodes:
        regions_prompt += region_to_prompt(region_node, scene_graph)
    # Construct the prompt
    scene_graph_prompt = (
        f"<Scene Graph>"
        f"\nObjects: {objects_prompt}"
        f"\nRegions: {regions_prompt}"
        f"</Scene Graph>"
    )
    return scene_graph_prompt

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
        scene_graph=None,
        response_format=None,
    ):
        self.system = system
        self.incontext_examples_preamble = incontext_examples_preamble
        self.incontext_examples = incontext_examples
        self.new_instruction_preamble = new_instruction_preamble
        self.new_instruction = new_instruction
        self.scene_graph = scene_graph
        self.response_format = response_format

    def __repr__(self):
        return repr( self.to_dict() )

    @classmethod
    def from_dict(cls, d, new_instruction=None, scene_graph=None):
        system = d["system"]
        incontext_examples_preamble = d["incontext-examples-preamble"]
        incontext_examples = [ IncontextExample.from_dict(e) for e in d["incontext-examples"] ]
        new_instruction_preamble = d["new-instruction-preamble"]
        response_format = d["response-format"]
        
        # Only load the "new instruction" from the dictionary if no function arg provided
        if not new_instruction and "new-instruction" in d:
            new_instruction = d["new-instruction"]
        if not scene_graph and "scene-graph" in d:
            scene_graph = d["scene-graph"]

        return cls(
            system,
            incontext_examples_preamble,
            incontext_examples,
            new_instruction_preamble,
            new_instruction,
            scene_graph,
            response_format,
        )

    def to_dict(self):
        d = {}
        if self.system:
            d["system"] = self.system
        if self.incontext_examples_preamble:
            d["incontext-examples-preamble"] = self.incontext_examples_preamble
        if self.incontext_examples:
            d["incontext-examples"] = self.incontext_examples
        if self.new_instruction_preamble:
            d["new-instruction-preamble"] = self.new_instruction_preamble
        if self.new_instruction:
            d["new-instruction"] = self.new_instruction
        if self.scene_graph:
            d["scene-graph"] = self.scene_graph
        if self.response_format:
            d["response-format"] = self.response_format
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
            user_message_content += "\n" + self.incontext_examples[i].to_prompt()
        user_message_content += self.new_instruction_preamble.format(response_format=self.response_format)
        user_message_content += self.scene_graph
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
        example_input = d["example-input"]
        example_output = d["example-output"]
        return cls(example_input, example_output)

    def to_dict(self):
        d = {}
        if self.example_input:
            d["example-input"] = self.example_input
        if self.example_output:
            d["example-output"] = self.example_output
        return d

    def to_prompt(self):
        return self.example_input + "\n" + self.example_output

class LLMInterface(ABC):
    valid_model_names = ()
    valid_prompt_modes = ()

    def __init__(self,
        model: str,
        mode: str,
        prompt: dict,
        num_incontext_examples: int,
        temperature: float,
        api_key_env_var: str="",
        debug: bool=False,
    ):
        self.model = model
        self.mode = mode
        self.prompt = Prompt.from_dict( prompt )
        self.num_incontext_examples = len(self.prompt.incontext_examples) if num_incontext_examples == -1 else num_incontext_examples
        self.temperature = temperature
        self.debug = debug
        self.response_history = []
        if api_key_env_var:
            self.api_key = os.getenv( api_key_env_var )
        else:
            logger.warning("No API key environment variable provided, using OPENAI_API_KEY as a default.")
            self.api_key = os.getenv("OPENAI_API_KEY")

    @classmethod
    @abstractmethod
    def from_dict(cls, d):
        raise NotImplementedError("Subclasses must implement \"from_dict()\".")

    @abstractmethod
    def to_dict(self):
        raise NotImplementedError("Subclasses must implement \"to_dict()\".")

    @abstractmethod
    def _validate_initialization(self):
        raise NotImplementedError("Subclasses must implement \"_validate_initialization()\".")

    @abstractmethod
    def _create_client(self):
        raise NotImplementedError("Subclasses must implement \"_create_client()\".")

    @abstractmethod
    def _query(self):
        raise NotImplementedError("Subclasses must implement \"_query()\".")

class OpenAIWrapper(LLMInterface):
    valid_model_names = ("gpt-4o", "gpt-4o-mini")
    valid_prompt_modes = ("default", "chain-of-thought")
    def __init__(self,
        model: str,
        mode: str,
        prompt: dict,
        num_incontext_examples: int,
        temperature: float,
        api_timeout: float,
        seed: int,
        api_key_env_var: str="",
        debug: bool=False,
    ):
        super().__init__(model, mode, prompt, num_incontext_examples, temperature, api_key_env_var, debug)
        self.api_timeout = api_timeout
        self.seed = seed
        self._validate_initialization()
        self._create_client()
        return

    def __repr__(self):
        return repr( self.to_dict() )

    @classmethod
    def from_dict(cls, d):
        # Load from the dict
        model = d["model"]
        mode = d["mode"]
        prompt = Prompt.from_dict( d["prompt"] )
        num_incontext_examples = d["num-incontext-examples"]
        temperature = d["temperature"]
        api_timeout = d["api-timeout"]
        seed = d["seed"]
        api_key_env_var = d["api-key-env-var"]
        debug = d["debug"]
        return cls(model, mode, prompt, num_incontext_examples, temperature, api_timeout, seed, api_key_env_var, debug)

    def _validate_initialization(self):
        if not self.model in self.valid_model_names:
            raise ConstructionFailure(f"The model \"{self.model}\" was not recognized (valid models: {str(self.valid_model_names)}).")
        if not self.mode in self.valid_prompt_modes:
            raise ConstructionFailure(f"The prompt mode \"{self.mode}\" was not recognized (valid modes: {str(self.valid_prompt_modes)}).")
        if self.num_incontext_examples < len(self.prompt.incontext_examples):
            raise ConstructionFailure(f"The requested number of incontext examples ({self.num_incontext_examples}) is greater than the number available ({len(self.prompt.incontext_examples)}).")
        if self.debug:
            logger.debug("Successfully validated the initialization.")  
        return

    def _create_client(self):
        import openai
        self.client = openai.OpenAI(
            api_key = self.api_key,
            timeout = self.api_timeout,
        )
        if self.debug:
            logger.debug(f"Successfully created an OpenAI client for model \"{self.model}\".")
        return

    def to_dict(self):
        d = {}
        if self.model:
            d["model"] = self.model
        if self.prompt_mode:
            d["mode"] = self.mode
        if self.prompt:
            d["prompt"] = self.prompt.to_dict()
        if self.num_incontext_examples:
            d["num-incontext-examples"] = self.num_incontext_examples
        if self.temperature:
            d["temperature"] = self.temperature
        if self.seed:
            d["seed"] = self.seed
        if self.api_timeout:
            d["api-timeout"] = self.api_timeout
        if self.api_key_env_var:
            d["api-key-env-var"] = self.api_key_env_var
        if self.debug:
            d["debug"] = self.debug
        return d

    def request_plan_specification(self, instruction, scene_graph):
        # Prepare the prompt
        self.prompt.new_instruction = instruction
        self.prompt.scene_graph = scene_graph_to_prompt( scene_graph )
        response = self._query()
        parsed_response_content = parse_response( response.choices[0].message.content )
        return parsed_response_content
 
    def _query(self):
        if not self.prompt:
            raise PromptingFailure("No prompt available in query.")
        messages = self.prompt.to_openai_messages(self.num_incontext_examples)
        if self.debug:
            logger.debug(f"Querying OpenAI Cloud API ({self.model}).\nCurrent Prompt: {messages}")
        response = self.client.chat.completions.create(
            model = self.model,
            messages = messages,
            temperature = self.temperature,
            seed = self.seed,
        )
        self.response_history.append(response)
        return response


















#class LLMInterface:
#    def __init__(self, model_name, prompt_mode, prompt, num_incontext_examples, temperature, seed, api_timeout, debug=False):
#        if num_incontext_examples == -1:
#            num_incontext_examples = len(prompt["incontext-examples"])
#
#        # Initialize the provided member variables
#        self.model_name = model_name
#        self.prompt_mode = prompt_mode
#        self.prompt = Prompt.from_dict( prompt )
#        self.num_incontext_examples = num_incontext_examples
#        self.temperature = temperature
#        self.seed = seed
#        self.api_timeout = api_timeout
#        self.response_history = []
#        self.debug = debug
#
#        # Validate the initialization
#        self.__validate_initialization()
#
#        # Create the LLM client
#        self.__create_client()
#
#    def __repr__(self):
#        return repr( self.to_dict() )
#
#    @classmethod
#    def from_dict(cls, d):
#        # Load from the dict
#        self.model_name = d["model-name"]
#        self.prompt_mode = d["prompt-mode"]
#        self.prompt = Prompt.from_dict( d["prompt"] )
#        self.num_incontext_examples = d["num-incontext-examples"]
#        self.temperature = d["temperature"]
#        self.seed = d["seed"]
#        self.api_timeout = d["api-timeout"]
#        self.response_history = d["response-history"]
#
#        # Validate the initialization
#        self.__validate_initialization()
#
#        # Create the LLM client
#        self.__create_client()
#
#    def __validate_initialization(self):
#        if not self.model_name in models_map.keys():
#            raise ConstructionFailure(f"The model \"{self.model_name}\" was not recognized.")
#        if not self.prompt_mode in prompt_modes_set:
#            raise ConstructionFailure(f"The prompt mode \"{self.prompt_mode}\" was not recognized.")
#        if self.num_incontext_examples < len(self.prompt.incontext_examples):
#            raise ConstructionFailure(f"The requested number of incontext examples ({self.num_incontext_examples}) is greater than the number available ({len(self.prompt.incontext_examples)}).")
#        if self.debug:
#            print("Successfully validated the initialization.")
#        return
#
#    def __create_client(self):
#        # Create LLM clients per the model
#        if "gpt" in self.model_name:
#            import openai
#            self.openai_client = openai.OpenAI(
#                api_key = os.getenv("OPENAI_API_KEY"),
#                timeout = self.api_timeout,
#            )
#        elif "anthropic" in self.model_name:
#            import anthropic
#            aws_region = "us-west-2"
#            self.anthropic_client = anthropic.AnthropicBedrock(
#                aws_access_key = os.getenv("AWS_ACCESS_KEY"),
#                aws_secret_key = os.getenv("AWS_SECRET_KEY"),
#                aws_region = aws_region,
#            )
#        elif "ollama" in self.model_name or "local" in self.model_name:
#            import ollama
#            self.url = os.getenv("LOCAL_LLM_URL", "http://localhost:11434")
#            self.ollama_client = ollama.Client(self.url)
#        else:
#            raise ConstructionFailure(f"Unable to create a client for the model \"{self.model_name}\".")
#        if self.debug:
#            if hasattr(self, "openai_client"):
#                print(f"Successfully created an OpenAI client for model \"{self.model_name}\".")
#            if hasattr(self, "anthropic_client"):
#                print(f"Successfully created an Anthropic client for model \"{self.model_name}\".")
#            if hasattr(self, "ollama_client"):
#                print(f"Successfully created an Ollama client for model \"{self.model_name}\".")
#        return
#
#    def to_dict(self):
#        d = {}
#        if self.model_name:
#            d["model-name"] = self.model_name
#        if self.prompt_mode:
#            d["prompt-mode"] = self.prompt_mode
#        if self.prompt:
#            d["model"] = self.prompt.to_dict()
#        if self.num_incontext_examples:
#            d["num-incontext-examples"] = self.num_incontext_examples
#        if self.temperature:
#            d["temperature"] = self.temperature
#        if self.seed:
#            d["seed"] = self.seed
#        if self.api_timeout:
#            d["api-timeout"] = self.api_timeout
#        if self.response_history:
#            d["response-history"] = self.response_history
#        return d
#
#    def request_plan_specification(self, instruction, scene_graph):
#        # Prepare the prompt
#        self.prompt.new_instruction = instruction
#        self.prompt.scene_graph = scene_graph_to_prompt( scene_graph )
#        response = self.query_llm()
#        parsed_response_content = parse_response( response.choices[0].message.content )
#        return parsed_response_content
#        
#    def query_llm(self):
#        if not self.prompt:
#            raise PromptingFailure("No prompt available in query_llm().")
#        if hasattr(self, "openai_client"):
#            response = self.__query_openai()
#        elif hasattr(self, "anthropic_client"):
#            response = self.__query_anthropic()
#        elif hasattr(self, "ollama_client"):
#            response = self.__query_ollama()
#        else:
#            raise PromptingFailure("Unexpected lack of language model client in query_llm()")
#        if self.debug:
#            print(f"Response: {response}")
#        self.response_history.append(response)
#        return response
#
#    def __query_openai(self):
#        messages = self.prompt.to_openai_messages(self.num_incontext_examples)
#        if self.debug:
#            print(f"Querying OpenAI Cloud API ({self.model_name}).\nCurrent Prompt: {messages}", flush=True)
#        response = self.openai_client.chat.completions.create(
#            model = self.model_name,
#            messages = messages,
#            temperature = self.temperature,
#            seed = self.seed,
#        )
#        return response
#
#    def __query_anthropic(self):
#        if self.debug:
#            print(f"Querying Anthropic Cloud API ({self.model_name}).\nCurrent Prompt: {self.prompt.to_anthropic()}", flush=True)
#        response = self.anthropic_client.messages.create(
#            model = self.model_name,
#            system = self.prompt.system,
#            messages = self.prompt.to_anthropic(),
#            max_tokens = 4096,
#            temperature = self.temperature,
#        )
#        return response
#
#    def __query_ollama(self):
#        if self.debug:
#            print(f"Querying Ollama API ({self.model_name}).\n Current Prompt: {self.prompt.to_ollama()}", flush=True)
#        response = self.ollama_client.chat(
#            model = models_map.get( self.model_name ),
#            message = self.prompt.to_ollama(),
#        )
#        return response
