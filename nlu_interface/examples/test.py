#!/usr/bin/env python
import argparse
import ast
import re

from ruamel.yaml import YAML

from nlu_interface.config import LLMConfig, OpenAIConfig, AnthropicBedrockConfig, OllamaConfig
from nlu_interface.interface import OpenAIWrapper, AnthropicBedrockWrapper, OllamaWrapper
from nlu_interface.prompt import Prompt, DefaultPrompt

yaml = YAML(typ="safe")

def parse_response(response_string) -> str:
    response_match = re.search(r"<Answer>(.*?)</Answer>", response_string)
    if response_match:
        parsed_response = response_match.group(1)
    else:
        raise ValueError(f"Unable to parse the answer from the response: {response_string}")
    return parsed_response

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_config", required=True, type=str)
    parser.add_argument("--prompt", required=True, type=str)
    args = parser.parse_args()

    # Load the prompt
    with open(args.prompt, "r") as file:
        prompt_dict = yaml.load(file)
    prompt = DefaultPrompt(**prompt_dict)
    print(f"Load the prompt: {prompt.render(0)}")


    # Load the config
    with open(args.llm_config, "r") as file:
        llm_config_dict = yaml.load(file)
    # Construct the interface
    interface_type = llm_config_dict["interface_type"]
    del llm_config_dict["interface_type"]
    print(f"Constructing an interface of type \"{interface_type}\".")
    llm_interface = None
    match interface_type:
        case "openai":
            llm_interface = OpenAIWrapper(config=OpenAIConfig(**llm_config_dict), prompt=prompt)
        case "anthropic":
            llm_interface = AnthropicBedrockWrapper(config=AnthropicBedrockConfig(**llm_config_dict), prompt=prompt)
        case "ollama":
            llm_interface = OllamaWrapper(config=OllamaConfig(**llm_config_dict), prompt=prompt)
        case _:
            raise ValueError(f"Unrecognized interface type: {interface_type}.")

    # Query via the OpenAIWrapper
    print("Querying the model...")
    response_text, response = llm_interface._query()
    print(f"Response Text: {response_text}")
    print(f"Response: {response}")

    # Parse the response
    print("Parsing the response...")
    answer = parse_response(response_text)
    print(f"Parsed Response: {answer}")

    exit(0)
