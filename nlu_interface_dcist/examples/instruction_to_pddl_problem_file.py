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

def main(llm_interface, pddl_problem_string, output_filepath):
    print(f"Querying to translate the instruction: {llm_interface.prompt.instruction}")
    response_text, response = llm_interface._query()
    goal_string = parse_response(response_text)
    print(f"Received the goal string: {goal_string}")
    pddl_problem_string = pddl_problem_string.format(goal=goal_string)
    with open(output_filepath, "w") as file:
        file.write(pddl_problem_string)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_config", required=True, type=str)
    parser.add_argument("--prompt", required=True, type=str)
    parser.add_argument("--pddl_problem", required=True, type=str)
    parser.add_argument("--instruction", required=False, type=str, default=None)
    parser.add_argument("--output", required=False, default="/tmp/problem.pddl")
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

    # Load the PDDL problem file template
    with open(args.pddl_problem, "r") as file:
        pddl_problem_string = file.read()
    print("Loaded the pddl problem file template")

    # Check that there is an instruction
    if args.instruction is None and llm_interface.prompt.instruction is None:
        raise ValueError("No instruction provided. Please add an instruction to either the prompt file or using the command line arg.")

    if args.instruction is not None:
        llm_interface.prompt.instruction = args.instruction
    main(llm_interface, pddl_problem_string, args.output)
