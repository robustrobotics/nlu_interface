#!/usr/bin/env python
import argparse
import ast

import spark_dsg
from ruamel.yaml import YAML

from nlu_interface.config import LLMConfig, OpenAIConfig
from nlu_interface.prompt import Prompt, DefaultPrompt
from nlu_interface.interface import OpenAIWrapper

yaml = YAML(typ="safe")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--prompt", required=True, type=str)
    args = parser.parse_args()

    # Load the config
    with open(args.config, "r") as file:
        config = yaml.load(file)

    # Make an LLMConfig
    print("Constructing an LLMConfig...")
    llm_config = LLMConfig(
        model=config["model"],
        prompt_mode=config["prompt_mode"],
        prompt_type=config["prompt_type"],
        num_incontext_examples=config["num_incontext_examples"],
        temperature=config["temperature"],
        debug=config["debug"],
    )
    print("Success!")

    # Make an OpenAIConfig
    print("Constructing an OpenAIConfig...")
    openai_config = OpenAIConfig(
        model=config["model"],
        prompt_mode=config["prompt_mode"],
        prompt_type=config["prompt_type"],
        num_incontext_examples=config["num_incontext_examples"],
        temperature=config["temperature"],
        api_key_env_var=config["api_key_env_var"],
        api_timeout=config["api_timeout"],
        seed=config["seed"],
        debug=config["debug"],
    )
    api_key = openai_config.resolve_api_key()
    print(f"Loaded the api_key: {api_key}")
    print("Success!")
    print("Constructing an OpenAIConfig using **kwargs (passing the config dict)")
    openai_config = OpenAIConfig(**config)
    api_key = openai_config.resolve_api_key()
    print(f"Loaded the api_key: {api_key}")
    print("Success!")

    # Load a prompt
    with open(args.prompt, "r") as file:
        prompt_dict = yaml.load(file)
    # Make a BaiscPrompt
    print("Constructing a DefaultPrompt...")
    default_prompt = DefaultPrompt(
        system=prompt_dict["system"],
        incontext_examples_preamble=prompt_dict["incontext_examples_preamble"],
        incontext_examples=prompt_dict["incontext_examples"],
        instruction_preamble=prompt_dict["instruction_preamble"],
        instruction=prompt_dict["instruction"],
        response_format=prompt_dict["response_format"],
    )
    print("Success!")
    print("Constructing a DefaultPrompt using **kwargs...")
    default_prompt = DefaultPrompt( **prompt_dict )
    print("Success!")

    # Render the prompt
    print("Rending the DefaultPrompt...")
    rendered_prompt = default_prompt.render( 1 )
    print(f"rendered prompt: {rendered_prompt}")
    print("Success!")

    # Construct an OpenAIWrapper
    print("Constructing an OpenAIWrapper")
    openai_wrapper = OpenAIWrapper(
        config=openai_config,
        prompt=default_prompt,
    )
    print("Success!")

    # Query via the OpenAIWrapper
    print("Querying OpenAI...")
    response = openai_wrapper._query()
    print(f"Response: {response}")
    print("Success!")

    exit(0)
