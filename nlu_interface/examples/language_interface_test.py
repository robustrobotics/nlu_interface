#!/usr/bin/env python
import os
import argparse
import yaml
from nlu_interface.llm_interface import Prompt, IncontextExample, LLMInterface

def main(llm_interface):
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    with open(config["prompt"], 'r') as file:
        prompt = yaml.safe_load(file)

    # Initialize LLMInterface
    llm_interface = LLMInterface(
        model_name = config["model_name"],
        prompt_mode = config["prompt_mode"],
        prompt = prompt,
        num_incontext_examples = config["num_incontext_examples"],
        temperature = config["temperature"],
        seed = config["seed"],
        api_timeout = config["api_timeout"],
        debug = config["debug"],
    )

    #prompt = config["prompt"]

    main(llm_interface) 
