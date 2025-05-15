#!/usr/bin/env python
import argparse
import ast

import spark_dsg
from ruamel.yaml import YAML

from nlu_interface.llm_interface import OpenAIWrapper

yaml = YAML(typ="safe")


def main(openai_interface, scene_graph):
    new_instruction = "Spot, go to region 83"
    response = openai_interface.request_plan_specification(new_instruction, scene_graph)
    print(f"response: {response}")
    response_dict = ast.literal_eval(response)
    print(f"response_dict: {response_dict}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    # Load the config
    with open(args.config, "r") as file:
        config = yaml.load(file)

    # Load the scene graph and the associated labelspaces; Force overwriting the metadata for now
    scene_graph = spark_dsg.DynamicSceneGraph.load(config["scene_graph"])

    # Load the LLM configuration and prompt
    with open(config["llm_config"], "r") as file:
        llm_config = yaml.load(file)
    with open(llm_config["prompt"], "r") as file:
        prompt = yaml.load(file)

    # Initialize an OpenAIWrapper
    openai_interface = OpenAIWrapper(
        model=llm_config["model"],
        mode=llm_config["mode"],
        prompt=prompt,
        num_incontext_examples=llm_config["num_incontext_examples"],
        temperature=llm_config["temperature"],
        api_timeout=llm_config["api_timeout"],
        seed=llm_config["seed"],
        api_key_env_var=llm_config["api_key_env_var"],
        debug=llm_config["debug"],
    )

    main(openai_interface, scene_graph)
