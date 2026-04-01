#!/usr/bin/env python
import argparse
import ast

import spark_dsg
from ruamel.yaml import YAML

from nlu_interface.config import OpenAIConfig
from nlu_interface_dcist.language_planning_interface import (
    LanguagePddlInterface,
    SimplePddlSceneGraphPrompt,
)

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

    # Load the scene graph
    scene_graph = spark_dsg.DynamicSceneGraph.load(config["scene_graph"])

    # Load the LLM configuration and prompt
    with open(config["llm_config"], "r") as file:
        llm_config = yaml.load(file)
    with open(llm_config["prompt"], "r") as file:
        prompt = SimplePddlSceneGraphPrompt(**yaml.load(file))

    config = OpenAIConfig(
        model=llm_config["model"],
        prompt_mode=llm_config.get("mode", "default"),
        prompt_type=llm_config.get("prompt_type", "default"),
        num_incontext_examples=llm_config["num_incontext_examples"],
        temperature=llm_config["temperature"],
        api_timeout=llm_config["api_timeout"],
        seed=llm_config["seed"],
        api_key_env_var=llm_config.get("api_key_env_var", ""),
        debug=llm_config.get("debug", False),
    )
    openai_interface = LanguagePddlInterface(config=config, prompt=prompt)

    main(openai_interface, scene_graph)
