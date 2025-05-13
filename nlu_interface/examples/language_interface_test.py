#!/usr/bin/env python
import ast
import os
import argparse
import yaml

import spark_dsg

from nlu_interface.llm_interface import Prompt, IncontextExample, LLMInterface

def main(llm_interface, scene_graph):
    new_instruction = "Spot, go to region 83"
    response = llm_interface.request_plan_specification( new_instruction, scene_graph )
    print(f"response: {response}")
    response_dict = ast.literal_eval(response)
    print(f"response_dict: {response_dict}")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    # Load the config
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # Load the scene graph and the associated labelspaces; Force overwriting the metadata for now
    scene_graph = spark_dsg.DynamicSceneGraph.load(config["scene_graph"])
    labelspace_metadata = {}
    with open(config["object_labelspace"], 'r') as file:
        object_labelspace = yaml.safe_load(file)
    id_to_label = {item["label"] : item["name"] for item in object_labelspace["label_names"]}
    labelspace_metadata["object_labelspace"] = id_to_label
    with open(config["region_labelspace"], 'r') as file:
        region_labelspace = yaml.safe_load(file)
    id_to_label = {item["label"] : item["name"] for item in region_labelspace["label_names"]}
    labelspace_metadata["region_labelspace"] = id_to_label
    scene_graph.metadata.set( labelspace_metadata )

    # Load the LLM configuration and prompt
    with open(config["llm_config"], 'r') as file:
        llm_config = yaml.safe_load(file)
    with open(llm_config["prompt"], 'r') as file:
        prompt = yaml.safe_load(file)

    # Initialize LLMInterface
    llm_interface = LLMInterface(
        model_name = llm_config["model_name"],
        prompt_mode = llm_config["prompt_mode"],
        prompt = prompt,
        num_incontext_examples = llm_config["num_incontext_examples"],
        temperature = llm_config["temperature"],
        seed = llm_config["seed"],
        api_timeout = llm_config["api_timeout"],
        debug = llm_config["debug"],
    )

    main(llm_interface, scene_graph) 
