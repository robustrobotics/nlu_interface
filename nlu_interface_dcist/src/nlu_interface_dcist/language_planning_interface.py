#!/usr/bin/env python
import logging
import os
import re
from nlu_interface import OpenAIWrapper, Prompt, IncontextExample

# Scene Graphs
import spark_dsg

logger = logging.getLogger(__name__)

""" Helper Methods
"""
def get_region_parent_of_object(object_node, scene_graph):
    # Get the parent of the object (Place)
    parent_place_id = object_node.get_parent()
    if not parent_place_id:
        return "none"
    parent_place_node = scene_graph.get_node(parent_place_id)
    parent_region_id = parent_place_node.get_parent()
    if not parent_region_id:
        return "none"
    parent_region_node = scene_graph.get_node(parent_region_id)
    return str(parent_region_node.id)

def object_to_prompt(object_node, scene_graph):
    attrs = object_node.attributes
    symbol = str(object_node.id)
    object_labelspace = scene_graph.get_labelspace(2, 0)
    if not object_labelspace:
        raise PromptingFailure("No available object labelspace")
    semantic_type = object_labelspace.get_category(attrs.semantic_label)
    position = f"({attrs.position[0]},{attrs.position[1]})"
    parent_region = get_region_parent_of_object(object_node, scene_graph)
    object_prompt = f"\n-\t(id={symbol}, type={semantic_type}, pos={position}, parent_region={parent_region})"
    return object_prompt

def region_to_prompt(region_node, scene_graph):
    attrs = region_node.attributes
    symbol = str(region_node.id)
    region_labelspace = scene_graph.get_labelspace(4, 0)
    if not region_labelspace:
        raise PromptingFailure("No available region labelspace")
    semantic_type = region_labelspace.get_category(attrs.semantic_label)
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

class SimplePddlSceneGraphPrompt(Prompt):
    def __init__(
        self,
        system=None,
        incontext_examples_preamble=None,
        incontext_examples=None,
        instruction_preamble=None,
        instruction=None,
        response_format=None,
        scene_graph=None,
    ):
        super().__init__(
            system=system,
            incontext_examples_preamble=incontext_examples_preamble,
            incontext_examples=incontext_examples,
            instruction_preamble=instruction_preamble,
            instruction=instruction,
            response_format=response_format,
        )
        self.scene_graph = scene_graph

#    @classmethod
#    def from_dict(cls, d, instruction=None, scene_graph=None):
#        system = d["system"] 
#        incontext_examples_preamble = d["incontext-examples-preamble"]
#        incontext_examples = [
#            IncontextExample.from_dict(e) for e in d["incontext-examples"]
#        ]
#        instruction_preamble = d["instruction-preamble"]
#        response_format = d["response-format"]
#        # Only load the instruction and scene graph if no function arg provided
#        if not instruction and "instruction" in d:
#            instruction = d["instruction"]
#        if not scene_graph and "scene-graph" in d:
#            scene_graph = d["scene-graph"]
#
#        return cls(
#            system=system,
#            incontext_examples_preamble=incontext_examples_preamble,
#            incontext_examples=incontext_examples,
#            instruction_preamble=instruction_preamble,
#            instruction=instruction,
#            response_format=response_format,
#            scene_graph=scene_graph,
#        )
#        
#
#    def to_dict(self, include_instruction: bool = False, include_scene_graph: bool = False):
#        d = {}
#        d["system"] = self.system
#        d["incontext-examples-preamble"] = self.incontext_examples_preamble
#        d["incontext-examples"] = [
#            e.to_dict() for e in self.incontext_examples
#        ]
#        d["instructon-preamble"] = self.instruction_preamble
#        d["response-format"] = self.response_format
#        if include_instruction:
#            d["instruction"] = self.instruction
#        if include_scene_graph:
#            d["scene-graph"] = self.scene_graph
#        return d

    def to_openai_messages(self, num_incontext_examples):
        if num_incontext_examples > len(self.incontext_examples):
            raise PromptingFailure(
                f"More requested incontext examples ({num_incontext_examples}) than available ({len(self.incontext_examples)})."
            )
        messages = []
        # Add the system message; Insert the system format via string formatting
        system_message_content = self.system.format(
            response_format=self.response_format
        )
        messages.append({"role": "system", "content": system_message_content})
        # Compose & Add the user message
        user_message_content = self.incontext_examples_preamble
        for i in range(num_incontext_examples):
            user_message_content += "\n" + self.incontext_examples[i].to_prompt()
        user_message_content += self.instruction_preamble.format(
            response_format=self.response_format
        )
        user_message_content += self.scene_graph
        user_message_content += self.instruction
        messages.append({"role": "user", "content": user_message_content})
        return messages

class LanguagePddlInterface(OpenAIWrapper):
    def __init__(
        self,
        model: str,
        prompt: Any,
        prompt_mode: str,
        prompt_type: str,
        num_incontext_examples: int,
        temperature: float,
        api_timeout: float,
        seed: int,
        api_key_env_var: str = "",
        debug: bool = False,
        
    ):
        # This should call OpenAIWrapper _validate_initialization & _create_client
        super().__init__(
            model=model,
            prompt=prompt,
            prompt_mode=prompt_mode,
            prompt_type=prompt_type,
            num_incontext_examples=num_incontext_examples,
            temperature=temperature,
            api_timeout=api_timeout,
            seed=seed,
            api_key_env_var=api_key_env_var,
            debug=debug,
        )
        return

    def parse_plan_specification_response(self, response_string):
        response_match = re.search(r"<Answer>(.*?)</Answer>", response_string)
        if response_match:
            parsed_response = response_match.group(1)
        else:
            raise ParsingFailure(
                f"Unable to parse the answer from the response: {response_string}"
            )
        return parsed_response

    def request_plan_specification(self, instruction, scene_graph):
        # Prepare the prompt
        self.prompt.new_instruction = instruction
        self.prompt.scene_graph = scene_graph_to_prompt(scene_graph)
        response = self._query()
        parsed_response_content = parse_response(response.choices[0].message.content)
        return parsed_response_content
