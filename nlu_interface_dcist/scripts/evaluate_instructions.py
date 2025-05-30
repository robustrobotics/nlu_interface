#!/usr/bin/env python
import os
import argparse
import ast
import re
import json
import time

from typing import List
from dataclasses import dataclass, field

from nlu_interface.config import OpenAIConfig, AnthropicBedrockConfig, OllamaConfig
from nlu_interface.prompt import DefaultPrompt
from nlu_interface.interface import OpenAIWrapper, AnthropicBedrockWrapper, OllamaWrapper

from ruamel.yaml import YAML
yaml = YAML(typ="safe")

@dataclass
class Result:
    uid: int
    instruction: str
    ground_truth: str
    answer: str
    runtime: float
    prompt: str
    response: str
    model: str

    def to_dict(self):
        d = {
            "id" : self.uid,
            "instruction" : self.instruction,
            "ground_truth" : self.ground_truth,
            "answer" : self.answer,
            "runtime" : self.runtime,
            "prompt" : self.prompt,
            "response" : self.response,
            "model" : self.model,
        }
        return d

@dataclass
class Answer:
    uid: str
    goal: List[float] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d):
        goal = [ x for x in d["goal"] ]
        uid = d["id"]
        return cls(uid=uid, goal=goal)

@dataclass
class Instruction:
    uid: int
    text: str
    answer: List[Answer] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d):
        uid = d["id"]
        text = d["instruction"]
        answer = [ Answer.from_dict(a) for a in d["answer"] ]
        return cls(uid=uid, text=text, answer=answer)

def parse_response(response_string) -> str:
    response_match = re.search(r"<Answer>(.*?)</Answer>", response_string)
    if response_match:
        parsed_response = response_match.group(1)
    else:
        raise ValueError(f"Unable to parse the answer from the response: {response_string}")
    return parsed_response

def main(llm_interface, instructions):
    print("Start of the evaluation program.")
    count = 0
    results = []
    for instruction in instructions:
        print(f"Instruction {count} of {len(instructions)}")
        llm_interface.prompt.instruction = instruction.text
        print("Querying...")
        start_time = time.time()  # Start the timer
        output, response = llm_interface._query()
        end_time = time.time()  # End the timer
        runtime = end_time - start_time  # Calculate runtime
        print("Parsing...")
        parsed_answer = parse_response(output)
        print("Logging...")
        results.append(
            Result(
                uid=instruction.uid,
                instruction=instruction.text,
                ground_truth=str(instruction.answer),
                answer=parsed_answer,
                runtime=runtime,
                prompt=llm_interface.prompt.render(3),
                response=str(response),
                model=llm_interface.model,
            )
        )
        count += 1
    # Export the results to JSON
    timestr = time.strftime("%Y%m%d-%H%M%S")
    results_filepath = f"/tmp/results-{llm_interface.model}-{timestr}.json"
    results_dict = [result.to_dict() for result in results]
    with open(results_filepath, 'w', encoding='utf-8') as file:
        json.dump(results_dict, file, ensure_ascii=False, indent=4)
    return
    
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_config", required=True, type=str)
    parser.add_argument("--prompt", required=True, type=str)
    parser.add_argument("--instruction_files", nargs="+", required=True)
    args = parser.parse_args()

    # Load the prompt
    with open(args.prompt, "r") as file:
        prompt_dict = yaml.load(file)
    prompt = DefaultPrompt(**prompt_dict)
    
    print(f"Loaded prompt: {prompt}")

    # Load the config
    with open(args.llm_config, "r") as file:
        llm_config_dict = yaml.load(file)

    # Construct the interface
    interface_type = llm_config_dict["interface_type"]
    print(f"Interface type: {interface_type}")
    del llm_config_dict["interface_type"]
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

    # Load the instructions
    instructions = []
    for file in args.instruction_files:
        with open(file, "r") as f:
            list_of_instructions = yaml.load(f)
        for instruction in list_of_instructions:
            instructions.append( Instruction.from_dict(instruction) )
    
    main(llm_interface=llm_interface, instructions=instructions)
