from nlu_interface.prompt import DefaultPrompt, IncontextExample
from typing import List, Dict, Tuple

class TampPrompt(DefaultPrompt):
    def __init__(
        self,
        system: str = None,
        domain_description: str = None,
        incontext_examples_preamble: str = None,
        incontext_examples: List[Dict[str, str]] = [],
        instruction_preamble: str = None,
        scene_graph: str = None,
        instruction: str = None,
        response_format: str = None,
    ): 
        self.system = system
        self.domain_description = domain_description
        self.incontext_examples_preamble = incontext_examples_preamble
        self.incontext_examples = [ IncontextExample.from_dict(e) for e in incontext_examples]
        self.instruction_preamble = instruction_preamble
        self.scene_graph = scene_graph
        self.instruction = instruction
        self.response_format = response_format

    def render(self, num_incontext_examples: int) -> str:
        """Assumes that both system and instruction will be templated to take a response_format; tries to fall back to empty strings"""
        self.validate(num_incontext_examples)
        # Construct the system message string
        system_string = ""
        if self.system:
            if self.response_format and ("{response_format}" in self.system):
                system_string = self.system.format(
                    response_format=self.response_format
                )
            else:
                system_string = self.system
        # Get the domain description
        domain_description_string = ""
        if self.domain_description:
            domain_description_string = self.domain_description
        # Compose the incontext examples string
        incontext_examples_string = ""
        if self.incontext_examples_preamble:
            incontext_examples_string = self.incontext_examples_preamble
            for i in range(num_incontext_examples):
                incontext_examples_string += "\n" + str(self.incontext_examples[i])
        # Compose the instruction string
        instruction_string = ""
        if self.instruction_preamble:
            if self.response_format and ("{response_format}" in self.instruction_preamble):
                instruction_string = self.instruction_preamble.format(
                    response_format=self.response_format
                )
            else:
                instruction_string = self.instruction_preamble
        if self.scene_graph:
            instruction_string += "\n" + self.scene_graph
        if self.instruction:
            instruction_string += "\n" + self.instruction
        ret = (
            f"System: {system_string}"
            f"\nUser: {domain_description_string}\n{incontext_examples_string}\n{instruction_string}"
        )
        return ret

    def to_openai(self, num_incontext_examples: int) -> Tuple[str, str]:
        """Assumes that both system and instruction will be templated to take a response_format; tries to fall back to empty strings"""
        self.validate(num_incontext_examples)
        # Construct the system message string
        system_string = ""
        if self.system:
            if self.response_format and ("{response_format}" in self.system):
                system_string = self.system.format(
                    response_format=self.response_format
                )
            else:
                system_string = self.system
        # Get the domain description string
        domain_description_string = self.domain_description
        # Compose the incontext examples string
        incontext_examples_string = ""
        if self.incontext_examples_preamble:
            incontext_examples_string = self.incontext_examples_preamble
            for i in range(num_incontext_examples):
                incontext_examples_string += "\n" + str(self.incontext_examples[i])
        # Compose the instruction string
        instruction_string = ""
        if self.instruction_preamble:
            if self.response_format and ("{response_format}" in self.instruction_preamble):
                instruction_string = self.instruction_preamble.format(
                    response_format=self.response_format
                )
            else:
                instruction_string = self.instruction_preamble
        if self.scene_graph:
            instruction_string += "\n" + self.scene_graph
        if self.instruction:
            instruction_string += "\n" + self.instruction
        user_string = domain_description_string + incontext_examples_string + instruction_string
        return system_string, user_string

    def to_ollama(self, num_incontext_examples: int) -> str:
        """Assumes that both system and instruction will be templated to take a response_format; tries to fall back to empty strings"""
        self.validate(num_incontext_examples)
        # Construct the system message string
        system_string = ""
        if self.system:
            if self.response_format and ("{response_format}" in self.system):
                system_string = self.system.format(
                    response_format=self.response_format
                )
            else:
                system_string = self.system
        # Get the domain description string
        domain_description_string = self.domain_description
        # Compose the incontext examples string
        incontext_examples_string = ""
        if self.incontext_examples_preamble:
            incontext_examples_string = self.incontext_examples_preamble
            for i in range(num_incontext_examples):
                incontext_examples_string += "\n" + str(self.incontext_examples[i])
        # Compose the instruction string
        instruction_string = ""
        if self.instruction_preamble:
            if self.response_format and ("{response_format}" in self.instruction_preamble):
                instruction_string = self.instruction_preamble.format(
                    response_format=self.response_format
                )
            else:
                instruction_string = self.instruction_preamble
        if self.scene_graph:
            instruction_string += "\n" + self.scene_graph
        if self.instruction:
            instruction_string += "\n" + self.instruction
        user_string = domain_description_string + incontext_examples_string + instruction_string
        full_sting = system_string + "\n" + user_string
        return full_sting

    def to_anthropic(self, num_incontext_examples: int) -> Tuple[str, List[Dict[str, str]]]:
        self.validate(num_incontext_examples)
        # Construct the system message string
        system_string = ""
        if self.system:
            if self.response_format and ("{response_format}" in self.system):
                system_string = self.system.format(
                    response_format=self.response_format
                )
            else:
                system_string = self.system
        # Get the domain description string
        domain_description_string = self.domain_description
        # Compose the incontext examples string
        incontext_examples_string = ""
        if self.incontext_examples_preamble:
            incontext_examples_string = self.incontext_examples_preamble
            for i in range(num_incontext_examples):
                incontext_examples_string += "\n" + str(self.incontext_examples[i])
        # Compose the instruction string
        instruction_string = ""
        if self.instruction_preamble:
            if self.response_format and ("{response_format}" in self.instruction_preamble):
                instruction_string = self.instruction_preamble.format(
                    response_format=self.response_format
                )
            else:
                instruction_string = self.instruction_preamble
        if self.scene_graph:
            instruction_string += "\n" + self.scene_graph
        if self.instruction:
            instruction_string += "\n" + self.instruction
        user_string = domain_description_string + incontext_examples_string + instruction_string
        user_message = [{"role": "user", "content": user_string}]
        return system_string, user_message
