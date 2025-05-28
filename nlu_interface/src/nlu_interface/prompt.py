from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Tuple

@dataclass
class IncontextExample:
    def __init__(self, example_input: str, example_output: str):
        example_input: str
        example_output: str

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.example_input + "\n" + self.example_output

class Prompt(ABC):
    @abstractmethod
    def validate(self, num_incontext_examples: int):
        raise NotImplementedError("validate() not implemented.")

    @abstractmethod
    def render(self) -> str:
        """Return the prompt as a string"""
        raise NotImplementedError("render() not implemented.")

    @abstractmethod
    def to_openai(self, num_incontext_examples) -> Tuple[str, str]:
        """Return the prompt as expected for OpenAI responses API"""
        raise NotImplementedError("to_openai() not implemented.")

class DefaultPrompt(Prompt):
    def __init__(
        self,
        system: str = None,
        incontext_examples_preamble: str = None,
        incontext_examples: List[IncontextExample] = [],
        instruction_preamble: str = None,
        instruction: str = None,
        response_format: str = None,
    ):
        self.system = system
        self.incontext_examples_preamble = incontext_examples_preamble
        self.incontext_examples = incontext_examples
        self.instruction_preamble = instruction_preamble
        self.instruction = instruction
        self.response_format = response_format

    def validate(self, num_incontext_examples: int):
        if num_incontext_examples > len(self.incontext_examples):
            raise ValueError(
                f"{num_incontext_examples} requested; not enough examples provided ({len(self.incontext_examples)})."
            )

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
        if self.instruction:
            instruction_string += self.instruction
        ret = (
            f"System: {system_string}"
            f"User: {incontext_examples_string}\n{instruction_string}"
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
        if self.instruction:
            instruction_string += self.instruction
        user_string = incontext_examples_string + instruction_string
        return system_string, user_string
