# nlu_interface
This is designed to be a general library for managing a way to interface with different LLM providers (e.g., OpenAI, Anthropoc, local Ollama, etc.) and to organize resources/tools that go into different projects (e.g., prompts, configs, etc.).

## Simple Example (OpenAI)
Note that the example config expects an API key to be stored in the environment variable $OPENAI\_API\_KEY.

To run a simple example for OpenAI, do the following:
```bash
cd nlu_interface/examples/
python test.py --config ../config/openai_config.yaml --prompt ../src/nlu_interface/resources/prompt.yaml
```
