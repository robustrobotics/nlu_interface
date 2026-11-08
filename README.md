# nlu_interface
This is designed to be a general library for managing a way to interface with different LLM providers (e.g., OpenAI, Anthropoc, local Ollama, etc.) and to organize resources/tools that go into different projects (e.g., prompts, configs, etc.).

## RViz manipulation approval panel (`nlu_interface_rviz`)

The panel shows the candidate images a Spot executor sends on
`/<robot>/spot_executor_node/manipulation_request` and answers on
`/<robot>/spot_executor_node/pick_confirmation`:

- **Publish**: approve the grasp at the selected pixel (red crosshair).
- **Walk To**: do not grasp yet. The robot walks toward the selected ground/object
  pixel, takes new images and sends a fresh request, which replaces the panel
  contents. Use **Select Pixel** to choose the point first if the detection is
  not where you want the robot to go.
- **Reject**: abandon the grasp attempt.

Heading control for Walk To is a feature flag, off by default. Set the rviz node
parameter `manipulation_walk_to_heading_selection` to `true` to show
**Select Stand Point (heading)**: click the ground pixel Spot should stand on and
it faces the selected target from there. The stand point must be on the same
image as the target; otherwise the default standoff is used. Robots are
configured with the `robot_ids` parameter.

## Simple Example (OpenAI)
Note that the example config expects an API key to be stored in the environment variable $OPENAI\_API\_KEY.

To run a simple example for OpenAI, do the following:
```bash
cd nlu_interface/examples/
python test.py --config ../config/openai_config.yaml --prompt ../src/nlu_interface/resources/prompt.yaml
```
