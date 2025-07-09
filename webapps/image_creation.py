# -*- coding: utf-8 -*-
from typing import Any

import gradio as gr
from sinapsis.webapp.agent_gradio_helper import add_logo_and_title, css_header
from sinapsis_core.agent.agent import Agent
from sinapsis_core.cli.run_agent_from_config import generic_agent_builder
from sinapsis_core.data_containers.data_packet import DataContainer
from sinapsis_core.utils.env_var_keys import AGENT_CONFIG_PATH, GRADIO_SHARE_APP

DEFAULT_CONFIG = "src/sinapsis_openai/configs/config_image_creation.yaml"
CONFIG_FILE = AGENT_CONFIG_PATH or DEFAULT_CONFIG
TEXT_INPUT_TEMPLATE_NAME = "TextInput"


class ImageGenerationApp:
    def __init__(self, config_file: str, text_input_template: str) -> None:
        self.config_file = config_file
        self.text_input_template = text_input_template

    def init_agent(self) -> tuple[Agent, bool]:
        agent = generic_agent_builder(self.config_file)
        return agent, True

    def generate_image(self, initialized: bool, agent: Agent, prompt: str) -> tuple[Any, str | None]:
        """
        Handles the image generation pipeline: updates agent, runs agent, and extracts result.

        Args:
            initialized (bool): Whether the agent is initialized.
            agent (Agent): The agent instance.
            prompt (str): The user prompt.

        Returns:
            tuple[Any, str | None]: The resulting image (or None) and a status message.
        """
        if not initialized:
            return None, "#### Model not ready! Please wait..."
        if not prompt:
            gr.Warning("Please enter a valid prompt.")
            return None, "Please enter a valid prompt."

        agent.update_template_attribute(self.text_input_template, "text", prompt)
        container = DataContainer()
        output_container = agent(container)
        if hasattr(output_container, "images") and output_container.images:
            img = output_container.images[-1].content
            return img, None
        return None, "No image generated."

    def inner_functionality(self, interface: gr.Blocks) -> None:
        """
        Defines the Gradio UI layout and connects UI events to backend logic.

        Args:
            interface (gr.Blocks): The Gradio Blocks interface.
        """
        agent_state = gr.State()
        initialized_state = gr.State(False)
        interface.load(self.init_agent, outputs=[agent_state, initialized_state])

        prompt = gr.Textbox(
            value="cat with boots",
            placeholder="Describe the image you want to generate",
        )
        generate_btn = gr.Button("Generate")
        status_msg = gr.Markdown("#### Initializing model...")

        image_output = gr.Image(label="Generated image", visible=True)

        generate_btn.click(
            self.generate_image,
            inputs=[initialized_state, agent_state, prompt],
            outputs=[image_output, status_msg],
        )

    def __call__(self) -> gr.Blocks:
        with gr.Blocks(css=css_header()) as demo:
            add_logo_and_title("Sinapsis OpenAI Image Generation")
            self.inner_functionality(demo)
        return demo


if __name__ == "__main__":
    app = ImageGenerationApp(CONFIG_FILE, TEXT_INPUT_TEMPLATE_NAME)
    app().launch(share=GRADIO_SHARE_APP)
