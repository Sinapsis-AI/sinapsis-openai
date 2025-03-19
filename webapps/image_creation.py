# -*- coding: utf-8 -*-

import gradio as gr
from sinapsis.webapp.agent_gradio_helper import add_logo_and_title, css_header, init_image_inference
from sinapsis_core.utils.env_var_keys import AGENT_CONFIG_PATH, GRADIO_SHARE_APP

DEFAULT_CONFIG = "src/sinapsis_openai/configs/config_image_creation.yaml"
CONFIG_FILE = AGENT_CONFIG_PATH or DEFAULT_CONFIG


def create_demo() -> gr.Blocks:
    """Creates and returns the Gradio Blocks demo interface.

    Returns:
        gr.Blocks: The Gradio Blocks object containing the entire interface.
    """
    with gr.Blocks(title="Sinapsis OpenAI Image Generation", css=css_header()) as demo:
        add_logo_and_title("Sinapsis OpenAI Image Generation")
        init_image_inference(CONFIG_FILE, image_input=None)
    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=GRADIO_SHARE_APP)
