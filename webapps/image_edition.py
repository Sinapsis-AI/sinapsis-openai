# -*- coding: utf-8 -*-
import base64
import io
import os
import tempfile

import gradio as gr
import numpy as np
import PIL.Image
from sinapsis.webapp.agent_gradio_helper import add_logo_and_title, css_header
from sinapsis_core.agent.agent import Agent
from sinapsis_core.cli.run_agent_from_config import generic_agent_builder
from sinapsis_core.data_containers.data_packet import DataContainer
from sinapsis_core.utils.env_var_keys import AGENT_CONFIG_PATH, GRADIO_SHARE_APP

DEFAULT_CONFIG = "src/sinapsis_openai/configs/config_image_edition.yaml"
CONFIG_FILE = AGENT_CONFIG_PATH or DEFAULT_CONFIG
TEXT_INPUT_TEMPLATE_NAME = "TextInput"
IMAGE_EDITION_TEMPLATE_NAME = "OpenAIImageEdition"


class ImageEditionApp:
    def __init__(self, config_file: str, text_input_template: str, image_edition_template: str) -> None:
        self.config_file = config_file
        self.text_input_template = text_input_template
        self.image_edition_template = image_edition_template
        self.temp_dir = tempfile.mkdtemp()

    def init_agent(self) -> tuple[Agent, bool]:
        agent = generic_agent_builder(self.config_file)
        return agent, True

    def _to_pil(self, img: np.ndarray | PIL.Image.Image) -> PIL.Image.Image:
        if isinstance(img, np.ndarray):
            return PIL.Image.fromarray(img)
        return img

    def _create_openai_mask(self, edited_img: PIL.Image.Image, original_img: PIL.Image.Image) -> PIL.Image.Image:
        """
        Creates an OpenAI-compatible mask image from the edited and original images.

        Args:
            edited_img (PIL.Image.Image): The edited image with erased (transparent) areas.
            original_img (PIL.Image.Image): The original image.

        Returns:
            PIL.Image.Image: The mask image in RGBA format.
        """
        edited_img = self._to_pil(edited_img)
        original_img = self._to_pil(original_img)
        if edited_img.mode != "RGBA":
            edited_img = edited_img.convert("RGBA")
        if original_img.mode != "RGBA":
            original_img = original_img.convert("RGBA")
        alpha = edited_img.split()[-1]
        mask_rgba = original_img.copy()
        mask_rgba.putalpha(alpha)
        return mask_rgba

    def _get_container_id(self, container: DataContainer) -> str:
        """
        Retrieves the unique container ID as a string.

        Args:
            container (DataContainer): The data container.

        Returns:
            str: The container ID.
        """
        return str(container.container_id)

    def _get_tmp_paths(self, container_id: str) -> tuple[str, str]:
        """
        Generates temporary file paths for the image and mask.

        Args:
            container_id (str): The unique container ID.

        Returns:
            tuple[str, str]: Paths for the image and mask files.
        """
        image_path = os.path.join(self.temp_dir, f"image_{container_id}.png")
        mask_path = os.path.join(self.temp_dir, f"mask_{container_id}.png")
        return image_path, mask_path

    def _save_image(self, img: PIL.Image.Image, path: str) -> None:
        """
        Saves a PIL image to the specified path.

        Args:
            img (PIL.Image.Image): The image to save.
            path (str): The file path.
        """
        img.save(path)

    def save_images_to_tmp(self, editor_output: dict, container: DataContainer) -> tuple[str, str]:
        """
        Saves the edited image and mask to temporary files.

        Args:
            editor_output (dict): Output from the Gradio ImageEditor.
            container (DataContainer): The data container.

        Returns:
            tuple[str, str]: Paths to the saved image and mask files.
        """
        container_id = self._get_container_id(container)
        image_path, mask_path = self._get_tmp_paths(container_id)
        edited_img = self._to_pil(editor_output["background"])
        original_img = edited_img.copy()
        self._save_image(original_img, image_path)
        mask_img = self._create_openai_mask(edited_img, original_img)
        self._save_image(mask_img, mask_path)
        return image_path, mask_path

    def _decode_b64_to_image(self, b64_json: str) -> PIL.Image.Image:
        """
        Decodes a base64-encoded image string to a PIL.Image.Image.

        Args:
            b64_json (str): The base64-encoded image string.

        Returns:
            PIL.Image.Image: The decoded image.
        """
        image_bytes = base64.b64decode(b64_json)
        return PIL.Image.open(io.BytesIO(image_bytes))

    def _update_agent_attributes(self, agent: Agent, prompt: str, image_path: str, mask_path: str) -> None:
        """
        Updates the agent's template attributes with the prompt, image path, and mask path.

        Args:
            agent (Agent): The agent instance.
            prompt (str): The user prompt.
            image_path (str): Path to the image file.
            mask_path (str): Path to the mask file.
        """
        agent.update_template_attribute(self.text_input_template, "text", prompt)
        agent.update_template_attribute(self.image_edition_template, "path_to_image", image_path)
        agent.update_template_attribute(self.image_edition_template, "path_to_mask", mask_path)

    def _get_result_image(self, output_container: DataContainer) -> PIL.Image.Image | None:
        """
        Extracts and decodes the resulting image from the output container

        Args:
            output_container (DataContainer): The output data container

        Returns:
            PIL.Image.Image | None: The decoded image, or None if decoding fails
        """
        if output_container.images:
            b64_json = output_container.images[-1].content
            try:
                return self._decode_b64_to_image(b64_json)
            except ValueError:
                return None
        return None

    def edit_image(
        self, initialized: bool, agent: Agent, editor_output: dict, prompt: str
    ) -> tuple[object, str | None]:
        """
        Handles the image editing pipeline: saves images, updates agent, runs agent, and decodes result

        Args:
            initialized (bool): Whether the agent is initialized.
            agent (Agent): The agent instance.
            editor_output (dict): Output from the Gradio ImageEditor.
            prompt (str): The user prompt.

        Returns:
            tuple[object, str]: The resulting image (or None) and a status message.
        """
        if not initialized:
            return None, "#### Model not ready! Please wait..."
        if not editor_output or "background" not in editor_output or editor_output["background"] is None:
            gr.Warning("Please upload an image and erase the area to edit.")
            return None, "Please upload an image and erase the area to edit."
        if not prompt:
            gr.Warning("Please enter a valid prompt.")
            return None, "Please enter a valid prompt."
        container = DataContainer()
        image_path, mask_path = self.save_images_to_tmp(editor_output, container)
        self._update_agent_attributes(agent, prompt, image_path, mask_path)
        output_container = agent(container)
        img = self._get_result_image(output_container)
        if img is not None:
            return img, None
        return None, "No image generated."

    def inner_functionality(self, interface: gr.Blocks) -> None:
        """
        Defines the Gradio UI layout and the pipeline for image editing

        Args:
            interface (gr.Blocks): The Gradio Blocks interface.
        """
        agent_state = gr.State()
        initialized_state = gr.State(False)
        interface.load(self.init_agent, outputs=[agent_state, initialized_state])
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    value="Add a hat to the image",
                    placeholder="Describe what you want to add or modify in the image",
                    label="Prompt",
                )
                image_editor = gr.ImageEditor(
                    type="pil",
                    label="Upload image and erase area to edit",
                    canvas_size=(512, 512),
                    brush=gr.Brush(colors=["#fff"], default_color="#fff"),
                    layers=False,
                )
                edit_btn = gr.Button("Edit Image")
                status_msg = gr.Markdown("#### Initializing model...")
            with gr.Column(scale=1):
                edited_image = gr.Image(label="Edited Image")
        edit_btn.click(
            self.edit_image,
            inputs=[initialized_state, agent_state, image_editor, prompt],
            outputs=[edited_image, status_msg],
        )

    def __call__(self) -> gr.Blocks:
        with gr.Blocks(css=css_header()) as demo:
            add_logo_and_title("Sinapsis OpenAI Image Edition")
            self.inner_functionality(demo)
        return demo


if __name__ == "__main__":
    app = ImageEditionApp(CONFIG_FILE, TEXT_INPUT_TEMPLATE_NAME, IMAGE_EDITION_TEMPLATE_NAME)
    app().launch(share=GRADIO_SHARE_APP)
