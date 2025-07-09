# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Any, Literal, cast

from openai import NOT_GIVEN, APIConnectionError, BadRequestError
from openai.types import ImageModel
from sinapsis_core.data_containers.data_packet import DataContainer, ImagePacket
from sinapsis_core.template_base.base_models import (
    OutputTypes,
    TemplateAttributes,
    TemplateAttributeType,
    UIPropertiesMetadata,
)
from sinapsis_core.template_base.dynamic_template import WrapperEntryConfig
from sinapsis_core.template_base.dynamic_template_factory import make_dynamic_template
from sinapsis_core.template_base.template import Template
from sinapsis_core.utils.env_var_keys import SINAPSIS_BUILD_DOCS

from sinapsis_openai.helpers.openai_env_var_keys import OpenAIEnvVars
from sinapsis_openai.helpers.openai_keys import OpenAIKeys
from sinapsis_openai.templates.openai_base import OpenAIBase, OpenAICreateType


class OpenAIImageCreation(OpenAIBase):
    """
    Template that wraps the `OpenAI.images.generate` API endpoint to
    perform image generation tasks. This template excludes the `prompt`,
    `model`, and `response_format` attributes from the API caller and are
    set through static template attributes. In this way, the `prompt` is received
    as input (via the content of a packet) and the `model` and `response_format` are defined in
    the `AttributesBaseModel`.

    Usage example:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: OpenAIImageCreationWrapper
      class_name: OpenAIImageCreationWrapper
      template_input: InputTemplate
      attributes:
        model: 'dall-e-2'
        response_format: url
        openai_init:
          api_key: openai_api_key
          max_retries: 2
        generate:
          n: 1

    """

    WrapperEntry = WrapperEntryConfig(
        wrapped_object=OpenAIBase.CLIENT,
        template_name_suffix="ImageCreationWrapper",
        additional_callables_to_inspect=[
            WrapperEntryConfig(
                wrapped_object=OpenAIBase.CLIENT(api_key=OpenAIEnvVars.OPENAI_API_KEY.value).images.generate,
                exclude_method_attributes=[OpenAIKeys.prompt, OpenAIKeys.model, OpenAIKeys.response_format],
            )
        ],
    )
    UIProperties = UIPropertiesMetadata(category="OpenAI", output_type=OutputTypes.IMAGE)

    class AttributesBaseModel(TemplateAttributes):
        """Attributes for the OpenAIImageCreation.

        Attributes:
            model (ImageModel): The image model to use for
            generating images.
            response_format (Literal["url", "b64_json"]):
                Specifies the format of the API response (default is "url").
        """

        model: ImageModel
        response_format: Literal["url", "b64_json"] = "url"

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self.create = self.openai.images.generate

    def parse_results(self, responses: list | dict | Any, container: DataContainer) -> DataContainer:
        """
        For each of the responses, creates an ImagePacket to be
        appended to the DataContainer

        Args:
            responses (list[str]): List of URL from the OpenAI responses.
            container (DataContainer): The original data container.

        Returns:
            DataContainer: The modified container with appended ImagePackets.
        """
        for response in responses:
            container.images.append(
                ImagePacket(
                    content=response,
                    source=self.instance_name,
                )
            )
        return container

    def return_create_response(self, content: str | list | bytes) -> OpenAICreateType:
        """
        Sends an image generation request to the API endpoint
        using the provided prompt.

        Args:
            content (str | list | bytes): The prompt extracted from the packet.

        Returns:
            OpenAICreateType: The raw response from the OpenAI image generation API.
        """
        content = cast(str, content)
        return self.create(
            prompt=content,
            model=self.attributes.model,
            response_format=self.attributes.response_format,
            **self.attributes.generate.model_dump(),
        )


class OpenAIImageEdition(OpenAIImageCreation):
    """
    Template that wraps the `OpenAI.images.edit` API endpoint to perform
    image edition tasks.
    This template excludes the `image`, `mask`, `response_format`, `model`,
    and `prompt` attributes from the API caller.
    Instead, the image is provided through the attribute `path_to_image`
    and an optional mask through `path_to_mask` in the `AttributesBaseModel`.
    Additionally, the original image must be in `png` format,
    and, if a mask is provided, it must have the same dimensions as the original
    image.
    The PACKAET_TYPE_NAME indicates that this template processes packets stored in the
    text field of the DataContainer as the prompt

    Usage example:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: OpenAIImageEditionWrapper
      class_name: OpenAIImageEditionWrapper
      template_input: InputTemplate
      attributes:
        model: dall-e-2
        response_format: url
        path_to_mask' '/path/to/mask/image'
        path_to_image: '/path/to/image/to/be/edited'
        openai_init:
          api_key: openai_api_key
          max_retries: 2
        edit:
          n: 1
    """

    class AttributesBaseModel(OpenAIImageCreation.AttributesBaseModel):
        """
        Configuration attributes for the OpenAIImageEdition template.

        Attributes:
            model (Literal["dall-e-2", "gpt-image-1"]): The fixed image model used for
            image edition tasks.
            path_to_mask (str | None): Optional file path to a mask image.
            If provided, the mask must have the same dimensions as the
            original image.
            path_to_image (str): The file path of the original image to
            be edited. The image is expected to be in `png` format.
        """

        model: Literal["dall-e-2", "gpt-image-1"]
        path_to_mask: str | None = None
        path_to_image: str

    WrapperEntry = WrapperEntryConfig(
        wrapped_object=OpenAIBase.CLIENT,
        template_name_suffix="ImageEditionWrapper",
        additional_callables_to_inspect=[
            WrapperEntryConfig(
                wrapped_object=OpenAIBase.CLIENT(api_key=OpenAIEnvVars.OPENAI_API_KEY.value).images.edit,
                exclude_method_attributes=[
                    OpenAIKeys.image,
                    OpenAIKeys.mask,
                    OpenAIKeys.response_format,
                    OpenAIKeys.model,
                    OpenAIKeys.prompt,
                ],
            )
        ],
    )
    PACKET_TYPE_NAME = "texts"

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self.create = self.openai.images.edit

    def return_create_response(self, content: str | list | bytes) -> OpenAICreateType:
        """
        Sends an image editing request to the API endpoint with the
        provided prompt.
        The image is taken from `path_to_image` and an optional
         mask from `path_to_mask` (if provided).
        These values are supplied exclusively via template attributes,
        which is why the corresponding API parameters are excluded.
        It is also expected that the image is in `png` format and,
        if a mask is used, that the mask matches the size of the
        original image.

        Args:
            content (str | list | bytes): The prompt for image editing,
            extracted from the packet.

        Returns:
            OpenAICreateType: The raw response from the OpenAI image editing API.
        """
        mask = Path(self.attributes.path_to_mask) if self.attributes.path_to_mask else NOT_GIVEN

        try:
            return self.create(
                image=Path(self.attributes.path_to_image),
                mask=mask,
                prompt=content,
                model=self.attributes.model,
                response_format=self.attributes.response_format,
                **self.attributes.edit.model_dump(),
            )
        except (APIConnectionError, BadRequestError):
            return self.create(
                image=Path(self.attributes.path_to_image),
                mask=mask,
                prompt=content,
                model=self.attributes.model,
                **self.attributes.edit.model_dump(),
            )


def __getattr__(name: str) -> Template:
    """
    Only create a template if it's imported, this avoids creating all
    the base models for all templates and potential import errors due
    to not available packages.
    """
    if name in OpenAIImageCreation.WrapperEntry.module_att_names:
        return make_dynamic_template(name, OpenAIImageCreation)
    if name in OpenAIImageEdition.WrapperEntry.module_att_names:
        return make_dynamic_template(name, OpenAIImageEdition)


__all__ = OpenAIImageCreation.WrapperEntry.module_att_names + OpenAIImageEdition.WrapperEntry.module_att_names

if SINAPSIS_BUILD_DOCS:
    dynamic_templates = [__getattr__(template_name) for template_name in __all__]
    for template in dynamic_templates:
        globals()[template.__name__] = template
        del template
