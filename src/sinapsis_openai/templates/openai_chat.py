# -*- coding: utf-8 -*-

from typing import Any, cast

from openai.types import ChatModel
from sinapsis_core.data_containers.data_packet import DataContainer, Packet, TextPacket
from sinapsis_core.template_base.base_models import TemplateAttributes, TemplateAttributeType
from sinapsis_core.template_base.dynamic_template import WrapperEntryConfig
from sinapsis_core.template_base.dynamic_template_factory import make_dynamic_template
from sinapsis_core.template_base.template import Template
from sinapsis_core.utils.env_var_keys import SINAPSIS_BUILD_DOCS

from sinapsis_openai.helpers.openai_env_var_keys import OpenAIEnvVars
from sinapsis_openai.helpers.openai_keys import OpenAIKeys
from sinapsis_openai.templates.openai_base import OpenAIBase, OpenAICreateType


class OpenAIChatCompletion(OpenAIBase):
    """
    OpenAIChatCompletion is a specialized template that wraps the
    'OpenAI.chat.completions.create' method.
    The template is built upon the common functionality provided by OpenAIBase
    to transform input data packets into appropriately formatted message lists,
    send the chat request to the OpenAI API, and parse the resulting responses
    into new text packets. This template requires configuration of both the API
    parameters (via the openai_init attribute) and the chat-specific settings
    (for example, the ChatModel to be used).

    The class attribute 'WrapperEntry' is defined using a nested WrapperEntryConfig.
    This setup enables dynamic template generation by first wrapping the base OpenAI
    client and then, in a nested manner, wrapping the specific OpenAI API call
    (chat.completions.create). In this way, only the desired method parameters
    (excluding those such as messages, audio, and model) are exposed for configuration.

    Usage example:
    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: OpenAIChatWrapper
      class_name: OpenAIChatWrapper
      template_input: InputTemplate
      attributes:
        model: gpt-4o
        openai_init:
          api_key: openai_api_key
          max_retries: 2
        create:
          max_tokens: 1000
          top_p: 3
    """

    WrapperEntry = WrapperEntryConfig(
        wrapped_object=OpenAIBase.CLIENT,
        template_name_suffix="ChatWrapper",
        additional_callables_to_inspect=[
            WrapperEntryConfig(
                wrapped_object=OpenAIBase.CLIENT(api_key=OpenAIEnvVars.OPENAI_API_KEY.value).chat.completions.create,
                exclude_method_attributes=[OpenAIKeys.messages, OpenAIKeys.audio, OpenAIKeys.model],
            )
        ],
    )

    class AttributesBaseModel(TemplateAttributes):
        """AttributesBaseModel for OpenAIChatCompletion.
        Attributes:
            model (ChatModel): The chat model to be used for the completion.
        """

        model: ChatModel

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self.create = self.openai.chat.completions.create

    @staticmethod
    def unpack_packet_content(packet: Packet) -> list | str:
        """
        Transforms the content of the given Packet into a format required by the OpenAI chat API. The method
        wraps the packet's text content into a list containing a single dictionary that specifies the role ("user")
        and the content of the message.

        Args:
            packet (Packet): A data packet containing the input text message.

        Returns:
            list | str: A list containing a dictionary with keys "role" and "content",
            which represents the chat message.
        """
        message_list = [
            {
                "role": "user",
                "content": packet.content,
            }
        ]
        return message_list

    def return_create_response(self, content: list | str | bytes) -> OpenAICreateType:
        """
        Sends a request to the OpenAI chat completions endpoint by invoking the wrapped callable with the transformed
        message content, selected chat model, and additional parameters specified in the 'create' attribute. The method
        casts the content to a list (of messages) and issues the API call, returning the raw API response.

        Args:
            content (list | str | bytes): The processed input payload (a list of chat messages) to be sent to the API.

        Returns:
            OpenAICreateType: The raw response received from the OpenAI chat completions endpoint.
        """
        content = cast(list, content)
        return self.create(messages=content, model=self.attributes.model, **self.attributes.create.model_dump())

    def parse_results(self, responses: str | dict | Any, container: DataContainer) -> DataContainer:
        """
        Processes the list of responses from the OpenAI API by converting each response into a TextPacket, setting the
        packet's source to the instance name, and appending the packet to the provided DataContainer. This method
        ensures that the API responses are properly encapsulated for DataContainer compatibility.

        Args:
            responses (list[str]): A list of response texts from OpenAI.
            container (DataContainer): The container where the results will be stored.

        Returns:
            DataContainer: The updated container with parsed results.
        """

        for response in responses:
            text_packet = TextPacket(content=response, source=self.instance_name)
            container.texts.append(text_packet)
        return container


def __getattr__(name: str) -> Template:
    """
    Only create a template if it's imported, this avoids creating all the base models for all templates
    and potential import errors due to not available packages.
    """
    if name in OpenAIChatCompletion.WrapperEntry.module_att_names:
        return make_dynamic_template(name, OpenAIChatCompletion)


__all__ = OpenAIChatCompletion.WrapperEntry.module_att_names

if SINAPSIS_BUILD_DOCS:
    dynamic_templates = [__getattr__(template_name) for template_name in __all__]
    for template in dynamic_templates:
        globals()[template.__name__] = template
        del template
