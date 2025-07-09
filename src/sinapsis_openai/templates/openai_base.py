# -*- coding: utf-8 -*-
"""OpenAI.chat template module"""

from typing import Any, Union

from openai import APIConnectionError, BadRequestError, OpenAI
from openai.types import ImagesResponse
from openai.types.audio import Transcription, Translation
from openai.types.chat import ChatCompletion
from openai.types.embedding import Embedding
from openai.types.image import Image
from sinapsis_core.data_containers.data_packet import DataContainer, Packet
from sinapsis_core.template_base.base_models import OutputTypes, TemplateAttributeType, UIPropertiesMetadata
from sinapsis_core.template_base.dynamic_template import (
    BaseDynamicWrapperTemplate,
    WrapperEntryConfig,
)

from sinapsis_openai.helpers.openai_env_var_keys import OpenAIEnvVars

OpenAICreateType = Union[ChatCompletion, Transcription, Image, Embedding, Translation, str, Any, ImagesResponse]


class OpenAIBase(BaseDynamicWrapperTemplate):
    """
    OpenAIBase is a base class for creating a wrapper around the OpenAI API.
    It uses the `BaseDynamicWrapperTemplate` as its base class and
    initializes an instance of the `OpenAI`
    client when it is instantiated.


    Attributes:
        CLIENT (OpenAI): An instance of the OpenAI client.
        PACKET_TYPE_NAME (str): The name of the packet type for text data.
        WrapperEntry (WrapperEntryConfig): A configuration for the wrapped object.

    """

    CLIENT = OpenAI
    PACKET_TYPE_NAME = "texts"
    UIProperties = UIPropertiesMetadata(category="OpenAI", output_type=OutputTypes.TEXT)

    WrapperEntry = WrapperEntryConfig(wrapped_object=CLIENT)

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self.api_key = OpenAIEnvVars.OPENAI_API_KEY.value or self.attributes.openai_init.api_key
        self.openai = self.CLIENT(api_key=self.api_key, **self.attributes.openai_init.model_dump(exclude="api_key"))

    @staticmethod
    def unpack_packet_content(packet: Packet) -> list | str:
        """
        Unpacks the raw content from a given Packet instance.
        This method is intended to extract the core message content which will
        then be sent to the OpenAI API. The default implementation simply returns
        the Packet's content, although subclasses may override this behavior if a
        more complex transformation is required.

        Args:
            packet (Packet): Packet to unpack.

        Returns:
            Any: The unpacked content.
        """
        return packet.content

    def get_results(self, results: OpenAICreateType) -> str | dict | Any:
        """
        Processes and extracts relevant data from the OpenAI API response.
        Depending on the type of the response, appropriate fields are extracted.
        In the case of ChatCompletion, the response to the message content is
        returned; while for AudioTranscription and/or AudioTranslation, the
        updated Audio is returned.

        Args:
            results (OpenAICreateType): The raw response object returned by an
            OpenAI API call.

        Returns:
            str | dict | Any: Processed response data
            (text for chat/audio, list for images, etc.).
        """
        if isinstance(results, ChatCompletion):
            return results.choices[0].message.content if results.choices else ""

        elif isinstance(results, Transcription | Translation):
            return results.text

        elif isinstance(results, ImagesResponse):
            if results.data:
                if self.attributes.response_format == "url":
                    return results.data[0].url
                else:
                    return results.data[0].b64_json
            return ""
        elif isinstance(results, Embedding):
            return results.embedding

        else:
            return results

    def return_create_response(self, content: list | str | bytes) -> OpenAICreateType:
        """
        Constructs and returns the API response by invoking the wrapped
        creation callable with the provided content.
        This method must be implemented by specialized templates in order
        to correctly map input parameters to
        the OpenAI API request parameters as required by the particular endpoint.

        Args:
            content (list | str | bytes): Content to be passed as input to the API

        Returns:
            OpenAICreateType: The response from the OpenAI API.
        """

    def generate_response_from_client(self, packet: Packet) -> OpenAICreateType | None:
        """
        Extracts content from a given packet, sends the content to the
        OpenAI API using the wrapped callable, and then returns the
        processed result. This method catches API-specific connection
        errors or bad requests, logs a warning and returns None
        to signal the failure.

        Args:
            packet (Packet): Packet in the container to be passed to the client.

        Returns:
            OpenAICreateType | None: Processed results from the API call
            if successful. In case of an APIConnectionError or BadRequestError,
            the method logs the error with warning and returns None.

        Raises:
            APIConnectionError: If there is a connection issue contacting the API.
            BadRequestError: If the API returns an error due to an invalid request.
        """
        message = self.unpack_packet_content(packet)
        try:
            results = self.return_create_response(message)

        except (APIConnectionError, BadRequestError) as err:
            self.logger.warning(f"Error processing request: {err}")
            results = None

        return results

    def parse_results(self, responses: str | dict | Any, container: DataContainer) -> DataContainer:
        """
        Parses the responses obtained from the OpenAI API call and
        appropriately packages them into new data packets that are
        appended to the provided DataContainer.
        The method is responsible for converting raw API responses
        into a format suitable for DataContainers.
        Args:
            responses (str | dict | Any): Result from the call to
            the OpenAI API
            container (DataContainer): DataContainer where the result
            will be inserted as a Packet
        Returns:
            DataContainer: The updated data container with the newly appended packets
        """

    def execute(self, container: DataContainer) -> DataContainer:
        """
        Executes the overall process of retrieving and processing
        OpenAI API responses. It iterates over each target packet in the DataContainer,
        sends the content of, so it can be processed by the API. Then, the response is
        gathered and processed using get_results. Finally, the processed responses are
        added to the container for downstream usage.

        Args:
            container (DataContainer): The container holding the input data packets.

        Returns:
            DataContainer: The updated container with responses from OpenAI.
        """
        data_packet = getattr(container, self.PACKET_TYPE_NAME)
        responses_from_openai: list[str | dict | Any] = []
        for packet in data_packet:
            response_output = self.generate_response_from_client(packet)
            responses_from_openai.append(self.get_results(response_output))

        container = self.parse_results(responses_from_openai, container)

        return container
