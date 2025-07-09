# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import Any, cast

from openai import _legacy_response
from openai.types import AudioModel
from openai.types.audio import SpeechModel
from sinapsis_core.data_containers.data_packet import AudioPacket, DataContainer, Packet
from sinapsis_core.template_base.base_models import (
    OutputTypes,
    TemplateAttributes,
    TemplateAttributeType,
    UIPropertiesMetadata,
)
from sinapsis_core.template_base.dynamic_template import WrapperEntryConfig
from sinapsis_core.template_base.dynamic_template_factory import make_dynamic_template
from sinapsis_core.template_base.template import Template
from sinapsis_core.utils.env_var_keys import SINAPSIS_BUILD_DOCS, SINAPSIS_CACHE_DIR

from sinapsis_openai.helpers.openai_env_var_keys import OpenAIEnvVars
from sinapsis_openai.helpers.openai_keys import OpenAIKeys
from sinapsis_openai.templates.openai_base import OpenAICreateType
from sinapsis_openai.templates.openai_chat import OpenAIChatCompletion


class OpenAIAudioTranscription(OpenAIChatCompletion):
    """
    Template that wraps the `OpenAI.audio.transcriptions.create`
    API endpoint to perform audio transcription tasks.
    This template excludes the `file` attribute
    (expected to be provided via the DataContainer packet `source`) and
    the `model` attribute (which is set in the template `AttributesBaseModel`)
    from the API caller. This ensures that the file path is dynamically
    extracted from the data packet and that the audio model is configured
    exclusively through the template attributes.
    PACKET_TYPE_NAME indicates that this template process AudioPackets
    stored in the audio field of DataContainer

    Usage example:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: OpenAIAudioTranscriptionWrapper
      class_name: OpenAIAudioTranscriptionWrapper
      template_input: InputTemplate
      attributes:
        model: 'whisper-1'
        openai_init:
          api_key: openai_api_key
          max_retries: 2
        create:
          language: en

    """

    WrapperEntry = WrapperEntryConfig(
        wrapped_object=OpenAIChatCompletion.CLIENT,
        template_name_suffix="AudioTranscriptionWrapper",
        additional_callables_to_inspect=[
            WrapperEntryConfig(
                wrapped_object=OpenAIChatCompletion.CLIENT(
                    api_key=OpenAIEnvVars.OPENAI_API_KEY.value
                ).audio.transcriptions.create,
                exclude_method_attributes=[OpenAIKeys.file, OpenAIKeys.model],
            )
        ],
    )

    class AttributesBaseModel(TemplateAttributes):
        """
        Attributes for OpenAIAudioTranscription.
        Attributes:
            model (AudioModel): The model to use for audio transcription.
        """

        model: AudioModel

    PACKET_TYPE_NAME: str = "audios"
    UIProperties = UIPropertiesMetadata(category="LlamaIndex", output_type=OutputTypes.TEXT)

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self.create = self.openai.audio.transcriptions.create

    @staticmethod
    def unpack_packet_content(packet: Packet) -> list | str:
        """
        Extracts the source attribute from the given Packet instance which is expected to be a file path.

        Args:
            packet (Packet): A data packet that is expected to contain a source path to audio file location.

        Returns:
            str: The extracted file path from the packet's source.
        """
        return packet.source

    def return_create_response(self, content: str | list | bytes) -> OpenAICreateType:
        """
        Sends a transcription request to the OpenAI.audio.transcriptions.create endpoint by creating a Path object
        from the provided content. The request is constructed using the audio model defined in the template attributes
        and any additional parameters specified in the 'create' attribute.

        Args:
            content (str | list | bytes): The file path (as string) extracted from the input packet to be transcribed.

        Returns:
            OpenAICreateType: The raw response from the OpenAI transcription API.
        """
        content = cast(str, content)
        return self.create(
            file=Path(content),
            model=self.attributes.model,
            **self.attributes.create.model_dump(),
        )


class OpenAIAudioTranslation(OpenAIAudioTranscription):
    """Template that wraps the `OpenAI.audio.translations.create` API endpoint
    to perform audio translation tasks.
    This template excludes the `file` attribute (expected to be provided via the
    DataContainers packet `source`)
    and the `model` attribute (which is configured exclusively in
    the templates `AttributesBaseModel`) from the API caller.
    This design ensures that the file path is dynamically extracted
    from the data packet and that the audio model comes solely from
    the template configuration.

    Usage example:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: OpenAIAudioTranslationWrapper
      class_name: OpenAIAudioTranslationWrapper
      template_input: InputTemplate
      attributes:
        model: 'whisper-1'
        openai_init:
          api_key: openai_api_key
          max_retries: 2
        create:
          language: en

    """

    AttributesBaseModel = OpenAIAudioTranscription.AttributesBaseModel

    WrapperEntry = WrapperEntryConfig(
        wrapped_object=OpenAIChatCompletion.CLIENT,
        template_name_suffix="AudioTranslationWrapper",
        additional_callables_to_inspect=[
            WrapperEntryConfig(
                wrapped_object=OpenAIChatCompletion.CLIENT(
                    api_key=OpenAIEnvVars.OPENAI_API_KEY.value
                ).audio.translations.create,
                exclude_method_attributes=[OpenAIKeys.file, OpenAIKeys.model],
            )
        ],
    )
    PACKET_TYPE_NAME = "audios"

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self.create = self.openai.audio.translations.create


class OpenAIAudioCreation(OpenAIAudioTranscription):
    """
        Template that wraps the `OpenAI.audio.speech.create` API endpoint to
        perform audio creation tasks.
        This template excludes the `input` attribute (expected to be provided
        via the DataContainer packet `content`)
        and the `model` attribute (which is set in the template `AttributesBaseModel`)
        from the API caller.
        This ensures that the text input is dynamically extracted from the packet
        and that the speech model is passed exclusively via the template attributes.
        The PACKET_TYPE_NAME indicates that the template processes texts packets
        in the DataContainer

        Usage example:

        agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
    - template_name: OpenAIAudioCreationWrapper
      class_name: OpenAIAudioCreationWrapper
      template_input: InputTemplate
      attributes:
        model: 'tts-1'
        output_dir: openai/openai_audio.mp4
        openai_init:
          api_key: openai_api_key
          max_retries: 2
        create:
          voice: 'alloy'
    """

    WrapperEntry = WrapperEntryConfig(
        wrapped_object=OpenAIChatCompletion.CLIENT,
        template_name_suffix="AudioCreationWrapper",
        additional_callables_to_inspect=[
            WrapperEntryConfig(
                wrapped_object=OpenAIChatCompletion.CLIENT(
                    api_key=OpenAIEnvVars.OPENAI_API_KEY.value
                ).audio.speech.create,
                exclude_method_attributes=[OpenAIKeys.input, OpenAIKeys.model],
            )
        ],
    )

    PACKET_TYPE_NAME = "texts"
    UIProperties = UIPropertiesMetadata(category="OpenAI", output_type=OutputTypes.AUDIO)

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self.create = self.openai.audio.speech.create

    class AttributesBaseModel(TemplateAttributes):
        """
        Attributes for the OpenAIAudioCreation template.

        Attributes:
            model (SpeechModel): Specifies the speech model to be used for generating audio.
            output_dir (str | None): Relative path where the binary audio output file will
            be saved (default is "openai/openai_audio.mp4").

        """

        model: SpeechModel
        output_dir: str | None = "openai/openai_audio.mp4"

    @staticmethod
    def unpack_packet_content(packet: Packet) -> list | str:
        """
        Retrieves the textual input from the given Packet instance which serves as the prompt for audio creation.

        Args:
            packet (Packet): A data packet containing the text input.

        Returns:
            list | str: The text content from the packet that will be used in the API request.
        """
        return packet.content

    def return_create_response(self, content: str | list | bytes) -> OpenAICreateType:
        """
        Sends an audio creation request to the OpenAI.audio.speech.create endpoint using a textual prompt retrieved
        from the input. The API call is constructed with the SpeechModel defined in the template attributes and any
        extra parameters specified within the configuration.

        Args:
            content (str | list | bytes): The text prompt extracted from the Packet to be converted into audio.

        Returns:
            OpenAICreateType: The raw API response containing the binary audio data or related information.
        """
        return self.create(
            input=content,
            model=self.attributes.model,
            **self.attributes.create.model_dump(),
        )

    def generate_response_from_client(self, packet: Packet) -> OpenAICreateType:
        """
        Executes the audio creation API call by extracting the prompt from the packet,
        invoking the wrapped API endpoint, and then streams the binary response to a file. The output file is saved in a
        directory specified by the template attributes under output_dir (relative to the SINAPSIS_CACHE_DIR).
        If the API call fails, an error is logged and None is returned.
        Args:
            packet (Packet): Packet to extract the input from.

        Returns:
            Any: The processed results.

        Raises:
            APIConnectionError: If the Connect API call fails.
            BadRequestError: In the event of an invalid API request.
        """
        results = super().generate_response_from_client(packet)
        results = cast(_legacy_response.HttpxBinaryResponseContent, results)
        results.stream_to_file(os.path.join(SINAPSIS_CACHE_DIR, self.attributes.output_dir))
        return os.path.join(SINAPSIS_CACHE_DIR, self.attributes.output_dir)

    def parse_results(self, responses: str | dict | Any, container: DataContainer) -> DataContainer:
        """
        Processes the responses from the audio creation API call by converting each resulting file
        path into an AudioPacket. Each AudioPacket is tagged with the file path as both its content
        and source, and then appended to the container's list of audio packets.

        Args:
            responses (list[str]): A list of response texts from OpenAI.
            container (DataContainer): The container where the results will be stored.

        Returns:
            DataContainer: The updated container with parsed results.
        """
        _ = self
        for response in responses:
            audio_packet = AudioPacket(content=response, source=response)
            container.audios.append(audio_packet)
        return container


def __getattr__(name: str) -> Template:
    """
    Only create a template if it's imported, this avoids creating all the base models for all templates
    and potential import errors due to not available packages.
    """
    if name in OpenAIAudioTranslation.WrapperEntry.module_att_names:
        return make_dynamic_template(name, OpenAIAudioTranslation)
    if name in OpenAIAudioTranscription.WrapperEntry.module_att_names:
        return make_dynamic_template(name, OpenAIAudioTranscription)
    if name in OpenAIAudioCreation.WrapperEntry.module_att_names:
        return make_dynamic_template(name, OpenAIAudioCreation)


__all__ = (
    OpenAIAudioTranslation.WrapperEntry.module_att_names
    + OpenAIAudioTranscription.WrapperEntry.module_att_names
    + OpenAIAudioCreation.WrapperEntry.module_att_names
)

if SINAPSIS_BUILD_DOCS:
    dynamic_templates = [__getattr__(template_name) for template_name in __all__]
    for template in dynamic_templates:
        globals()[template.__name__] = template
        del template
