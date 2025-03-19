# -*- coding: utf-8 -*-

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class OpenAIKeys:
    """Keys to exclude from create or generate methods in OpenAI
    These also include methods to be called from OpenAI

    Attributes:
        messages (str): Key for chat messages.
        file (str): Key for file parameter in audio tasks.
        audio (str): Key for audio parameters.
        input (str): Key for the input parameter in audio creation.
        create (str): Key representing the create method.
        init (str): Key for initialization parameters.
        model (str): Key for the model parameter.
        not_given (str): Placeholder key for undefined parameters.
        image (str): Key for image parameters.
        mask (str): Key for mask parameter in image edition.
        prompt (str): Key for the textual prompt.
        response_format (str): Key to specify the expected API response format.
    """

    messages: str = "messages"
    file: str = "file"
    audio: str = "audio"
    input: str = "input"
    create: str = "create"
    init: str = "init"
    model: str = "model"
    not_given: str = "NOT_GIVEN"
    image: str = "image"
    mask: str = "mask"
    prompt: str = "prompt"
    response_format: str = "response_format"
