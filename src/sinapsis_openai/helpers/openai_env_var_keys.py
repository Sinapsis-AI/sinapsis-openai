# -*- coding: utf-8 -*-
from typing import Any

from pydantic import BaseModel
from sinapsis_core.utils.env_var_keys import EnvVarEntry, doc_str, return_docs_for_vars


class _OpenAIEnvVars(BaseModel):
    """
    Env vars for OpenAI
    """

    OPENAI_API_KEY: EnvVarEntry = EnvVarEntry(
        var_name="OPENAI_API_KEY",
        default_value="",
        allowed_values=None,
        description="set api key for OPENAI",
    )


OpenAIEnvVars = _OpenAIEnvVars()

doc_str = return_docs_for_vars(OpenAIEnvVars, docs=doc_str, string_for_doc="""OpenAI env vars available: \n""")
__doc__ = doc_str


def __getattr__(name: str) -> Any:
    """to use as an import, when updating the value is not important"""
    if name in OpenAIEnvVars.model_fields:
        return OpenAIEnvVars.model_fields[name].default.value

    raise AttributeError(f"Agent does not have `{name}` env var")


_all__ = (*list(OpenAIEnvVars.model_fields.keys()), "OpenAIEnvVars")
