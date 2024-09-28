# SGL API Components

from coalinfer.api import (
    Runtime,
    assistant,
    assistant_begin,
    assistant_end,
    flush_cache,
    function,
    gen,
    gen_int,
    gen_string,
    get_server_args,
    image,
    select,
    set_default_backend,
    system,
    system_begin,
    system_end,
    user,
    user_begin,
    user_end,
    video,
)
from coalinfer.lang.choices import (
    greedy_token_selection,
    token_length_normalized,
    unconditional_likelihood_normalized,
)

# Coalinfer DSL APIs
__all__ = [
    "Runtime",
    "assistant",
    "assistant_begin",
    "assistant_end",
    "flush_cache",
    "function",
    "gen",
    "gen_int",
    "gen_string",
    "get_server_args",
    "image",
    "select",
    "set_default_backend",
    "system",
    "system_begin",
    "system_end",
    "user",
    "user_begin",
    "user_end",
    "video",
    "greedy_token_selection",
    "token_length_normalized",
    "unconditional_likelihood_normalized",
]

# Global Configurations
from coalinfer.global_config import global_config

__all__ += ["global_config"]

from coalinfer.version import __version__

__all__ += ["__version__"]

# SGL Backends
from coalinfer.lang.backend.runtime_endpoint import RuntimeEndpoint
from coalinfer.utils import LazyImport

Anthropic = LazyImport("coalinfer.lang.backend.anthropic", "Anthropic")
LiteLLM = LazyImport("coalinfer.lang.backend.litellm", "LiteLLM")
OpenAI = LazyImport("coalinfer.lang.backend.openai", "OpenAI")
VertexAI = LazyImport("coalinfer.lang.backend.vertexai", "VertexAI")

__all__ += ["Anthropic", "LiteLLM", "OpenAI", "VertexAI", "RuntimeEndpoint"]
