"""LLM generator factory. Swap provider = change LLM_PROVIDER + LLM_MODEL in config.py.

Supported providers:
  "google"    -- GoogleGenAIChatGenerator (requires GOOGLE_API_KEY)
  "openai"    -- OpenAIChatGenerator      (requires OPENAI_API_KEY)
  "anthropic" -- AnthropicChatGenerator   (requires ANTHROPIC_API_KEY)

streaming_callback: if provided, the generator will call it with each StreamingChunk
as tokens arrive. Use the queue bridge in code_pipeline.py for SSE streaming.
"""
from config import LLM_PROVIDER, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_OUTPUT_TOKENS
from haystack.utils import Secret


def make_generator(streaming_callback=None):
    kwargs = dict(
        generation_kwargs={
            "temperature": LLM_TEMPERATURE,
            "max_output_tokens": LLM_MAX_OUTPUT_TOKENS,
        },
    )
    if streaming_callback is not None:
        kwargs["streaming_callback"] = streaming_callback

    provider = LLM_PROVIDER.lower()

    if provider == "google":
        from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator
        return GoogleGenAIChatGenerator(
            model=LLM_MODEL,
            api_key=Secret.from_env_var("GOOGLE_API_KEY"),
            **kwargs,
        )

    if provider == "openai":
        from haystack.components.generators.chat import OpenAIChatGenerator
        return OpenAIChatGenerator(
            model=LLM_MODEL,
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            **kwargs,
        )

    if provider == "anthropic":
        from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator
        return AnthropicChatGenerator(
            model=LLM_MODEL,
            api_key=Secret.from_env_var("ANTHROPIC_API_KEY"),
            **kwargs,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER {LLM_PROVIDER!r}. "
        "Set to 'google', 'openai', or 'anthropic' in config.py."
    )


def make_lite_client():
    """Return a raw (non-Haystack) client for cheap rewrite/decompose calls.

    Returns a callable: fn(system_prompt, user_prompt) -> str
    Provider-specific, but all return plain text.
    """
    provider = LLM_PROVIDER.lower()

    if provider == "google":
        import os
        from google import genai
        from google.genai import types as genai_types
        from config import REWRITER_MODEL

        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

        def call(system_prompt, user_prompt):
            full = f"{system_prompt}\n\n{user_prompt}"
            resp = client.models.generate_content(
                model=REWRITER_MODEL,
                contents=full,
                config=genai_types.GenerateContentConfig(temperature=0.0, max_output_tokens=256),
            )
            return (resp.text or "").strip()

        return call

    if provider == "openai":
        import os
        from openai import OpenAI
        from config import REWRITER_MODEL

        oc = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        def call(system_prompt, user_prompt):
            resp = oc.chat.completions.create(
                model=REWRITER_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=256,
            )
            return resp.choices[0].message.content.strip()

        return call

    if provider == "anthropic":
        import os
        import anthropic
        from config import REWRITER_MODEL

        ac = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

        def call(system_prompt, user_prompt):
            msg = ac.messages.create(
                model=REWRITER_MODEL,
                max_tokens=256,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return msg.content[0].text.strip()

        return call

    raise ValueError(f"Unknown LLM_PROVIDER {LLM_PROVIDER!r}.")
