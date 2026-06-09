"""LLM generator factory. Reads provider/model from settings (runtime-editable)."""
from haystack.utils import Secret


def make_generator(streaming_callback=None):
    from livedocs.settings import get_settings
    s = get_settings()
    llm = s.llm

    gen_kwargs = {
        "temperature": llm.temperature,
        "max_output_tokens": llm.max_output_tokens,
    }
    kwargs = dict(generation_kwargs=gen_kwargs)
    if streaming_callback is not None:
        kwargs["streaming_callback"] = streaming_callback

    provider = llm.provider.lower()

    if provider == "google":
        from google.genai import types as genai_types
        from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator
        gen_kwargs["thinking_config"] = genai_types.ThinkingConfig(
            thinking_budget=llm.thinking_budget,
        )
        return GoogleGenAIChatGenerator(
            model=llm.model,
            api_key=Secret.from_env_var("GOOGLE_API_KEY"),
            **kwargs,
        )

    if provider == "openai":
        from haystack.components.generators.chat import OpenAIChatGenerator
        return OpenAIChatGenerator(
            model=llm.model,
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            **kwargs,
        )

    if provider == "anthropic":
        from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator
        return AnthropicChatGenerator(
            model=llm.model,
            api_key=Secret.from_env_var("ANTHROPIC_API_KEY"),
            **kwargs,
        )

    raise ValueError(
        f"Unknown LLM provider {llm.provider!r}. Set to 'google', 'openai', or 'anthropic'."
    )


def make_lite_client():
    """Cheap raw client for rewrite/decompose. Returns fn(system_prompt, user_prompt) -> str."""
    from livedocs.settings import get_settings
    s = get_settings()
    provider = s.llm.provider.lower()
    model = s.llm.rewriter_model

    if provider == "google":
        import os
        from google import genai
        from google.genai import types as genai_types
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

        def call(system_prompt, user_prompt):
            resp = client.models.generate_content(
                model=model,
                contents=f"{system_prompt}\n\n{user_prompt}",
                config=genai_types.GenerateContentConfig(temperature=0.0, max_output_tokens=256),
            )
            return (resp.text or "").strip()
        return call

    if provider == "openai":
        import os
        from openai import OpenAI
        oc = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        def call(system_prompt, user_prompt):
            resp = oc.chat.completions.create(
                model=model,
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
        ac = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

        def call(system_prompt, user_prompt):
            msg = ac.messages.create(
                model=model,
                max_tokens=256,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return msg.content[0].text.strip()
        return call

    raise ValueError(f"Unknown LLM provider {provider!r}.")
