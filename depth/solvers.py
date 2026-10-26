"""Registered models: the OpenRouter providers each model is pinned to (fallbacks disabled) and the model info
passed to litellm. solver_kwargs raises KeyError for an unregistered model."""
import json

SOLVERS = {
    "openrouter/deepseek/deepseek-v4.1-flash": {
        "provider": ["DeepSeek", "Together", "Wafer"],
        "info": {"max_input_tokens": 1048576, "max_output_tokens": 384000,
                 "input_cost_per_token": 1.5e-7, "output_cost_per_token": 6e-7,
                 "cache_read_input_token_cost": 3e-9, "litellm_provider": "openrouter", "mode": "chat"}},
    "openrouter/z-ai/glm-5.3-flash": {
        "provider": ["Fireworks"],
        "info": {"max_input_tokens": 1048576, "max_output_tokens": 131072,
                 "input_cost_per_token": 1.5e-7, "output_cost_per_token": 5e-7,
                 "cache_read_input_token_cost": 3e-8, "litellm_provider": "openrouter", "mode": "chat"}},
}


def harbor_model(model):
    """Model name handed to Harbor (litellm)."""
    return model


def gateway_body(model):
    """Request fields sent with every call: pin the provider, plus the model's extra_body (e.g. a reasoning effort)."""
    sv = SOLVERS[model]
    return {"provider": {"order": sv["provider"], "allow_fallbacks": False}, **sv.get("extra_body", {})}


def model_info(model):
    """Context size and prices registered with litellm (None if litellm already knows the model)."""
    return SOLVERS[model].get("info")


def solver_kwargs(model, max_turns):
    """Harbor --agent-kwarg values for terminus-2: turn limit, pinned provider and model info."""
    kw = [f"max_turns={max_turns}"] if max_turns else []
    kw.append("llm_call_kwargs=" + json.dumps({"extra_body": gateway_body(model)}))
    if model_info(model):
        kw.append("model_info=" + json.dumps(model_info(model)))
    return kw


def tag(model):
    """Directory-safe label of a model, used to name result directories."""
    return model.replace("openrouter/", "").replace("/", "_")
