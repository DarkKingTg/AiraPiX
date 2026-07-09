from .config import AiraConfig

__all__ = ["AiraConfig", "AiraForCausalLM"]


def __getattr__(name: str):
    if name == "AiraForCausalLM":
        from .model import AiraForCausalLM

        return AiraForCausalLM
    raise AttributeError(name)
