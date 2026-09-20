"""Fast-dLLM-v2's own modelling code, pinned to one checkpoint revision.

Imported lazily by the adapter: it needs transformers 4.53+ and a torch with
``flex_attention``, which the LLaDA-1.5 half of this project (transformers 4.46)
does not have.
"""

# The checkpoint revision these files were taken from.
REVISION = "0661abf5f9f0ee338970d091052a26c8efa51974"

__all__ = ["REVISION"]
