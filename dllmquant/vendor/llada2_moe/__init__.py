"""LLaDA2.0's own modelling code, pinned to one checkpoint revision.

Importing this needs transformers >= 4.56 (`dynamic_rope_update`,
`TransformersKwargs`), so it is imported lazily by the adapter rather than at
package import time -- the LLaDA-1.5 half of this project runs on 4.46 and
must not be dragged down by a module it never uses.
"""

# The checkpoint revision these files were taken from. Reported by the
# adapter so a result can be traced to the model code that produced it --
# remote code is fetched at load time and can change between runs.
REVISION = "dad945cac317da394b390f82c7b40691d8a881ed"

__all__ = ["REVISION"]
