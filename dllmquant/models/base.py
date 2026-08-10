"""Model adapter protocol.

Everything the quantization pipeline needs to know about a specific DLLM lives
behind this interface: where the transformer blocks are, what the mask token
is, how to roll out a denoising trajectory, and how to observe the attention
matrix for IA-AQ.

The adapters below are written against the published architectures but could
not be executed against real weights while writing them.  They therefore
*verify* their assumptions at load time and raise with a precise message rather
than silently quantizing the wrong tensors -- run ``scripts/selfcheck.py``
before a long job.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..calib.tmas import Snapshot
from ..config import TMASConfig

# Leaf names that mark a module as an attention layer.  Used to tell a decoder
# block apart from the other uniform ModuleLists a model may contain.
_ATTENTION_NAMES = (
    "self_attn", "attention", "attn", "att_proj", "attn_out",
    "q_proj", "wq", "qkv_proj", "Wqkv",
)


class ArchitectureMismatch(RuntimeError):
    """Raised when the loaded checkpoint does not look like what we expect."""


@dataclass
class AttentionParts:
    """What IA-AQ needs from one attention layer for one forward pass."""

    value_states: torch.Tensor  # [B, heads, K, head_dim] or [B, K, D]
    attn_probs: torch.Tensor  # [B, heads, Q, K], rows sum to 1


def preflight_memory(required_gb: float, strict: bool = True) -> float:
    """Refuse to start a long job on a GPU that cannot hold it.

    On a shared node the free memory at launch is not the free memory five
    minutes later, but checking up front still turns the common case -- the
    card was already full -- from a twenty-minute wait into a two-second
    message naming the emptier GPU to use.
    """
    import torch as _torch

    if not _torch.cuda.is_available():
        return 0.0

    free, total = _torch.cuda.mem_get_info()
    free_gb, total_gb = free / 2**30, total / 2**30
    print(f"GPU 0: {free_gb:.1f} GB free of {total_gb:.1f} GB "
          f"(need about {required_gb:.0f} GB)")

    if free_gb >= required_gb:
        return free_gb

    msg = (
        f"only {free_gb:.1f} GB free, this job needs about {required_gb:.0f} GB "
        f"(weights plus Hessians and the calibration cache).\n"
        "On a shared node, pick an emptier GPU:\n"
        "    nvidia-smi --query-gpu=index,memory.used,memory.total "
        "--format=csv\n"
        "    export CUDA_VISIBLE_DEVICES=<index>\n"
        "    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
        "Pass --force to start anyway."
    )
    if strict:
        raise RuntimeError(msg)
    print(f"[warn] {msg}")
    return free_gb


def checkpoint_size_gb(model_path: str) -> Optional[float]:
    """Actual checkpoint size, read from the safetensors index if downloaded.

    Guessing from the model name does not survive contact with a MoE: LLaDA-8B
    and LLaDA2.0-mini differ by 2x in weights while the second is advertised as
    "16B-A1B", and the active-parameter count is not what has to fit in memory.
    """
    import glob
    import json
    import os

    slug = "models--" + model_path.replace("/", "--")
    root = os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        "hub", slug,
    )
    for index in glob.glob(os.path.join(root, "snapshots", "*", "*.index.json")):
        try:
            with open(index, encoding="utf-8") as f:
                total = json.load(f).get("metadata", {}).get("total_size")
            if total:
                return float(total) / 2**30
        except (OSError, ValueError):
            continue

    shards = glob.glob(os.path.join(root, "snapshots", "*", "*.safetensors"))
    if shards:
        seen, total = set(), 0
        for s in shards:
            real = os.path.realpath(s)
            if real not in seen:
                seen.add(real)
                total += os.path.getsize(real)
        if total:
            return float(total) / 2**30
    return None


def estimate_required_gb(cfg) -> float:
    """Weights plus working room for the solver."""
    measured = checkpoint_size_gb(cfg.model_path)
    if measured is not None:
        bytes_per = {"float32": 4, "float16": 2, "bfloat16": 2}.get(cfg.dtype, 2)
        # The checkpoint is normally stored in bf16; loading in float32 doubles
        # it, and Hessians for the widest layer add roughly a gigabyte.
        return measured * (bytes_per / 2.0) + 4.0

    bytes_per = {"float32": 4, "float16": 2, "bfloat16": 2}.get(cfg.dtype, 2)
    return 8.0 * bytes_per + 4.0  # not downloaded yet: assume 8B


def _dtype_kwargs(dtype) -> dict:
    """`torch_dtype` was renamed to `dtype` in transformers 4.56."""
    import transformers

    try:
        parts = transformers.__version__.split(".")
        major, minor = int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        major, minor = 0, 0
    key = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    return {key: dtype}


# Each checkpoint ships its own remote code, written against whatever
# transformers existed at release, and the windows do NOT overlap:
#
#   LLaDA-1.5   4.38 - 4.46   breaks above on `tie_weights()` and friends
#   LLaDA2.0    4.56+         needs `dynamic_rope_update`, `TransformersKwargs`
#
# So the supported range belongs to the adapter, not to this module. A single
# global bound cannot express "one model needs an old library and the other
# needs a new one", and pretending otherwise sends you round the loop of
# installing a version, hitting one missing symbol, installing another.
MAX_TESTED_TRANSFORMERS_MAJOR = 4  # kept for backwards compatibility

VersionBound = Optional[Tuple[int, int]]


def _parse_version(version: str) -> Optional[Tuple[int, int]]:
    try:
        parts = version.split(".")
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return None


def check_transformers_version(
    minimum: VersionBound = None,
    maximum: VersionBound = None,
    strict: bool = True,
    model: str = "this checkpoint",
) -> str:
    """Verify the installed transformers falls in the adapter's window.

    ``maximum`` is exclusive: (4, 47) means "below 4.47".
    """
    import transformers

    version = transformers.__version__
    parsed = _parse_version(version)
    if parsed is None or not strict:
        return version

    too_old = minimum is not None and parsed < minimum
    too_new = maximum is not None and parsed >= maximum
    if not (too_old or too_new):
        return version

    window = []
    if minimum:
        window.append(f">={minimum[0]}.{minimum[1]}")
    if maximum:
        window.append(f"<{maximum[0]}.{maximum[1]}")
    wanted = " and ".join(window) if window else "any"

    raise RuntimeError(
        f"transformers {version} is {'older' if too_old else 'newer'} than the "
        f"remote code of {model} supports (needs {wanted}).\n\n"
        f"Loading would fail inside the checkpoint's own modelling file, on a "
        f"missing symbol whose name tells you nothing useful.\n\n"
        f"    pip install 'transformers=={_suggest(minimum, maximum)}'\n\n"
        f"Note the two LLaDA families need incompatible versions -- 1.5 wants "
        f"4.38-4.46, 2.0 wants 4.56+ -- so keep a separate venv if you need "
        f"both. Weights on disk are reused either way.\n"
        f"To try anyway, pass --allow-untested-transformers."
    )


def _suggest(minimum: VersionBound, maximum: VersionBound) -> str:
    if maximum == (4, 47):
        return "4.46.3"
    if minimum and minimum >= (4, 56):
        return "4.57.1"
    return "4.46.3"


def ensure_tied_weights_attr(cfg) -> bool:
    """Make LLaDA's remote code loadable under a modern transformers.

    Transformers ~4.57 reads ``model.all_tied_weights_keys`` in several places
    along the loading path (device-map inference, then
    ``_move_missing_keys_from_meta_to_device``).  It is set as an *instance*
    attribute during ``PreTrainedModel`` initialisation, and LLaDA's
    remote-code ``LLaDAModelLM`` -- written against transformers 4.38 -- never
    sets it, so attribute lookup falls through to ``nn.Module.__getattr__``
    and raises.

    A class-level default fixes every call site at once, and ``{}`` is the
    *correct* value exactly when the checkpoint ties nothing.  So it is only
    installed after checking the config: claiming "nothing is tied" for a model
    that does tie would silently leave the output head randomly initialised
    instead of bound to the embeddings.

    Returns True if the shim was installed.
    """
    import transformers

    base = transformers.PreTrainedModel
    if hasattr(base, "all_tied_weights_keys"):
        return False  # this transformers already provides a class-level default

    from transformers import AutoConfig

    conf = AutoConfig.from_pretrained(cfg.model_path, trust_remote_code=True)
    tied = getattr(conf, "weight_tying", None)
    if tied is None:
        tied = getattr(conf, "tie_word_embeddings", None)

    if tied:
        raise RuntimeError(
            f"{cfg.model_path} ties its embedding and output weights, and this "
            f"transformers ({transformers.__version__}) needs "
            "`all_tied_weights_keys`, which the checkpoint's remote code does "
            "not define. Pretending nothing is tied would leave the output "
            "head uninitialised, so pin the library instead:\n"
            "    pip install 'transformers==4.38.2'"
        )

    base.all_tied_weights_keys = {}
    return True


def load_pretrained(
    auto_class,
    cfg,
    minimum: VersionBound = None,
    maximum: VersionBound = None,
):
    """Load a checkpoint, avoiding accelerate's device-map inference by default.

    LLaDA ships as remote code written against an older transformers.  Newer
    releases infer the device map through ``model.all_tied_weights_keys``,
    which ``PreTrainedModel`` defines but the remote-code class does not, so
    ``device_map="auto"`` raises ``AttributeError`` before any weight is
    placed.  Nothing here needs a device map for a model that fits on one GPU,
    so it is opt-in via ``cfg.device_map``.
    """
    import torch as _torch

    check_transformers_version(
        minimum=minimum,
        maximum=maximum,
        strict=not getattr(cfg, "allow_untested", False),
        model=cfg.model_path or "this checkpoint",
    )
    if ensure_tied_weights_attr(cfg):
        print("[compat] installed PreTrainedModel.all_tied_weights_keys = {} "
              "(checkpoint reports weight_tying=False)")

    dtype = getattr(_torch, cfg.dtype)
    kwargs = {"trust_remote_code": True, **_dtype_kwargs(dtype)}

    if cfg.device_map:
        kwargs["device_map"] = cfg.device_map
        return auto_class.from_pretrained(cfg.model_path, **kwargs)

    model = auto_class.from_pretrained(cfg.model_path, **kwargs)
    if cfg.device and cfg.device != "cpu":
        if not _torch.cuda.is_available():
            raise RuntimeError(
                f"device='{cfg.device}' but CUDA is unavailable; "
                "pass --device cpu (very slow) or fix the environment"
            )
        model = model.to(cfg.device)
    return model


def _contains_attention(module: nn.Module) -> bool:
    """Whether a module has anything that looks like an attention projection."""
    return any(
        name.split(".")[-1] in _ATTENTION_NAMES for name, _ in module.named_modules()
    )


def discover_blocks(model: nn.Module) -> Tuple[str, nn.ModuleList]:
    """Find the transformer block list without hardcoding a module path.

    Returns the qualified name and the ModuleList.  Candidates are
    ``nn.ModuleList``s whose entries all share one class; among those, the one
    whose entries contain attention wins, and length breaks ties.

    Attention is the filter rather than length because in an MoE the expert
    list is *longer than the layer list* -- LLaDA2.0-mini has 20 layers and 256
    experts in each -- so "longest ModuleList" picks ``layers.1.mlp.experts``.
    Nothing downstream notices: the rotation plan, CGQ grouping and the
    attention probe all just operate on one layer's experts and report
    plausible-looking numbers for a model that was never touched.
    """
    candidates: List[Tuple[str, nn.ModuleList]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList) or len(module) < 2:
            continue
        if len({type(m) for m in module}) != 1:
            continue
        candidates.append((name, module))

    if not candidates:
        raise ArchitectureMismatch(
            "could not find a transformer block list in this model; "
            "pass the module path explicitly"
        )

    with_attention = [c for c in candidates if _contains_attention(c[1][0])]
    if not with_attention:
        warnings.warn(
            "no ModuleList whose entries contain attention; falling back to the "
            "longest one, which in an MoE is likely the expert list rather than "
            "the layers. Check the path printed by describe().",
            RuntimeWarning,
        )
    pool = with_attention or candidates
    return max(pool, key=lambda c: len(c[1]))


def find_submodule(root: nn.Module, names: Sequence[str]) -> Optional[nn.Module]:
    """Return the first direct-or-nested child whose attribute name matches."""
    for name in names:
        for mod_name, module in root.named_modules():
            if mod_name.split(".")[-1] == name:
                return module
    return None


def _split_evenly(total: int, parts: int) -> List[int]:
    """Split ``total`` into ``parts`` near-equal integers, remainder first."""
    base, remainder = divmod(total, parts)
    return [base + (1 if i < remainder else 0) for i in range(parts)]


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """How many tokens to commit at each step, spread as evenly as possible.

    Mirrors LLaDA's reference implementation: the masked budget of a block is
    split over ``steps``, with the remainder front-loaded.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    out = torch.zeros(mask_num.size(0), steps, device=mask_index.device,
                      dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        out[i, : int(remainder[i])] += 1
    return out


class ModelAdapter(ABC):
    """Bridges one DLLM family to the quantization pipeline."""

    mask_id: int
    model: nn.Module
    tokenizer: object

    # ------------------------------------------------------------ structure

    @abstractmethod
    def load(self) -> None:
        """Instantiate model + tokenizer and validate the architecture."""

    @property
    @abstractmethod
    def blocks(self) -> nn.ModuleList:
        """The transformer blocks, in execution order."""

    @abstractmethod
    def make_probe(self, block: nn.Module) -> "AttentionProbe":
        """A probe that captures V and the softmax matrix for this block.

        Block inputs are captured by the pipeline with a catcher hook rather
        than by re-running the embedding stack, so adapters do not need to know
        what happens before block 0.
        """

    # ------------------------------------------------------------ forward

    def forward_kwargs(self, input_ids: torch.Tensor) -> Dict[str, Any]:
        """Extra arguments this model's forward needs beyond ``input_ids``.

        Empty for LLaDA-1.5, which attends over everything and is happy with
        ``model(x)``.  LLaDA2.0 refuses to run without an explicit 4-D block
        attention mask, so it fills this in.  Every call site that runs the
        model -- sampler, calibration capture, rotation check, probes -- goes
        through here so the two families cannot drift apart.
        """
        return {}

    def _sequence_layout(
        self, p_len: int, cfg: TMASConfig
    ) -> Tuple[int, List[Tuple[int, int]]]:
        """Total sequence length and the ``[lo, hi)`` span of each block.

        LLaDA-1.5 attends over the whole sequence, so its blocks start wherever
        the prompt happens to end and nothing else has to line up.  Block
        diffusion models pin their blocks to a grid anchored at position 0 --
        the attention mask is built from that grid -- and override this.
        """
        if cfg.gen_length % cfg.block_length != 0:
            raise ValueError("gen_length must be divisible by block_length")
        n_blocks = cfg.gen_length // cfg.block_length
        bounds = [
            (p_len + i * cfg.block_length, p_len + (i + 1) * cfg.block_length)
            for i in range(n_blocks)
        ]
        return p_len + cfg.gen_length, bounds

    # ------------------------------------------------------------ generation

    @torch.no_grad()
    def trace_generation(
        self, prompt_ids: torch.Tensor, cfg: TMASConfig
    ) -> List[Snapshot]:
        """Roll out a trajectory and return every intermediate state."""
        snapshots: List[Snapshot] = []
        self._denoise(prompt_ids, cfg, on_step=snapshots.append)
        return snapshots

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, cfg: TMASConfig) -> torch.Tensor:
        """Same sampler without instrumentation; returns the full sequence."""
        return self._denoise(prompt_ids, cfg, on_step=None)

    @torch.no_grad()
    def _denoise(
        self,
        prompt_ids: torch.Tensor,
        cfg: TMASConfig,
        on_step=None,
    ) -> torch.Tensor:
        """Block-wise low-confidence remasking, the reference LLaDA sampler.

        ``on_step`` receives a :class:`Snapshot` of the state the network just
        saw, before any token is committed -- that is what TMAS samples from.
        """
        device = next(self.model.parameters()).device
        prompt_ids = prompt_ids.to(device)
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)

        p_len = prompt_ids.shape[1]
        total_len, bounds = self._sequence_layout(p_len, cfg)
        x = torch.full(
            (1, total_len), self.mask_id, dtype=torch.long, device=device
        )
        x[:, :p_len] = prompt_ids

        if cfg.steps < len(bounds):
            raise ValueError(
                f"{cfg.steps} steps cannot cover {len(bounds)} blocks -- some "
                "block would never be visited and stay masked"
            )
        # Even split, remainder front-loaded.  With the LLaDA-1.5 layout and a
        # divisible step count this is exactly the old steps // n_blocks; the
        # block-diffusion layout derives its block count from the prompt
        # length, which the caller does not control, so it cannot demand
        # divisibility.
        schedule = _split_evenly(cfg.steps, len(bounds))

        step_offset = 0
        for block_idx, (lo, hi) in enumerate(bounds):
            steps_per_block = schedule[block_idx]
            block_mask = x[:, lo:hi] == self.mask_id
            budget = get_num_transfer_tokens(block_mask, steps_per_block)

            for step in range(steps_per_block):
                mask_index = x == self.mask_id
                logits = self.model(x, **self.forward_kwargs(x)).logits

                probs = torch.softmax(logits.to(torch.float32), dim=-1)
                conf_all, x0 = probs.max(dim=-1)

                # Positions outside the current block are not eligible yet.
                conf = torch.where(mask_index, conf_all, torch.zeros_like(conf_all))
                eligible = torch.zeros_like(mask_index)
                eligible[:, lo:hi] = True
                selectable = mask_index & eligible

                # ---- snapshot the state the network just saw ----
                if on_step is not None:
                    resp_mask = mask_index[0, p_len:]
                    on_step(
                        Snapshot(
                            input_ids=x[0].detach().cpu().clone(),
                            mask=mask_index[0].detach().cpu().clone(),
                            confidence=conf[0].detach().float().cpu().clone(),
                            step=step_offset + step,
                            total_steps=cfg.steps,
                            block_idx=block_idx,
                            mask_ratio=float(resp_mask.float().mean()),
                        )
                    )

                # ---- commit the most confident tokens of this block ----
                k = int(budget[0, step])
                if k > 0:
                    score = torch.where(
                        selectable, conf_all, torch.full_like(conf_all, -float("inf"))
                    )
                    idx = torch.topk(score[0], k=min(k, int(selectable.sum()))).indices
                    x[0, idx] = x0[0, idx]

            step_offset += steps_per_block

        return x

    # -------------------------------------------------------------- rotation

    def _extra_residual_readers(self, block: nn.Module) -> List[Tuple[str, nn.Module]]:
        """``(leaf, module)`` pairs that read the residual but are not Linear.

        Nothing in a dense block qualifies, so the rotation planner's scan for
        ``nn.Linear`` sees everything there is.  An MoE router does not: it is
        a bare ``nn.Parameter`` with an ``F.linear`` call around it.
        """
        return []

    def rotation_plan(self):
        """Which modules take part in the QuaRot residual-stream rotation."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement rotation_plan(); "
            "run without --rotate or add it"
        )

    def down_projections(self) -> List[nn.Linear]:
        """MLP down-projections, the only place needing an online Hadamard."""
        out = []
        for block in self.blocks:
            for name, module in block.named_modules():
                if (
                    name.split(".")[-1] in ("ff_out", "down_proj", "w2")
                    and isinstance(module, nn.Linear)
                ):
                    out.append(module)
        return out

    def install_online_hadamards(self) -> None:
        """Wrap each down-projection so its input is rotated at runtime."""
        from ..algos.quarot import OnlineHadamard

        for block in self.blocks:
            for parent_name, parent in list(block.named_modules()):
                for attr, child in list(parent.named_children()):
                    if attr in ("ff_out", "down_proj", "w2") and isinstance(
                        child, nn.Linear
                    ):
                        setattr(parent, attr, OnlineHadamard(child, name=attr))

    # ------------------------------------------------------------- utilities

    def encode_prompts(self, texts: Sequence[str], max_len: int) -> List[torch.Tensor]:
        out = []
        for t in texts:
            ids = self.tokenizer(t, return_tensors="pt", truncation=True,
                                 max_length=max_len)["input_ids"][0]
            out.append(ids)
        return out

    def describe(self) -> str:
        name, blocks = discover_blocks(self.model)
        return (
            f"{type(self).__name__}: {len(blocks)} blocks at '{name}', "
            f"mask_id={self.mask_id}"
        )


class AttentionProbe(ABC):
    """Captures the value matrix and softmax weights of one attention layer."""

    def __init__(self, module: nn.Module):
        self.module = module
        self.parts: Optional[AttentionParts] = None
        self._handle = None

    @abstractmethod
    def _hook(self, module, args, kwargs, output) -> None: ...

    def __enter__(self) -> "AttentionProbe":
        self._handle = self.module.register_forward_hook(
            self._hook, with_kwargs=True
        )
        return self

    def __exit__(self, *exc) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self.parts = None


__all__ = [
    "ModelAdapter",
    "load_pretrained",
    "ensure_tied_weights_attr",
    "check_transformers_version",
    "MAX_TESTED_TRANSFORMERS_MAJOR",
    "preflight_memory",
    "estimate_required_gb",
    "AttentionProbe",
    "AttentionParts",
    "ArchitectureMismatch",
    "discover_blocks",
    "find_submodule",
    "get_num_transfer_tokens",
]
