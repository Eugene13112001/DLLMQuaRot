#!/usr/bin/env python3
"""LongBench evaluation of the QuMix KV cache on SDAR-8B-Chat and TraDo-8B-Instruct (block diffusion, block 4).

The cache is quantized in simulation (sealed fake quantization): the forward pass that produces keys and values uses
them exactly; once a 32-token group of committed tokens is complete, its cached keys and values are replaced by
their quantized reconstruction, and the incomplete tail group stays exact. With --prerope, keys are quantized before
RoPE and RoPE is re-applied after reconstruction.

Arms (--arms, comma separated; every arm runs on the same examples):
  base                  16-bit cache
  qmx<B>qs8ts           QuMix with calibration tag qmx<B>qs8 (B = 15, 10, 075 -> 1.99, 1.69, 1.55 KV bits; use --vsbits 8)
  qmxmatch<X>ts         QuMix re-allocated at the KV bits of compared method X (tags from calibration/build_same_kv.py;
                        --vsbits 8)
Other arms and options in this file (the compared methods and development ablations) are not part of this release.

Calibration (--calibonly --savecalib): capture pre-RoPE queries and keys while the model decodes 128 tokens for each
WikiText-2 segment, then build the query-weighted basis and the bit allocations (see README, Section 2).

Reading guide. qumix/reference.py implements the QuMix computations of this file in about 200 lines, and
tests/test_reference.py checks that the two agree bit for bit. In this file, the sections are marked by "==========":
  [QuMix] calibration capture ........ cap_rope and build_calib_prompts (statistics C_q and G_k per layer and KV head)
  [QuMix] basis and bit allocation ... the block that fills PROJ; the paper's {0,1,2,4,8,16}-bit allocation is then
                                      computed by calibration/build_po2q.py from the stored spectrum 'sig2'
  [QuMix] key quantization ........... qmx_k with tokq=True (values: _asym_sq, enabled by --vsbits 8)
  [QuMix] sealed cache update ........ sealed_qmx_update; quant_rope records the pre-RoPE keys it quantizes
  [evaluation] ...................... LongBench prompts and metrics, then the loop over arms
"""
import sys, re, argparse, time
import os as _os0; sys.path.insert(0, _os0.path.dirname(_os0.path.abspath(__file__)))   # vendored SDAR generate.py (MIT)
import torch, torch.nn.functional as F
import transformers.utils as U
from typing import TypedDict
for nm in ('LossKwargs', 'KwargsForCausalLM'):
    if not hasattr(U, nm): setattr(U, nm, TypedDict(nm, {}, total=False))
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from datasets import load_dataset
import generate as G

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=50)
ap.add_argument('--start', type=int, default=0)   # shard: eval test[start:n]
ap.add_argument('--gen', type=int, default=256)
ap.add_argument('--block', type=int, default=4)
ap.add_argument('--steps', type=int, default=4)
ap.add_argument('--temp', type=float, default=0.1)
ap.add_argument('--ncalib', type=int, default=3)
ap.add_argument('--arms', type=str, default='base,kivi4,kivi2,qmx4,qmx2')
ap.add_argument('--calibonly', action='store_true')
ap.add_argument('--calibset', type=str, default='gsm8k')     # gsm8k (in-domain) | wikitext (generic)
ap.add_argument('--savecalib', type=str, default='')         # save built PROJ/CENTK/KVQF to path
ap.add_argument('--loadcalib', type=str, default='')         # load cached calib, skip calib (fast eval shards)
ap.add_argument('--noiseprof', type=str, default='')         # per-layer sigma table for noiseNNN arms
ap.add_argument('--vnuq', action='store_true')
ap.add_argument('--vqmx', type=str, default='')                # V-QuMix calib (wiki_calib_vqmx.pt) -> VPROJ
ap.add_argument('--vqmxtag', type=str, default='vqmx15')      # which V budget in VPROJ
ap.add_argument('--vgs', type=int, default=32)
ap.add_argument('--vhead', type=str, default='')               # per-head V bits file
ap.add_argument('--vheadtag', type=str, default='v20')                 # V per-direction scale group (tokens) # V: per-token NUQ (codebook) instead of uniform asym
ap.add_argument('--vattnlo', type=int, default=1)   # floor for per-token V bits (1 forces a 1-bit tail at avg=2)
ap.add_argument('--vattnmode', type=str, default='plain')  # plain | nosink | norm
ap.add_argument('--vattn', action='store_true')              # V: attention-weighted per-token bit allocation (avg=vbits)
ap.add_argument('--calibgen', type=int, default=128)         # gen tokens during basis calib (qmx only; kvqf range uses plain-fwd)
ap.add_argument('--prerope', action='store_true')   # (c): calibrate+quantize K PRE-RoPE (via apply_rotary)
ap.add_argument('--vbits', type=int, default=4)     # V quant bits for qmx/kvq2 (match KIVI=2 for fair K2V2)
ap.add_argument('--dataset', type=str, default='gsm8k')  # gsm8k | math500 | gpqa | mmlupro | longbench
ap.add_argument('--lbtask', type=str, default='multifieldqa_en')  # LongBench task (F1 QA)
ap.add_argument('--revision', type=str, default='')     # Hugging Face revision when --model is a hub id (TraDo: 2d37bd3e...)
ap.add_argument('--model', type=str, default='models/SDAR-8B-Chat')   # SDAR: directory made by scripts/prepare_sdar_model.sh  # SDAR-8B-Chat | TraDo-8B-Instruct (same arch/MASK)
ap.add_argument('--topk', type=int, default=1)                            # official TraDo eval: 0 (no top-k); SDAR default 1
ap.add_argument('--remask', type=str, default='low_confidence_static')    # official TraDo eval: low_confidence_dynamic
ap.add_argument('--lbmax', type=int, default=4096)   # truncation length axis
ap.add_argument('--lbgen', type=int, default=0)                           # >0: override LongBench per-task maxgen (gen-sensitivity check)
ap.add_argument('--ctxnoise', type=float, default=0.0)  # shuffle-substitute this fraction of context tokens (base-degradation control at fixed steps; vocab-preserving)
ap.add_argument('--kvqfnative', action='store_true')  # capture kvqf Fisher under the NATIVE block-causal mask (baseline-fairness fix; default ones_like = full-bidir on SDAR)
ap.add_argument('--ladder', action='store_true')      # also build the sub-2bit budget alphabet (qmx15/10/075/05/033) into PROJ
ap.add_argument('--gradglobal', action='store_true')  # older behaviour: grad enabled for the whole calibration, not just the Fisher block
ap.add_argument('--vsbits', type=int, default=0)     # >0: quantize QuMix value scales to this many bits (the paper uses 8)
ap.add_argument('--dump', type=str, default='')       # write per-example predictions and scores (JSON) to this path
ap.add_argument('--bs', type=int, default=1)                              # >1: exact-length batching (zero-pad, batch=1 semantics; not for longbench/squat)
args = ap.parse_args()
# --- run fingerprint: record the command line and the calibration file in the log ---
import os as _os
if args.dump: _os.environ['DUMP'] = args.dump
if args.vsbits: _os.environ['QMX_VSBITS'] = str(args.vsbits)
print(f"[fingerprint] argv: {' '.join(sys.argv)}", flush=True)
if args.loadcalib:
    _st = _os.stat(args.loadcalib)
    print(f"[fingerprint] calib: {args.loadcalib} size={_st.st_size} "
          f"mtime={time.strftime('%Y-%m-%d %H:%M', time.localtime(_st.st_mtime))}", flush=True)
PREROPE = args.prerope; VBITS = args.vbits
torch.set_grad_enabled(not args.gradglobal)   # eval-only; autograd activations were the 8K OOM
                                              # --gradglobal restores the older behaviour, to test
                                              # whether that change is why fresh kvqf codebooks are weaker
# SDAR passes a BOOL mask + enable_gqa -> only the math SDPA backend accepts it, and math
# materializes the full LxL score matrix (8K -> ~26GB transient -> OOM). Chunk the QUERY rows of the
# masked prefill call instead: numerics identical, transient memory capped at chunk x L.
_orig_sdpa = torch.nn.functional.scaled_dot_product_attention
def _chunked_sdpa(query, key, value, attn_mask=None, **kw):
    CH = 1024
    if attn_mask is None or query.shape[-2] <= CH:
        return _orig_sdpa(query, key, value, attn_mask=attn_mask, **kw)
    outs = []
    for i in range(0, query.shape[-2], CH):
        outs.append(_orig_sdpa(query[..., i:i+CH, :], key, value,
                               attn_mask=attn_mask[..., i:i+CH, :], **kw))
    return torch.cat(outs, dim=-2)
torch.nn.functional.scaled_dot_product_attention = _chunked_sdpa
KIVIHB = torch.load(_os.environ['QUMIX_KIVIHB_BITS']) if 'kivihb' in args.arms.split(',') else None  # (L,HKV) margin-aware per-head K bits
MODEL = args.model; MASK = 151669; EOS = [151645, 151643]  # MASK=<|MASK|> id 151669 shared by SDAR & TraDo
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, revision=args.revision or None)
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16, revision=args.revision or None).to('cuda').eval()
if 'TraDo' in MODEL:  # TraDo's rewritten attention expects a 4D (B,1,Q,K) mask; SDAR generate emits 3D (1,T,T)
    _mfwd = model.forward
    def _fwd_4dmask(*aa, **kw):
        am = kw.get('attention_mask', None)
        if am is not None and am.dim() == 3: kw['attention_mask'] = am.unsqueeze(1)
        return _mfwd(*aa, **kw)
    model.forward = _fwd_4dmask
    print("TraDo: patched model.forward to lift 3D attn mask -> 4D", flush=True)
cfg = model.config
HQ, HKV = cfg.num_attention_heads, cfg.num_key_value_heads
HD = cfg.hidden_size // HQ; GS = HQ // HKV; L = cfg.num_hidden_layers
print(f"SDAR {HQ}Q/{HKV}KV hd={HD} L={L} group={GS}", flush=True)

# ========== quantization primitives (QuMix values: _asym_sq; baselines: the rest) ==========
def qe4m3(s):
    m, e = torch.frexp(s); return torch.ldexp((m*8).round()/8, e)
def fq(x, bits, s8=False):
    if bits == 1:
        s = x.abs().mean(-1, keepdim=True).clamp(min=1e-8)
        if s8: s = qe4m3(s)
        return torch.sign(x)*s
    qmax = 2**(bits-1)-1
    s = x.abs().amax(-1, keepdim=True).clamp(min=1e-8)/qmax
    if s8: s = qe4m3(s)
    return (x/s).round().clamp(-qmax-1, qmax)*s
def _asym_sq(x, bits, rdim, sbits, widen=1.0, fixT=256):           # per-token asym with second-level (log step, min) scale quantization per head
    mn = x.amin(rdim, keepdim=True); mx = x.amax(rdim, keepdim=True)
    s = (mx-mn).clamp(min=1e-8)/(2**bits-1)                        # (B,H,T,1)
    Lv = 2**sbits-1; ng = max(1, min(s.shape[2], fixT))
    ls = s.log2(); lo = ls[:, :, :ng].amin(2, keepdim=True)-widen; hi = ls[:, :, :ng].amax(2, keepdim=True)+widen; d = (hi-lo).clamp(min=1e-8)/Lv
    s = 2**((((ls-lo)/d).round().clamp(0, Lv))*d+lo)
    lo = mn[:, :, :ng].amin(2, keepdim=True); hi = mn[:, :, :ng].amax(2, keepdim=True); w = (hi-lo)*widen*0.25; lo = lo-w; hi = hi+w; d = (hi-lo).clamp(min=1e-8)/Lv
    mn = (((mn-lo)/d).round().clamp(0, Lv))*d+lo
    return ((x-mn)/s).round().clamp(0, 2**bits-1)*s+mn
def _asym(x, bits, rdim):
    mn = x.amin(rdim, keepdim=True); mx = x.amax(rdim, keepdim=True)
    s = (mx-mn).clamp(min=1e-8)/(2**bits-1)
    return ((x-mn)/s).round().clamp(0, 2**bits-1)*s+mn
def _fp8(x):                                  # FP8 baseline: E4M3 with one fp16 absmax scale per (token, head)
    s = x.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 448.0
    return (x / s).to(torch.float8_e4m3fn).float() * s
def kivi_k(k, bits, gs=32, R=128):              # k:(B,T,H,D) per-channel group32, recent R fp16
    B, T, H, D = k.shape
    if T <= R: return k
    Tq = T-R; kq, kr = k[:, :Tq], k[:, Tq:]
    pad = (gs-Tq % gs) % gs
    if pad: kq = torch.cat([kq, kq[:, -1:].expand(B, pad, H, D)], 1)
    ng = kq.shape[1]//gs
    g = _asym(kq.view(B, ng, gs, H, D), bits, 2)
    return torch.cat([g.view(B, ng*gs, H, D)[:, :Tq], kr], 1)
def kivi_v(v, bits, R=128):                      # v:(B,T,H,D) per-token, recent R fp16
    B, T, H, D = v.shape
    if T <= R: return v
    Tq = T-R; vq, vr = v[:, :Tq], v[:, Tq:]
    return torch.cat([_asym(vq, bits, -1), vr], 1)
def kvquant_k(k, bits=2, sp=0.01):               # KVQuant-core: pre-RoPE per-channel(over T) uniform
    # k:(B,HKV,T,hd). per-channel asym scale + dense-sparse top-sp outliers fp16. (NUQ/Fisher omitted)
    lo = torch.quantile(k.float(), sp/2, dim=2, keepdim=True)        # per (B,H,d) over tokens
    hi = torch.quantile(k.float(), 1-sp/2, dim=2, keepdim=True)
    outlier = (k < lo) | (k > hi)
    rng = (hi-lo).clamp(min=1e-8)
    xn = ((k-lo)/rng).clamp(0, 1)
    q = (xn*(2**bits-1)).round()/(2**bits-1)*rng + lo               # uniform per-channel
    return torch.where(outlier, k, q)
def fit_cent(x, nlev, iters=20):                 # 1D k-means per stream. x:(G,N)->cent:(G,nlev)
    qs = torch.linspace(0, 1, nlev+2, device=x.device)[1:-1]
    cent = torch.quantile(x, qs, dim=1).T.contiguous()
    for _ in range(iters):
        a = (x.unsqueeze(1)-cent.unsqueeze(2)).abs().argmin(1)
        for kk in range(nlev):
            m = (a == kk).float(); den = m.sum(1)
            cent[:, kk] = torch.where(den > 0, (x*m).sum(1)/den.clamp(min=1), cent[:, kk])
    return cent
def nuq_v(v, cb):                                # V per-token NUQ: normalize per token (hd) -> shared codebook -> denorm
    zp = (v.amax(-1, keepdim=True)+v.amin(-1, keepdim=True))/2
    dist = ((v.amax(-1, keepdim=True)-v.amin(-1, keepdim=True))/2).clamp(min=1e-8)
    nm = (v-zp)/dist                                                          # ~[-1,1] per token
    return cb[(nm.unsqueeze(-1)-cb).abs().argmin(-1)]*dist + zp              # cb:(nlev,)
def asym_perbits(v, bpt):                        # V per-token asym with PER-TOKEN bit count. v:(H,T,hd), bpt:(H,T)->deq
    mn = v.amin(-1, keepdim=True); mx = v.amax(-1, keepdim=True)
    nl = (2.0**bpt).unsqueeze(-1)                                             # (H,T,1) levels
    s = (mx-mn).clamp(min=1e-8)/(nl-1).clamp(min=1)
    q = torch.minimum(((v-mn)/s).round().clamp(min=0), nl-1)
    return q*s+mn
def alloc_v_bits(imp, bbar, lo=1, hi=4):         # water-fill V bits ACROSS TOKENS by importance. imp:(H,T)->bits
    log2s = 0.5*torch.log2(imp.clamp(min=1e-20)); tgt = bbar*imp.shape[-1]   # per-head budget
    H = imp.shape[0]                                                          # PER-HEAD bisection, fully on-device (no host sync)
    clo = torch.full((H, 1), -60.0, device=imp.device); chi = torch.full((H, 1), 60.0, device=imp.device)
    for _ in range(30):
        mid = (clo+chi)/2; b = (log2s+mid).round().clamp(lo, hi)
        over = (b.sum(-1, keepdim=True) > tgt)                                # (H,1) bool tensor, NO .all/.item sync
        chi = torch.where(over, mid, chi); clo = torch.where(over, clo, mid)
    return (log2s+(clo+chi)/2).round().clamp(lo, hi).long()
VBHIST=[0]
def vattn_quant(v, q, K, vbits):                 # attn-weighted V: bits per token by attn-received (avg=vbits)
    T = v.shape[-2]; bb = torch.arange(T, device=v.device)//args.block
    ng = torch.zeros(T, T, device=v.device); ng[~(bb[None, :] <= bb[:, None])] = float('-inf')  # block-causal
    A = ((q @ K.repeat_interleave(GS, 0).transpose(-1, -2))*SC + ng).softmax(-1)   # (HQ,T,T) post-RoPE attn
    imp = (A.view(HKV, GS, T, T)**2).sum(1).sum(1)                                 # (HKV,T) attention-received²
    # the plain rule funds the SINK first (it receives the most attention), but the sink's V
    # is norm-suppressed ~0.1x so those bits buy nothing — the candidate reason the
    # attention-weighted V ablation backfired on long reasoning (MATH .472 -> .398).
    if args.vattnmode == 'nosink':
        imp[:, 0] = imp[:, 1:].median(dim=-1).values                               # sink gets AVERAGE bits, not max
    elif args.vattnmode == 'norm':
        imp = imp * (v[0].float().norm(dim=-1)**2)                                 # contribution ~ a_i * ||dv_i||
    bpt = alloc_v_bits(imp, float(vbits), args.vattnlo, vbits+2)                              # per-token bits, avg=vbits
    if VBHIST[0] < 3:                                  # log the realised bit histogram once
        import collections as _c
        _h=_c.Counter(bpt.flatten().tolist())
        print(f'  [vattn bits] mode={args.vattnmode} lo={args.vattnlo} hist='+str(sorted(_h.items())), flush=True)
        VBHIST[0]+=1
    return asym_perbits(v[0], bpt)[None]                                          # (1,HKV,T,hd) (B=1)
def kvquant_k_nuq(k, cent, sp=0.01):             # k:(B,HKV,T,hd), cent:(HKV*HD,nlev): per-ch NUQ + outlier fp16
    B, H, T, D = k.shape
    lo = torch.quantile(k, sp/2, dim=2, keepdim=True); hi = torch.quantile(k, 1-sp/2, dim=2, keepdim=True)
    outlier = (k < lo) | (k > hi)
    stc = k.permute(0, 1, 3, 2).reshape(B*H*D, T)                   # (BG,T) per channel
    ce = cent.repeat(B, 1)                                          # (BG,nlev)
    a = (stc.unsqueeze(1)-ce.unsqueeze(2)).abs().argmin(1)          # (BG,T)
    q = torch.gather(ce, 1, a).reshape(B, H, D, T).permute(0, 1, 3, 2)
    return torch.where(outlier, k, q)
def round_pole(x, poles):                        # x:(...) -> nearest of poles (P,) (KVQuant round_to_nearest_pole)
    return poles[(x.unsqueeze(-1)-poles).abs().argmin(-1)]
def wfit_cent(x, w, nlev, iters=50):             # Fisher-WEIGHTED 1D k-means (KVQuant NUQ). x,w:(N,) -> (nlev,)
    if x.numel() > 4_000_000:                    # cap for torch.quantile (16.7M limit) + speed; integer stride (float idx overflows)
        st = x.numel()//4_000_000 + 1; x = x[::st]; w = w[::st]
    cent = torch.quantile(x, torch.linspace(0, 1, nlev+2, device=x.device)[1:-1])
    for _ in range(iters):
        a = (x.unsqueeze(1)-cent.unsqueeze(0)).abs().argmin(1)
        for k in range(nlev):
            m = (a == k).float(); ww = (w*m).sum()
            if ww > 0: cent[k] = (x*w*m).sum()/ww
    return cent.sort().values
def kvquant_k_fisher(k, cb, zp, dist):           # KVQuant faithful: per-ch normalize -> shared codebook -> denorm, |norm|>1 fp16
    B, H, T, D = k.shape
    kc = k.permute(0, 1, 3, 2).reshape(B, H*D, T)                   # (B,G,T) per channel
    z = zp.view(1, -1, 1); d = dist.view(1, -1, 1)
    norm = (kc - z) / d
    outl = norm.abs() > 1
    q = round_pole(norm, cb)*d + z
    q = torch.where(outl, kc, q).reshape(B, H, D, T).permute(0, 1, 3, 2)
    return q
AUX = {}
def _load_aux():                                     # KVQF/VLUT/SVDQ from source-verified calib
    if not AUX:
        _x = torch.load(_os.environ.get('AUXCALIB', 'aux_calib_kvq_svdq.pt'), map_location='cuda')
        for k in _x: AUX[k] = _x[k]
        print(f"[aux] aux calib loaded (KVQF {len(AUX['KVQF'])} / VLUT {len(AUX['VLUT'])} / "
              f"SVDQ {len(AUX['SVDQ'])} layers)", flush=True)
    return AUX
def kvquant_v_fisher(v, cb):                         # official V path: per-token DYNAMIC 1% trim ->
    lo = v.float().quantile(0.005, dim=-1, keepdim=True)   # normalize -> layer LUT -> outliers fp16
    hi = v.float().quantile(0.995, dim=-1, keepdim=True)   # (simquant: pertoken, dynamicquantization=True)
    zp = (hi+lo)/2; dist = ((hi-lo)/2).clamp(min=1e-8)
    norm = (v.float() - zp)/dist
    outl = norm.abs() > 1
    q = round_pole(norm, cb)*dist + zp
    return torch.where(outl, v.float(), q)
SVDQ_SCHED = {'svdqm2': (8,4,2,1,1,0,0,0), 'svdqm1': (4,2,1,1,0,0,0,0), 'svdqmnd': (8,4,2,2,2,2,2,2)}
# SVDq as specified in its paper (Table 2). The svdqm_* schedules above are graded-tier development variants,
# not SVDq.
# Paper spec: per-SEQUENCE prefill SVD (per-layer heads-concat, per-channel mean centered, pre-RoPE),
# 8 equal latent groups, schedules b2=(8,4,4,0,0,0,0,0) b3=(8,4,4,4,2,2,0,0), per-latent-channel
# asym min/max over the prefill tokens (no token groups), V kept BF16, decode K left fp16 (decode
# handling unspecified in the paper -- generous choice, disclosed).
SVDQP_SCHED = {'svdqp2': (8,4,4,0,0,0,0,0), 'svdqp3': (8,4,4,4,2,2,0,0),
               'svdqv2': (8,4,4,0,0,0,0,0), 'svdqv15': (8,4,0,0,0,0,0,0),
               'svdqv10': (8,0,0,0,0,0,0,0), 'svdqv05': (4,0,0,0,0,0,0,0),
               'svdqv10w': (2,2,0,0,0,0,0,0)}  # SVDq's BEST schedule at K=1.5 by our
                                              # theory (256 dims @2b vs svdqv10's 128 @8b; payload
                                              # 0.5 + basis 16*256/4100 = 1.0 -> K tot 1.5, matched).
                                              # Named svdqv* so both dispatch sites (V-path allowlist
                                              # below, pre-RoPE K branch) match it -- the first
                                              # attempt used 'svdqw2' and every task was
                                              # (correctly) REFUSED as matching no quantizer branch.
SVDQP_SCHED['svdqv10wg'] = (2,2,0,0,0,0,0,0)  # CONTROL: same allocation as svdqv10w, but
SVDQP_TOKGRP = {'svdqv10wg': 32}              # 32-token group scales instead of SVDq's global
# per-channel prefill min/max -- isolates SCALE GRANULARITY from allocation (changed both).
# ACCOUNTING (T=4100, D=1024, bits per K value): group scales cost 2 fp16 per (kept channel,
# 32-tok group) = 32*(T/32)*kappa bits = kappa/D = 0.250, vs 32*kappa/(T*D) = 0.002 for global.
# payload 0.500 + basis 16*256/4100 = 1.000 + scales 0.250 -> K tot 1.75 (KV 2.01).
# NOT budget-matched to svdqv10's 1.50 and CANNOT be: 8 equal groups admit only kappa in {128,256},
# giving 0.875 (strictly worse than svdqv10 in both dims and bits) or 1.75. Run at 1.75 with the
# handicap DISCLOSED -- the control gets +0.25 bit over svdqv10 and sits above QuMix's KV 1.83 too,
# so a loss here is a fortiori. A win would need a budget-matched follow-up before it counts.
def svdqp_prefill(k, sched, tokgrp=0):              # k:(1,H,T,hd) pre-RoPE prefill
    B, H, T, Dh = k.shape; D = H*Dh
    Kf = k[0].permute(1, 0, 2).reshape(T, D).float()
    mean = Kf.mean(0)
    X = Kf - mean
    cov = X.T @ X
    try:
        w, U = torch.linalg.eigh(cov)               # ascending
    except torch._C._LinAlgError:                   # rare ill-conditioned prefill: fp64 + jitter
        c64 = cov.double(); c64 += 1e-4*torch.eye(c64.shape[0], device=c64.device, dtype=c64.dtype)
        w, U = torch.linalg.eigh(c64); w = w.float(); U = U.float()
    U = U.flip(-1)                                  # descending sigma
    C = X @ U
    gsz = D // 8
    for gi, b in enumerate(sched):
        sl = slice(gi*gsz, (gi+1)*gsz)
        if b == 0: C[:, sl] = 0; continue
        if b >= 16: continue
        seg = C[:, sl]
        if tokgrp:                                  # per-32-token-group asym (QuMix's scaler)
            G_ = tokgrp; n = (T//G_)*G_; W = seg.shape[1]
            if n:
                g = seg[:n].reshape(-1, G_, W)
                mn = g.amin(1, keepdim=True); mx = g.amax(1, keepdim=True)
                sc = (mx - mn).clamp(min=1e-8) / (2**b - 1)
                seg[:n] = (((g - mn)/sc).round()*sc + mn).reshape(n, W)
            if n < T:                               # tail group: asym over the remaining tokens
                g = seg[n:]
                mn = g.amin(0, keepdim=True); mx = g.amax(0, keepdim=True)
                sc = (mx - mn).clamp(min=1e-8) / (2**b - 1)
                seg[n:] = ((g - mn)/sc).round()*sc + mn
            C[:, sl] = seg
        else:                                       # SVDq paper spec: global per-channel min/max
            mn = seg.amin(0, keepdim=True); mx = seg.amax(0, keepdim=True)
            sc = (mx - mn).clamp(min=1e-8) / (2**b - 1)
            C[:, sl] = ((seg - mn)/sc).round()*sc + mn
    Xq = C @ U.T + mean
    return Xq.reshape(T, H, Dh).permute(1, 0, 2)[None]   # paper-style, mean = 2 / 1
def svdq_k(k, U, mean, sched, gs=32):                # SVDq (2502.15304, from-paper: no public code):
    B, H, T, Dh = k.shape; D = H*Dh                  # layer-level heads-concat SVD latent, 8 equal
    X = k.permute(0, 2, 1, 3).reshape(B, T, D) - mean   # groups, monotone tier bits incl. 0-drops
    C = X @ U
    gsz = D // 8
    n = (T//gs)*gs
    for gi, b in enumerate(sched):
        sl = slice(gi*gsz, (gi+1)*gsz)
        if b == 0: C[..., sl] = 0; continue
        seg = C[..., sl]
        if n:
            g = seg[:, :n].reshape(B, -1, gs, gsz)
            mn = g.amin(2, keepdim=True); mx = g.amax(2, keepdim=True)
            sc = (mx-mn).clamp(min=1e-8)/(2**b-1)
            seg[:, :n] = (((g-mn)/sc).round()*sc+mn).reshape(B, n, gsz)
        if n < T:
            g = seg[:, n:]; mn = g.amin(1, keepdim=True); mx = g.amax(1, keepdim=True)
            sc = (mx-mn).clamp(min=1e-8)/(2**b-1)
            seg[:, n:] = ((g-mn)/sc).round()*sc+mn
        C[..., sl] = seg
    Xq = C @ U.T + mean
    return Xq.reshape(B, T, H, Dh).permute(0, 2, 1, 3)
def alloc_bits(sig2, bbar):
    log2s = 0.5*torch.log2(sig2.clamp(min=1e-20)); lo, hi = -60.0, 60.0; tgt = bbar*sig2.shape[-1]
    for _ in range(60):
        c = (lo+hi)/2; b = (log2s+c).round().clamp(0, 16)
        b = torch.where(b == 1, torch.full_like(b, 2), b)
        b = torch.where(b > 8, torch.full_like(b, 16), b)
        if b.sum() > tgt: hi = c
        else: lo = c
    return b.long()
def alloc_bits_min1(sig2, bbar):                 # (b): floor 1 bit, NO 0-bit drops, allow b=1
    log2s = 0.5*torch.log2(sig2.clamp(min=1e-20)); lo, hi = -60.0, 60.0; tgt = bbar*sig2.shape[-1]
    for _ in range(60):
        c = (lo+hi)/2; b = (log2s+c).round().clamp(1, 16)
        b = torch.where(b > 8, torch.full_like(b, 16), b)
        if b.sum() > tgt: hi = c
        else: lo = c
    return b.long()
def alloc_bits_alpha(sig2, bbar, alpha):         # QuMix-lite: water-fill over a constrained bit alphabet
    A = torch.tensor(sorted(alpha), dtype=torch.float, device=sig2.device)
    mid = (A[1:]+A[:-1])/2                       # nearest-in-alphabet via bin midpoints
    log2s = 0.5*torch.log2(sig2.clamp(min=1e-20)); lo, hi = -60.0, 60.0; tgt = bbar*sig2.shape[-1]
    for _ in range(60):
        c = (lo+hi)/2; b = A[torch.bucketize(log2s+c, mid)]
        if b.sum() > tgt: hi = c
        else: lo = c
    return b.long()

# ========== [QuMix] calibration capture: queries and keys (pre-RoPE with --prerope) of every attention call while decoding ==========
attn_mod = None
for n, m in model.named_modules():
    if type(m).__name__.endswith('Attention'): attn_mod = sys.modules[type(m).__module__]; break
print(f"attn module: {attn_mod.__name__}", flush=True)
_orig_rope = attn_mod.apply_rotary_pos_emb
GK, CQ, CNT, NT, KSAMP, KSCNT = {}, {}, [0], [0], {}, {}
KS = {}                                                            # per-head key SUM for centered-G_K variant
NEEDK = ('kvqn' in args.arms.split(','))                                 # capture pre-RoPE K for NUQ centroids
def cap_rope(q, k, cos, sin, *a, **kw):
    qr, kr = _orig_rope(q, k, cos, sin, *a, **kw)
    i = CNT[0] % L; CNT[0] += 1
    if PREROPE: kk = k[0].detach().float(); qq = q[0].detach().float()   # (c) pre-RoPE input
    else: kk = kr[0].detach().float(); qq = qr[0].detach().float()       # post-RoPE output
    T = kk.shape[1]
    GK[i] = GK.get(i, 0) + torch.einsum('htd,hte->hde', kk, kk)
    KS[i] = KS.get(i, 0) + kk.sum(1)                               # (HKV,hd)
    qg = qq.view(HKV, GS, T, HD).reshape(HKV, GS*T, HD)
    CQ[i] = CQ.get(i, 0) + torch.einsum('htd,hte->hde', qg, qg)
    if PREROPE and NEEDK and KSCNT.get(i, 0) < 512:                      # accumulate pre-RoPE K (capped)
        KSAMP.setdefault(i, []).append(k[0].detach().float()); KSCNT[i] = KSCNT.get(i, 0)+T
    if i == 0: NT[0] += T
    return qr, kr

def build_calib_prompts(name, n):
    if name == 'wikitext':                                       # generic corpus, n×512-tok segments
        wt = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        wids = tok("\n".join(t for t in wt['text'] if t.strip()), return_tensors='pt', add_special_tokens=False)['input_ids'][0]
        return [tok.decode(wids[i*512:(i+1)*512]) for i in range(n)]
    ds = load_dataset('gsm8k', 'main', split=f'train[:{n}]')     # in-domain (default)
    return [tok.apply_chat_template([dict(role="user", content=s["question"]+"\nPlease reason step by step, and put your final answer within \\boxed{}.\n")],
                                    add_generation_prompt=True, tokenize=False) for s in ds]
CALIB_PROMPTS = [] if args.loadcalib else build_calib_prompts(args.calibset, args.ncalib)
print(f"calib set = {args.calibset} ({len(CALIB_PROMPTS)} items){' [LOADCALIB]' if args.loadcalib else ''}", flush=True)
VSAMP = {}; VSCNT = {}; _vhooks = []                                          # V capture for V-NUQ codebook
if args.vnuq and not args.loadcalib:
    def _mkvh(li):
        def h(mod, inp, out):
            if VSCNT.get(li, 0) < 512:
                v = out.view(out.shape[0], out.shape[1], HKV, HD)[0].permute(1, 0, 2).detach().float()  # (HKV,T,hd)
                VSAMP.setdefault(li, []).append(v); VSCNT[li] = VSCNT.get(li, 0)+v.shape[1]
        return h
    for _n, _m in model.named_modules():
        _mt = re.search(r'layers\.(\d+)\..*v_proj$', _n)
        if _mt: _vhooks.append(_m.register_forward_hook(_mkvh(int(_mt.group(1)))))
attn_mod.apply_rotary_pos_emb = cap_rope
for sstr in CALIB_PROMPTS:
    enc = tok.batch_encode_plus([sstr], return_tensors='pt', add_special_tokens=False)
    enc = {k: v.to('cuda') for k, v in enc.items()}
    G.block_diffusion_generate(model, prompt=enc, mask_id=MASK, gen_length=args.calibgen, block_length=args.block,
        denoising_steps=args.steps, temperature=args.temp, top_k=1, top_p=1.0,
        remasking_strategy="low_confidence_static", confidence_threshold=0.9, eb_threshold=0.0,
        stopping_criteria_idx=EOS)
attn_mod.apply_rotary_pos_emb = _orig_rope
for _h in _vhooks: _h.remove()
ntok = NT[0]
print(f"calibrated on {ntok} tok ({args.ncalib} prompts)", flush=True)
VCB = {}                                                                      # V-NUQ codebook per layer (per-token normalized)
if args.vnuq and not args.loadcalib:
    nlev = 2**VBITS
    for i in range(L):
        v = torch.cat(VSAMP[i], dim=1)                                        # (HKV,Ttot,hd)
        zp = (v.amax(-1, keepdim=True)+v.amin(-1, keepdim=True))/2
        dist = ((v.amax(-1, keepdim=True)-v.amin(-1, keepdim=True))/2).clamp(min=1e-8)
        VCB[i] = fit_cent(((v-zp)/dist).reshape(1, -1), nlev)[0]              # (nlev,) shared codebook
    del VSAMP
    print(f"V-NUQ codebook fit ({len(VCB)} layers, {nlev} levels)", flush=True)
CENTK = {}
if NEEDK and not args.loadcalib:                                         # fit per-channel NUQ centroids (2-bit)
    for i in range(L):
        ks = torch.cat(KSAMP[i], dim=1)                                  # (HKV,Ttot,hd)
        stc = ks.permute(0, 2, 1).reshape(HKV*HD, -1)                    # (HKV*HD,Ttot) per channel
        lo = stc.quantile(0.005, dim=1, keepdim=True); hi = stc.quantile(0.995, dim=1, keepdim=True)
        stc = torch.maximum(torch.minimum(stc, hi), lo)                  # clip outliers (match dense-sparse inliers)
        CENTK[i] = fit_cent(stc, 4)                                      # (HKV*HD,4)
    del KSAMP
    print(f"kvq-NUQ centroids fit ({len(CENTK)} layers)", flush=True)

KVQF = {}                                                                # KVQuant faithful: Fisher-weighted NUQ
if 'kvqf' in args.arms.split(',') and not args.loadcalib:
    kp_mods = {}
    for n, m in model.named_modules():
        mt = re.search(r'layers\.(\d+)\..*k_proj$', n)
        if mt: kp_mods[int(mt.group(1))] = m
    kacts = {}
    def _mk(i):
        def h(mod, inp, out): kacts[i] = out                             # keep graph tensor (k_proj output = pre-RoPE K act)
        return h
    hks = [kp_mods[i].register_forward_hook(_mk(i)) for i in kp_mods]
    KVv, KFi = {}, {}
    torch.set_grad_enabled(True)      # global no-grad broke this Fisher capture; 512-tok
                                      # calib prompts, so the 8K-prefill OOM it guarded against is absent
    for sstr in CALIB_PROMPTS:
        ids = tok.batch_encode_plus([sstr], return_tensors='pt', add_special_tokens=False)['input_ids'][:, :512].to('cuda')
        kacts.clear()
        if args.kvqfnative:                                              # fairness: SDAR reads ones_like as FULL-BIDIR
            _bb = torch.arange(ids.shape[1], device=ids.device) // args.block
            _am = (_bb[None, :] <= _bb[:, None])[None]                   # (1,T,T) bool block-causal
            assert not bool(_am.all()), 'kvqfnative mask degenerated to all-ones'
            out = model(input_ids=ids, labels=ids, attention_mask=_am)
        else:
            out = model(input_ids=ids, labels=ids, attention_mask=torch.ones_like(ids))
        order = sorted(kacts); grads = torch.autograd.grad(out.loss, [kacts[i] for i in order])  # (∂L/∂K_act), no param.grad
        for j, i in enumerate(order):
            kv = kacts[i].detach()[0].float(); fi = grads[j][0].float()**2     # (T, HKV*HD) Fisher=(∂L/∂K)^2
            KVv[i] = torch.cat([KVv[i], kv], 0) if i in KVv else kv
            KFi[i] = torch.cat([KFi[i], fi], 0) if i in KFi else fi
    for h in hks: h.remove()
    torch.set_grad_enabled(False)                                        # restore eval-only mode
    for i in KVv:                                                        # per-ch normalize -> shared Fisher-weighted codebook
        data, fish = KVv[i], KFi[i]                                      # (Ttot, G=HKV*HD)
        nt = data.shape[0]; dsub = data if nt <= 6000 else data[torch.linspace(0, nt-1, 6000).long()]
        lo = dsub.quantile(0.005, dim=0); hi = dsub.quantile(0.995, dim=0)   # KVQuant: range on OUTLIER-TRIMMED data (1%)
        zp = (hi+lo)/2; dist = ((hi-lo)/2).clamp(min=1e-8)
        norm = (data - zp)/dist; inl = norm.abs() <= 1                   # ~1% now outliers (|norm|>1 -> fp16 at apply)
        cb = wfit_cent(norm[inl].flatten(), fish[inl].flatten(), 4)      # 2-bit shared codebook on TIGHT inlier range
        KVQF[i] = dict(cb=cb, zp=zp, dist=dist)
    del KVv, KFi; torch.cuda.empty_cache()
    print(f"kvqf Fisher-NUQ fit ({len(KVQF)} layers)", flush=True)

# ========== [QuMix] query-weighted basis and bit allocation per layer (or load a calibration file) ==========
if args.loadcalib:
    _d = torch.load(args.loadcalib, map_location='cuda')
    PROJ, CENTK, KVQF = _d['PROJ'], _d['CENTK'], _d['KVQF']; VCB = _d.get('VCB', {})
    VPROJ = torch.load(args.vqmx, map_location='cuda')['VPROJ'] if args.vqmx else {}
    VHB = {i: t for i, t in enumerate(torch.load(args.vhead)[args.vheadtag])} if args.vhead else {}
    if args.vqmx: print(f"[vqmx] V-QuMix calib {args.vqmx} tag={args.vqmxtag} vgs={args.vgs} ({len(VPROJ)} layers)", flush=True)
    # development arms trunc/unif: reuse the qmx2 query-weighted basis (Wqw/Wrw), change only the bit allocation
    _rtr = max(1, int(2*HD/16))                                   # truncation: top-rtr dirs 16b, rest 0 (avg 2b)
    for _i in PROJ:
        if 'qmx2' in PROJ[_i] and 'trunc' not in PROJ[_i]:
            _Wq = PROJ[_i]['qmx2']['Wqw']; _Wr = PROJ[_i]['qmx2']['Wrw']
            _Btr = torch.zeros(HKV, HD, device='cuda'); _Btr[:, :_rtr] = 16
            PROJ[_i]['trunc'] = dict(Wqw=_Wq, Wrw=_Wr, groups=[(16, (_Btr == 16).float().unsqueeze(1))])
            PROJ[_i]['unif'] = dict(Wqw=_Wq, Wrw=_Wr, groups=[(2, torch.ones(HKV, 1, HD, device='cuda'))])
    print(f"loaded calib from {args.loadcalib} (PROJ {len(PROJ)} / CENTK {len(CENTK)} / KVQF {len(KVQF)} layers) +trunc/unif", flush=True)
else:
    PROJ = {}
    for i in range(L):
        cq = CQ[i]; jit = 1e-4*cq.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None]*torch.eye(HD, device='cuda')[None]
        Lc = torch.linalg.cholesky(cq+jit); Minv = torch.linalg.inv(Lc)
        M = Lc.transpose(-1, -2)@GK[i]@Lc; M = 0.5*(M+M.transpose(-1, -2))
        mev, mvec = torch.linalg.eigh(M); mev, mvec = mev.flip(-1), mvec.flip(-1)
        sig = (mev.clamp(min=0)/ntok).sqrt()                          # (HKV,hd)
        Wqw = ((Lc@mvec)/sig.clamp(min=1e-8).unsqueeze(1))            # (HKV,hd,hd): c = k@Wqw
        Wrw = ((Minv.transpose(-1, -2)@mvec)*sig.unsqueeze(1))        # k = c@Wrw
        P = dict()
        for bbar in (2, 4):
            BITS = torch.stack([alloc_bits(sig[h]**2, bbar) for h in range(HKV)])  # (HKV,hd)
            groups = [(int(b), (BITS == b).float().unsqueeze(1)) for b in BITS.unique().tolist() if b > 0]
            P[f'qmx{bbar}'] = dict(Wqw=Wqw, Wrw=Wrw, groups=groups)
        if args.ladder:                                # sub-2bit budget alphabet (builder verified
            for _bb, _tag in ((1.5, 'qmx15'), (1.0, 'qmx10'), (0.75, 'qmx075'),   # against wiki_calib_32k_
                              (0.5, 'qmx05'), (0.33, 'qmx033')):                   # cpair2_lowbit: 36/36 layers)
                BL = torch.stack([alloc_bits(sig[h]**2, _bb) for h in range(HKV)])
                gl = [(int(b), (BL == b).float().unsqueeze(1)) for b in BL.unique().tolist() if b > 0]
                P[_tag] = dict(Wqw=Wqw, Wrw=Wrw, groups=gl)
        B1 = torch.stack([alloc_bits_min1(sig[h]**2, 2) for h in range(HKV)])       # (b): no 0-bit drops
        g1 = [(int(b), (B1 == b).float().unsqueeze(1)) for b in B1.unique().tolist() if b > 0]
        P['qmx2b1'] = dict(Wqw=Wqw, Wrw=Wrw, groups=g1)
        # ---- development variants: hardware-aligned bit alphabets, same basis and budget ----
        for _tag, _alpha in (('qmx2l', (0, 2, 4, 8, 16)), ('qmx2m', (0, 2, 4, 16))):
            BA = torch.stack([alloc_bits_alpha(sig[h]**2, 2.0, _alpha) for h in range(HKV)])
            ga = [(int(b), (BA == b).float().unsqueeze(1)) for b in BA.unique().tolist() if b > 0]
            P[_tag] = dict(Wqw=Wqw, Wrw=Wrw, groups=ga)
        P['sig2'] = sig**2                        # store spectrum (future re-allocation without recapture)
        # ---- ABLATION svdk: K-variance eigenbasis (NO query metric) — isolates the 'query' claim ----
        gev, gvec = torch.linalg.eigh(GK[i]); gev, gvec = gev.flip(-1), gvec.flip(-1)  # eig of K-cov only
        sigk = (gev.clamp(min=0)/ntok).sqrt()                        # K's own std per eigendir
        BK = torch.stack([alloc_bits(sigk[h]**2, 2) for h in range(HKV)])             # water-fill by K-variance
        gk = [(int(b), (BK == b).float().unsqueeze(1)) for b in BK.unique().tolist() if b > 0]
        P['svdk'] = dict(Wqw=gvec, Wrw=gvec, groups=gk)              # orthonormal: c=k@gvec, k=c@gvecᵀ
        # ---- CENTERED-G_K variant: mean key removed before basis + allocation ----
        # measured 72% of K energy to be the common (mean) component, which group
        # zero-points can absorb; this tests whether the UNcentered basis/alloc wastes bits on it.
        # C_q stays UNcentered (the damage metric must include the mean query). Only G_K changes.
        km = KS[i]/ntok                                              # (HKV,hd) mean pre-RoPE key
        GKc = GK[i] - ntok*torch.einsum('hd,he->hde', km, km)
        Mc = Lc.transpose(-1, -2)@GKc@Lc; Mc = 0.5*(Mc+Mc.transpose(-1, -2))
        cev, cvec = torch.linalg.eigh(Mc); cev, cvec = cev.flip(-1), cvec.flip(-1)
        sigc = (cev.clamp(min=0)/ntok).sqrt()
        Wqwc = ((Lc@cvec)/sigc.clamp(min=1e-8).unsqueeze(1))
        Wrwc = ((Minv.transpose(-1, -2)@cvec)*sigc.unsqueeze(1))
        BC = torch.stack([alloc_bits(sigc[h]**2, 2) for h in range(HKV)])
        gc2 = [(int(b), (BC == b).float().unsqueeze(1)) for b in BC.unique().tolist() if b > 0]
        P['qmx2c'] = dict(Wqw=Wqwc, Wrw=Wrwc, groups=gc2)
        P['kmean'] = km                                          # (HKV,hd) pre-RoPE mean key (explicit-mean arm)
        PROJ[i] = P
    del GK, CQ
if args.savecalib:
    torch.save({'PROJ': PROJ, 'CENTK': CENTK, 'KVQF': KVQF, 'VCB': VCB}, args.savecalib)
    print(f"saved calib to {args.savecalib}", flush=True)
tb = sum(float(sum(int(b)*m.sum() for b, m in PROJ[i]['qmx2']['groups'])/HKV) for i in range(L))/L
print(f"qmx2 avg K bits/head/tok ~ {tb:.1f} | basis built for {L} layers", flush=True)

# ---- isotropic-noise control arm (noiseNNN, NNN = dose x100 of KIVI's own score noise) ----
# NOTE: this MUST sit outside the loadcalib if/else above — an earlier version was placed between
# them at column 0, which re-bound the `else:` (the build-calib-from-scratch path) to this check
# and made every `--loadcalib` run without a noise arm crash on CQ[0]. Keep it here.
NSIG = {}
if any(a.startswith('noise') for a in args.arms.split(',')):
    if not args.noiseprof:
        print("REFUSED: a noiseNNN arm needs --noiseprof (per-layer sigma table)."); sys.exit(1)
    _npf = torch.load(args.noiseprof, map_location='cuda', weights_only=False)
    _sg = _npf['sigma_scale1']
    NSIG = {i: float(_sg[i]) for i in range(len(_sg))}
    _st2 = _os.stat(args.noiseprof)
    print(f"[fingerprint] noiseprof: {args.noiseprof} size={_st2.st_size} tag={_npf.get('tag')} "
          f"block={_npf.get('block')} L={len(_sg)}", flush=True)
    print(f"noise arm armed: sigma@scale1.0 per layer, L0={NSIG[0]:.5f} L18={NSIG[18]:.5f} "
          f"L35={NSIG[max(NSIG)]:.5f} (reproduces KIVI's per-layer RMS|q^T dk|)", flush=True)
if args.calibonly:
    import statistics as _st
    for tag in ('qmx2', 'qmx4', 'qmx2c'):                                  # kept-rank = #dirs with >0 bits (Wrw is hd×r)
        rk = [sum(m.sum().item() for b, m in PROJ[i][tag]['groups'])/HKV for i in range(L)]
        nz = [sum(int(b)*m.sum().item() for b, m in PROJ[i][tag]['groups'])/HKV for i in range(L)]
        print(f"{tag}: kept-rank/head mean={_st.mean(rk):.1f}/{HD} (dropped {HD-_st.mean(rk):.1f}) "
              f"min={min(rk):.0f} max={max(rk):.0f} | avg bits/head={_st.mean(nz):.1f} | rot cost vs full ~{_st.mean(rk)/HD:.2f}x", flush=True)
    sys.exit(0)

# ========== [QuMix] key quantization in the query-weighted basis: qmx_k(..., tokq=True) ==========
ARM = ['base']; ECNT = [0]
SV_Q = {}; SV_K = {}; SC = HD**-0.5                              # post-RoPE q,k capture for attention-weighted V
def fq_tok(x, bits, gs=32):                                       # per-dim × 32-tok asym scales; x:(B,H,T,R)
    B, H, T, R = x.shape; pad = (gs - T % gs) % gs
    xp = torch.cat([x, x[:, :, -1:].expand(B, H, pad, R)], 2) if pad else x
    xg = xp.view(B, H, -1, gs, R)
    mn = xg.amin(3, keepdim=True); mx = xg.amax(3, keepdim=True)
    s = (mx-mn).clamp(min=1e-8)/(2**bits-1)
    return (((xg-mn)/s).round().clamp(0, 2**bits-1)*s+mn).view(B, H, -1, R)[:, :, :T]
_VQC = {}                                                         # per-P vectorized allocation cache (no group loop, ~7x faster)
def _vq(P):
    pid = id(P)
    if pid not in _VQC:
        mk0 = P['groups'][0][1]
        qm = torch.zeros_like(mk0).unsqueeze(1); k16 = torch.zeros_like(mk0); cov = torch.zeros_like(mk0)
        for b, mk in P['groups']:
            cov = cov + mk
            if b >= 16: k16 = k16 + mk
            else: qm = qm + (2**b-1)*mk.unsqueeze(1)
        _VQC[pid] = (qm, k16, cov)                                # qm:(H,1,1,D), k16/cov:(H,1,D)
    return _VQC[pid]
def qmx_k(k, P, tokq=False, gs=32):                               # k:(B,HKV,T,hd) -> qmx-quantized
    c = torch.einsum('bhtd,hdr->bhtr', k, P['Wqw'])
    if tokq:                                                      # qmx2t: per-dim x 32-token asym in one pass over all dims (numerically identical to the fq_tok group loop, 3.2x faster)
        qm, k16, cov = _vq(P)
        B, H, T, R = c.shape; pad = (gs - T % gs) % gs
        xp = torch.cat([c, c[:, :, -1:].expand(B, H, pad, R)], 2) if pad else c
        xg = xp.view(B, H, -1, gs, R)
        mn = xg.amin(3, keepdim=True); mx = xg.amax(3, keepdim=True)
        s = (mx-mn).clamp(min=1e-8)/qm.clamp(min=1)
        sbits = P.get('sbits', 16)
        if sbits < 16:                                             # second-level scale quantization: per (h,r) ranges from the first fixT tokens, widened
            Lv = 2**sbits-1; widen = P.get('widen', 1.0); ng = max(1, min(xg.shape[2], P.get('fixT', 256)//gs))
            ls = s.log2(); lo = ls[:, :, :ng].amin(2, keepdim=True)-widen; hi = ls[:, :, :ng].amax(2, keepdim=True)+widen; d = (hi-lo).clamp(min=1e-8)/Lv
            s = 2**((((ls-lo)/d).round().clamp(0, Lv))*d+lo)
            lo = mn[:, :, :ng].amin(2, keepdim=True); hi = mn[:, :, :ng].amax(2, keepdim=True); w = (hi-lo)*widen*0.25; lo = lo-w; hi = hi+w; d = (hi-lo).clamp(min=1e-8)/Lv
            mn = (((mn-lo)/d).round().clamp(0, Lv))*d+lo
        q = (((xg-mn)/s).round().clamp(torch.zeros_like(qm), qm)*s+mn).view(B, H, -1, R)[:, :, :T]
        cqd = torch.where(k16 > 0, c, q)*(cov > 0)
    else:
        cqd = torch.zeros_like(c)
        for b, mk in P['groups']:
            if b >= 16: cqd = cqd + c*mk
            else: cqd = cqd + fq(c*mk, b)
    return torch.einsum('bhtr,hdr->bhtd', cqd, P['Wrw'])
_orig_update = DynamicCache.update
# --- seal-on-commit KIVI, mirroring AR policy -------------------------------
# The default kivi path quantizes ONLY the tensor being appended, which at decode is the 4-token
# committed block -> effective group 4 and ZERO fp16 window, while the AR harness uses
# 32-token groups and keeps R + the incomplete group in fp16. The sealed arms below use the same policy as QuMix:
# kivis<bits> (group 32, R=0) and kivis2r128 (KIVI's 128-token exact residual; the paper's KIVI 2-bit).
SEALED = {}; SEALEDV = {}; LASTT = {}
def _cache_len(cache, layer_idx):
    """tokens already in this layer's cache BEFORE the pending update; 0 == fresh sample."""
    kc = getattr(cache, 'key_cache', None) or getattr(cache, 'layers', None)
    try:
        if kc is None or len(kc) <= layer_idx: return 0
        t = kc[layer_idx]
        t = getattr(t, 'keys', t)
        return 0 if t is None or not hasattr(t, 'shape') else int(t.shape[-2])
    except Exception:
        return 0
# qmx sealed path: buffers filled by quant_rope, consumed here on store_kv=True commits only.
# ========== [QuMix] sealed cache update: quantize each complete 32-token group once, keep the tail exact ==========
PENDK = {}; KPRE = {}; COSB = {}; SINB = {}; KOFF = {}; SEALCNT = [0]
def sealed_qmx_update(self, key, value, layer_idx, cache_kwargs=None):
    """qmx2ts: 32-token groups sealed on commit, pre-RoPE coeff quant then RoPE re-applied.
    Fixes two asymmetries vs the default qmx path: (i) group = block(4) -> 32, no fp16 window
    -> incomplete group stays fp16; (ii) the default path quantizes via apply_rotary on EVERY
    forward, so the current block (never cached) was being quantized too -- KIVI never was."""
    a = ARM[0]                                                   # qmx<budget>ts[<R>]
    _pre, _, _post = a.partition('ts')
    _r = int(_post) if _post.isdigit() else 0
    _key = _pre                                                  # 'qmx2' | 'qmx15' | 'qmx10'
    _fresh = _cache_len(self, layer_idx) == 0                    # empty cache == new sample
    ka, va = _orig_update(self, key, value, layer_idx, cache_kwargs)
    B, H, T, D = ka.shape
    _ex = (ka.clone(), va.clone()) if _fresh else None           # exact tensors at prefill
    if _fresh:                                                   # fix: length heuristic was
        SEALED[layer_idx] = 0; KOFF[layer_idx] = 0               # unreliable when prompt lengths
        KPRE.pop(layer_idx, None); COSB.pop(layer_idx, None); SINB.pop(layer_idx, None)
    pk, pc, ps = PENDK.get(layer_idx, (None, None, None))
    if pk is None: return _ex if _ex is not None else (ka, va)
    KPRE[layer_idx] = pk if layer_idx not in KPRE else torch.cat([KPRE[layer_idx], pk], 2)
    COSB[layer_idx] = pc if layer_idx not in COSB else torch.cat([COSB[layer_idx], pc], 1)
    SINB[layer_idx] = ps if layer_idx not in SINB else torch.cat([SINB[layer_idx], ps], 1)
    GS_ = 32
    g_end = ((T - _r) // GS_) * GS_ if T > _r else 0
    s0 = SEALED.get(layer_idx, 0); off = KOFF.get(layer_idx, 0)
    if g_end <= s0: return _ex if _ex is not None else (ka, va)
    kseg = KPRE[layer_idx][:, :, s0-off:g_end-off]                # (B,HKV,btok,hd) PRE-RoPE
    if _key not in PROJ[layer_idx]:
        raise SystemExit(f"REFUSED: calib has no '{_key}' allocation (need the low-bit calib)")
    kq = qmx_k(kseg.float(), PROJ[layer_idx][_key], tokq=True)
    _, krot = _orig_rope(kq, kq, COSB[layer_idx][:, s0-off:g_end-off].float(),
                         SINB[layer_idx][:, s0-off:g_end-off].float())
    ka[:, :, s0:g_end] = krot.to(ka.dtype)
    if VBITS < 16:
        if args.vhead and layer_idx in VHB:                        # per-head V bits (head-level water-fill)
            _vb = VHB[layer_idx]
            for _b in _vb.unique().tolist():
                _sel = (_vb == _b)
                va[:, _sel, s0:g_end] = _asym(va[:, _sel, s0:g_end].float(), int(_b), -1).to(va.dtype)
        elif args.vqmx and layer_idx in VPROJ:                     # V-QuMix (output-metric basis)
            va[:, :, s0:g_end] = qmx_k(va[:, :, s0:g_end].float(), VPROJ[layer_idx][args.vqmxtag],
                                       tokq=True, gs=args.vgs).to(va.dtype)
        else:
            va[:, :, s0:g_end] = (_asym_sq(va[:, :, s0:g_end].float(), VBITS, -1, int(_os.environ['QMX_VSBITS'])) if _os.environ.get('QMX_VSBITS') else _asym(va[:, :, s0:g_end].float(), VBITS, -1)).to(va.dtype)
    SEALED[layer_idx] = g_end
    SEALCNT[0] += (g_end - s0)                                     # provenance: tokens actually sealed
    KPRE[layer_idx] = KPRE[layer_idx][:, :, g_end-off:]            # drop consumed prefix
    COSB[layer_idx] = COSB[layer_idx][:, g_end-off:]
    SINB[layer_idx] = SINB[layer_idx][:, g_end-off:]
    KOFF[layer_idx] = g_end
    return _ex if _ex is not None else (ka, va)
# ========== [baseline] MixKVQ ==========
MKQ_IMP = {}; MKQ_N = {}
def sealed_mkq_update(self, key, value, layer_idx, cache_kwargs=None):
    """MixKVQ faithful (arXiv 2512.19206 / ACL'26; official repo is README-only, so this is a
    from-paper reimplementation): K per-channel 3-tier {BF16, INT4, INT2} by salience
    A_d = E[|Q_d|] * range_d/(2^2-1), GQA-group level, G=32, R=128 batch-flush K + sliding V,
    salience refreshed at each flush (paper: every R tokens). Thresholds -> FRACTIONS to hit a
    target mean-K-bits (paper tunes thresholds with Optuna on accuracy; fractions disclosed as the
    budget-matching adaptation). Arm: mkq<f4pct> with fixed 2/128 BF16 channels, e.g. mkq24.
    V: uniform 2-bit per-token (identical to every other arm)."""
    a = ARM[0]
    _f4 = int(a[3:]) / 100.0 if a[3:].isdigit() else 0.24
    _r = 128
    _fresh = _cache_len(self, layer_idx) == 0
    ka, va = _orig_update(self, key, value, layer_idx, cache_kwargs)
    B, H, T, D = ka.shape
    _ex = (ka.clone(), va.clone()) if _fresh else None           # exact prefill (KIVI Sec 3.3)
    if _fresh:
        SEALED[layer_idx] = 0; SEALEDV[layer_idx] = 0
        MKQ_IMP.pop(layer_idx, None); MKQ_N.pop(layer_idx, None)
    GS_ = 32
    k_end = (T // _r) * _r
    v_end = max(0, T - _r)
    s0 = SEALED.get(layer_idx, 0)
    if k_end > s0:
        imp = MKQ_IMP.get(layer_idx)
        X = ka[:, :, s0:k_end].float()                           # (B,H,btok,D)
        rng_d = (X.amax(2) - X.amin(2)).mean(0) / 3.0            # (H,D) sensitivity, B=2
        sal = (imp if imp is not None else torch.ones_like(rng_d)) * rng_d
        n16 = 2; n4 = int(round(_f4 * D))
        order = torch.argsort(sal, dim=-1, descending=True)      # (H,D)
        ng = (k_end - s0) // GS_
        Xg = X.view(B, H, ng, GS_, D)
        for h in range(H):
            o = order[h]
            for cls, bits in ((o[n16:n16+n4], 4), (o[n16+n4:], 2)):
                g = Xg[:, h][..., cls]                           # (B,ng,GS_,nc)
                mn = g.amin(2, keepdim=True); mx = g.amax(2, keepdim=True)
                sc = (mx - mn).clamp(min=1e-8) / (2**bits - 1)
                Xg[:, h][..., cls] = ((g - mn)/sc).round().clamp(0, 2**bits-1)*sc + mn
        ka[:, :, s0:k_end] = Xg.view(B, H, -1, D).to(ka.dtype)
        SEALED[layer_idx] = k_end; SEALCNT[0] += (k_end - s0)
    v0 = SEALEDV.get(layer_idx, 0)
    if VBITS < 16 and v_end > v0:
        va[:, :, v0:v_end] = _asym(va[:, :, v0:v_end].float(), VBITS, -1).to(va.dtype)
        SEALEDV[layer_idx] = v_end
    return _ex if _ex is not None else (ka, va)
# ========== [baselines] KIVI, INT2 per-token, FP8 (sealed like QuMix) ==========
def sealed_update(self, key, value, layer_idx, cache_kwargs=None):
    a = ARM[0]                                                   # kivis<bits>[r<R>] | pertok<bits> (per-token K) | fp8 (K and V E4M3)
    _mode = 'fp8' if a == 'fp8' else ('pertok' if a.startswith('pertok') else 'kivis')
    _t = a[6:] if _mode == 'pertok' else (a[5:] if _mode == 'kivis' else ''); _r = 0
    if 'r' in _t: _t, _rs = _t.split('r', 1); _r = int(_rs)
    kb = int(_t) if _t.isdigit() else 2
    _fresh = _cache_len(self, layer_idx) == 0
    ka, va = _orig_update(self, key, value, layer_idx, cache_kwargs)
    B, H, T, D = ka.shape
    # KIVI Sec 3.3 -- "during the prefill phase, the exact key and value tensors are passed
    # to the next layers, although only the quantized KV cache is retained in memory". Hand this
    # forward the exact tensors; the in-place quantization below still lands in the stored cache.
    _ex = (ka.clone(), va.clone()) if _fresh else None
    if _fresh: SEALED[layer_idx] = 0; SEALEDV[layer_idx] = 0
    GS_ = 32
    # faithful KIVI Sec 3.3 semantics, which differ between K and V.
    # K: residual buffer fills to R then the WHOLE buffer is quantized and reset -> the fp16
    # window oscillates in [0,R), average R/2 ("expected to be R/2 for key cache").
    # V: a sliding queue of exactly R tokens ("and R for value cache") -> quantized up to T-R.
    # R=0 degenerates to: K seals every complete group, V seals immediately.
    k_end = (T // _r) * _r if _r > 0 else (T // GS_) * GS_
    v_end = max(0, T - _r) if _r > 0 else T
    s0 = SEALED.get(layer_idx, 0)
    if k_end > s0:
        if _mode == 'pertok':
            ka[:, :, s0:k_end] = _asym(ka[:, :, s0:k_end].float(), kb, -1).to(ka.dtype)          # per token over the 128 channels
        elif _mode == 'fp8':
            ka[:, :, s0:k_end] = _fp8(ka[:, :, s0:k_end].float()).to(ka.dtype)
        else:
          ng = (k_end - s0) // GS_
          x = ka[:, :, s0:k_end].float().view(B, H, ng, GS_, D)    # per-channel over G-token groups
          mn = x.amin(3, keepdim=True); mx = x.amax(3, keepdim=True)
          sc = (mx - mn).clamp(min=1e-8) / (2**kb - 1)
          ka[:, :, s0:k_end] = (((x-mn)/sc).round().clamp(0, 2**kb-1)*sc + mn).view(B, H, -1, D).to(ka.dtype)
        SEALED[layer_idx] = k_end; SEALCNT[0] += (k_end - s0)
    v0 = SEALEDV.get(layer_idx, 0)
    if _mode == 'fp8' and v_end > v0:
        va[:, :, v0:v_end] = _fp8(va[:, :, v0:v_end].float()).to(va.dtype)
        SEALEDV[layer_idx] = v_end
    elif VBITS < 16 and v_end > v0:
        va[:, :, v0:v_end] = _asym(va[:, :, v0:v_end].float(), VBITS, -1).to(va.dtype)
        SEALEDV[layer_idx] = v_end
    return _ex if _ex is not None else (ka, va)
# ========== cache-update dispatch: routes the current arm to its update function ==========
def patched_update(self, key, value, layer_idx, cache_kwargs=None):
    a = ARM[0]
    if (a.startswith('kivis') or a.startswith('pertok') or a == 'fp8') and layer_idx in PROJ:
        return sealed_update(self, key, value, layer_idx, cache_kwargs)
    if a.startswith('qmx') and 'ts' in a and layer_idx in PROJ:
        return sealed_qmx_update(self, key, value, layer_idx, cache_kwargs)
    if a.startswith('mkq'):
        return sealed_mkq_update(self, key, value, layer_idx, cache_kwargs)
    if a.startswith('svdqp'):                        # V stays BF16 (paper); K done pre-RoPE
        return _orig_update(self, key, value, layer_idx, cache_kwargs)
    if a != 'base' and layer_idx in PROJ:
        k = key.float(); v = value.float()
        if a == 'kvqfv':                                             # official V: per-token dynamic NUQ + 1% sparse
            v = kvquant_v_fisher(v, _load_aux()['VLUT'][layer_idx]['cb'].cuda())
            return _orig_update(self, key, v.to(value.dtype), layer_idx, cache_kwargs)
        if a.startswith('qmx') or a.startswith('svdqm') or a.startswith('svdqv') or a in ('kvq2', 'kvqn', 'kvqf', 'svdk', 'trunc', 'unif'):  # all pre-RoPE K (done in apply_rotary); V here
            if a.startswith('qmx') and not PREROPE:
                _tq2 = a.endswith('t'); k = qmx_k(k, PROJ[layer_idx][a[:-1] if _tq2 else a], tokq=_tq2)
            if VBITS >= 16: pass
            elif args.vattn and layer_idx in SV_Q and SV_Q[layer_idx].shape[-2] == v.shape[-2]:
                v = vattn_quant(v, SV_Q[layer_idx], SV_K[layer_idx], VBITS)       # attn-weighted per-token V bits
            elif args.vnuq and layer_idx in VCB: v = nuq_v(v, VCB[layer_idx])   # V per-token NUQ (K/V asymmetry)
            else: v = _asym(v, VBITS, -1)                         # per-token uniform asym
        elif a.startswith('kivi'):
            if a == 'kivihb':                                     # margin-aware per-head K bits (mean 2.0, )
                hbits = KIVIHB[layer_idx]
                for b_ in hbits.unique().tolist():
                    sel = (hbits == b_)
                    k[:, sel] = kivi_k(k[:, sel].permute(0, 2, 1, 3), int(b_), R=0).permute(0, 2, 1, 3)
            else:
                kb = int(a[4:]) if a[4:].isdigit() else 4         # K bits from arm name (kivi2/3/4; R=0)
                k = kivi_k(k.permute(0, 2, 1, 3), kb, R=0).permute(0, 2, 1, 3)
            if VBITS < 16: v = kivi_v(v.permute(0, 2, 1, 3), VBITS, R=0).permute(0, 2, 1, 3)  # V bits=VBITS (K2V4: kivi2 --vbits 4)
        elif a.startswith('noise'):   # control: STRUCTURELESS K error at a controlled RMS|q^T dk|
            _t = a[5:]; _va = _t.endswith('a'); _t = _t[:-1] if _va else _t
            sc = float(_t)/100.0                                  # noise100 = KIVI's own per-layer score noise
            k = k + torch.randn_like(k)*(NSIG[layer_idx]*sc)      # isotropic; post-RoPE == pre-RoPE (RoPE orthogonal)
            # V path is chosen to MATCH the arm this noise arm is being compared against, so the
            # comparison isolates K-error structure only: noiseNNN <-> kivi2 (kivi_v),
            # noiseNNNa <-> qmx2t (per-token asym).
            if VBITS < 16:
                v = _asym(v, VBITS, -1) if _va else kivi_v(v.permute(0, 2, 1, 3), VBITS, R=0).permute(0, 2, 1, 3)
        else:
            raise SystemExit(f"REFUSED: arm '{a}' matched no quantizer branch — it would silently run as base.")
        key, value = k.to(key.dtype), v.to(value.dtype)
    return _orig_update(self, key, value, layer_idx, cache_kwargs)
DynamicCache.update = patched_update
if PREROPE:                                                      # (c): quant K PRE-RoPE in apply_rotary (layer=counter%L)
    def quant_rope(q, k, cos, sin, *aa, **kw):
        ai = ARM[0]; i = ECNT[0] % L; ECNT[0] += 1
        if ai.startswith('qmx') and 'ts' in ai:                              # defer to sealed_qmx_update
            PENDK[i] = (k.detach().float(), cos.detach(), sin.detach())
            return _orig_rope(q, k, cos, sin, *aa, **kw)
        if ai.startswith('mkq'):                                 # MixKVQ: importance = E[|Q_d|]
            qr_, kr_ = _orig_rope(q, k, cos, sin, *aa, **kw)     # post-RoPE queries, GQA-group level
            _aq = qr_[0].detach().float().abs().mean(1)          # (HQ,T,hd)->(HQ,hd) mean over tokens... shape (HQ,T,hd)
            _aq = qr_[0].detach().float().abs().mean(1).view(HKV, GS, HD).mean(1)   # (HKV,hd)
            _n = MKQ_N.get(i, 0)
            MKQ_IMP[i] = _aq if i not in MKQ_IMP else (MKQ_IMP[i]*_n + _aq)/(_n+1)
            MKQ_N[i] = _n + 1
            return qr_, kr_
        if ai == 'qmx2et' and i in PROJ:                         # centered basis + EXPLICIT mean
            if 'kmean' not in PROJ[i]:
                raise SystemExit("REFUSED: qmx2et needs a calib with 'kmean' (rebuild the calibration).")
            _km = PROJ[i]['kmean'][None, :, None, :]                 # (1,HKV,1,hd)
            k = (qmx_k(k.float()-_km, PROJ[i]['qmx2c'], tokq=True)+_km).to(k.dtype)
        elif (ai.startswith('qmx') or ai in ('svdk', 'trunc', 'unif')) and i in PROJ:
            _tq = ai.startswith('qmx') and ai.endswith('t')          # qmx2t: qmx2 basis+alloc, tok32 quantizer
            k = qmx_k(k.float(), PROJ[i][ai[:-1] if _tq else ai], tokq=_tq).to(k.dtype)
        elif ai == 'kvq2':
            k = kvquant_k(k.float(), bits=2).to(k.dtype)         # KVQuant-core pre-RoPE per-channel uniform
        elif ai == 'kvqn':
            k = kvquant_k_nuq(k.float(), CENTK[i]).to(k.dtype)   # KVQuant + per-channel NUQ centroids
        elif ai == 'kvqf':
            kf = KVQF[i]; k = kvquant_k_fisher(k.float(), kf['cb'], kf['zp'], kf['dist']).to(k.dtype)  # KVQuant faithful (Fisher-NUQ)
        elif ai == 'kvqfv':                          # full-faithful nuq2-1% (K here, V at cache)
            kf = _load_aux()['KVQF'][i]
            k = kvquant_k_fisher(k.float(), kf['cb'].cuda(), kf['zp'].cuda(), kf['dist'].cuda()).to(k.dtype)
        elif ai.startswith('svdqm'):
            sv = _load_aux()['SVDQ'][i]
            k = svdq_k(k.float(), sv['U'].cuda().float(), sv['mean'].cuda().float(),
                       SVDQ_SCHED[ai]).to(k.dtype)
        elif ai.startswith('svdqp') or ai.startswith('svdqv'):   # per-sequence basis
            if k.shape[2] >= 32:                     # prefill call; decode blocks (T=4) pass fp16
                k = svdqp_prefill(k, SVDQP_SCHED[ai], SVDQP_TOKGRP.get(ai, 0)).to(k.dtype)
        qr, kr = _orig_rope(q, k, cos, sin, *aa, **kw)
        if args.vattn: SV_Q[i] = qr[0].detach().float(); SV_K[i] = kr[0].detach().float()  # post-RoPE q,k for attn-weighted V
        return qr, kr
    attn_mod.apply_rotary_pos_emb = quant_rope
    print(f"PREROPE mode: K quantized pre-RoPE via apply_rotary (V={VBITS}bit at cache)", flush=True)

def extract(t):
    t = re.split(r'(?i)\n\s*(?:question|problem)\s*[:.]', t)[0]   # cut trailing new problem
    m = re.findall(r'\\boxed\{([^}]*)\}', t)
    if m:
        nb = re.findall(r'-?\d[\d,]*\.?\d*', m[-1].replace(',', ''))
        if nb: return nb[-1].rstrip('.')
    nums = re.findall(r'-?\d[\d,]*\.?\d*', t.replace(',', ''))
    return nums[-1].rstrip('.') if nums else ''
def gold(a): return a.split('####')[-1].strip().replace(',', '')
# ========== [evaluation] datasets and metrics (LongBench is used in the paper; the others are development only) ==========
DSET = args.dataset
BOX = "\nPlease reason step by step, and put your final answer within \\boxed{}.\n"
if DSET == 'math500':                                            # MATH-500, faithful math_verify scoring
    from math_verify import parse as mvparse, verify as mvverify
    from math_verify import LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig
    _MVCFG = [LatexExtractionConfig(), ExprExtractionConfig(), StringExtractionConfig()]
    ds = load_dataset('HuggingFaceH4/MATH-500', split=f'test[{args.start}:{args.n}]')
    def _last_boxed(t):                                          # final answer = LAST \boxed{} (brace-balanced)
        k = t.rfind('\\boxed')
        if k < 0: return None
        i = t.find('{', k)
        if i < 0: return None
        d = 0
        for j in range(i, len(t)):
            if t[j] == '{': d += 1
            elif t[j] == '}':
                d -= 1
                if d == 0: return t[i+1:j]
        return None
    def mkmsg(s, i): return s['problem'] + BOX
    def score(txt, s, i):                                        # math_verify equivalence (gold first, last-boxed pred)
        try:
            lb = _last_boxed(txt); src = '\\boxed{'+lb+'}' if lb is not None else txt
            g = mvparse('\\boxed{'+s['answer']+'}', extraction_config=_MVCFG, extraction_mode='first_match')
            p = mvparse(src, extraction_config=_MVCFG, extraction_mode='first_match')
            return bool(mvverify(g, p))
        except Exception: return False
    def show(txt, s, i): return f"boxed={_last_boxed(txt)!r} vs gold {s['answer']!r}"
elif DSET == 'gpqa':                                             # GPQA-Diamond, SDAR OpenCompass protocol (MC, first capital letter)
    ds = load_dataset('Idavidrein/gpqa', 'gpqa_diamond', split=f'train[{args.start}:{args.n}]')
    GHINT = 'For the multiple choice question below, please provide the correct answer option directly.'
    def _opts(s, i):                                             # deterministic shuffle (opencompass GPQADataset cnt%4)
        o = [s['Correct Answer'], s['Incorrect Answer 1'], s['Incorrect Answer 2'], s['Incorrect Answer 3']]
        pat = ['ABCD', 'BCDA', 'CDAB', 'DABC'][i % 4]; Ln = {}
        for j in range(4): Ln['ABCD'[j]] = o[ord(pat[j]) - ord('A')]
        ans = next('ABCD'[j] for j in range(4) if Ln['ABCD'[j]] == o[0])
        return Ln, ans
    def mkmsg(s, i):
        Ln, _ = _opts(s, i)
        return f"{GHINT}\nQuestion: {s['Question']}\nA. {Ln['A']}\nB. {Ln['B']}\nC. {Ln['C']}\nD. {Ln['D']}\nAnswer:"
    def score(txt, s, i):
        _, ans = _opts(s, i); m = re.search(r'[ABCD]', txt); return bool(m) and m.group(0) == ans
    def show(txt, s, i):
        _, ans = _opts(s, i); m = re.search(r'[ABCD]', txt); return f"{m.group(0) if m else '?'} vs {ans} | {txt[:50]!r}"
elif DSET == 'mmlupro':                                          # MMLU-Pro, SDAR OpenCompass protocol (0-shot CoT, ANSWER:X)
    ds = load_dataset('TIGER-Lab/MMLU-Pro', split=f'test[{args.start}:{args.n}]')
    _CH = 'ABCDEFGHIJKLMNOP'
    def _optstr(s): return '\n'.join(f'{_CH[j]}. {o}' for j, o in enumerate(s['options']) if o != 'N/A')
    def mkmsg(s, i):
        return ("Answer the following multiple choice question. The last line of your response should be of "
                "the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCDEFGHIJKLMNOP. "
                f"Think step by step before answering.\n\nQuestion:\n{s['question']}\n\nOptions:\n{_optstr(s)}")
    def score(txt, s, i):
        m = re.findall(r'(?i)ANSWER\s*:\s*([A-P])', txt)
        return bool(m) and m[-1].upper() == s['answer'].upper()
    def show(txt, s, i):
        m = re.findall(r'(?i)ANSWER\s*:\s*([A-P])', txt); return f"{m[-1] if m else '?'} vs {s['answer']}"
elif DSET == 'longbench':                                        # LongBench: QA-F1 / ROUGE-L / classification / code edit similarity (official LongBench metrics)
    import json, string
    from collections import Counter
    from rouge import Rouge
    from fuzzywuzzy import fuzz
    _LBD = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'longbench')   # official LongBench prompt/length config (MIT)
    _LBP = json.load(open(f'{_LBD}/config/dataset2prompt.json'))[args.lbtask]
    _LBMAX = int(json.load(open(f'{_LBD}/config/dataset2maxlen.json')).get(args.lbtask, args.gen))
    if args.lbgen <= 0: args.gen = _LBMAX                         # per-task max gen (qasper128/qmsum512/...); --lbgen overrides for gen-sensitivity check
    with open(_os.path.join(_os.environ.get('LONGBENCH_DIR', 'data/longbench'), 'data', f'{args.lbtask}.jsonl')) as _f:   # official LongBench data.zip, extracted
        _all = [json.loads(l) for l in _f]
    ds = _all[args.start:args.n]
    def _norm(s):
        import re as _re
        s = s.lower(); s = ''.join(c if c not in string.punctuation else ' ' for c in s)
        s = _re.sub(r'\b(a|an|the)\b', ' ', s); return ' '.join(s.split())
    def _f1(pred, gt, **kw):
        pt = _norm(pred).split(); gt_ = _norm(gt).split()
        common = Counter(pt) & Counter(gt_); ns = sum(common.values())
        if ns == 0 or not pt or not gt_: return 0.0
        p = ns/len(pt); r = ns/len(gt_); return 2*p*r/(p+r)
    def _rouge(pred, gt, **kw):
        try: return Rouge().get_scores([pred if pred.strip() else ' '], [gt])[0]['rouge-l']['f']
        except Exception: return 0.0
    def _codesim(pred, gt, **kw):                                # first non-comment line, fuzzy ratio
        for line in pred.lstrip('\n').split('\n'):
            if ('`' not in line) and ('#' not in line) and ('//' not in line): pred = line; break
        return fuzz.ratio(pred, gt) / 100
    def _cls(pred, gt, all_classes=None, **kw):                  # classification: which class names appear in pred
        em = [c for c in (all_classes or []) if c in pred]
        for m in list(em):
            if m in gt and m != gt: em.remove(m)
        return 1.0/len(em) if gt in em else 0.0
    _LBMETRIC = {'qasper':_f1,'triviaqa':_f1,'multifieldqa_en':_f1,'hotpotqa':_f1,'2wikimqa':_f1,'musique':_f1,'narrativeqa':_f1,
                 'qmsum':_rouge,'multi_news':_rouge,'samsum':_rouge,'gov_report':_rouge,
                 'trec':_cls,'lcc':_codesim,'repobench-p':_codesim}
    _MET = _LBMETRIC.get(args.lbtask, _f1)
    def mkmsg(s, i): return _LBP.format(context=s['context'], input=s['input'])
    def score(txt, s, i): return max((_MET(txt, a, all_classes=s.get('all_classes')) for a in s['answers']), default=0.0)
    def show(txt, s, i): return f"m={score(txt,s,i):.2f} | {txt[:35]!r} vs {s['answers'][0][:30]!r}"
else:                                                            # gsm8k (default, numeric boxed)
    ds = load_dataset('gsm8k', 'main', split=f'test[{args.start}:{args.n}]')
    def mkmsg(s, i): return s['question'] + BOX
    def score(txt, s, i): return extract(txt) == gold(s["answer"])
    def show(txt, s, i): return f"{extract(txt)!r} vs {gold(s['answer'])!r} | {txt[:70]!r}"
# ========== [evaluation] main loop: every arm runs on the same examples ==========
res = {}; PERSAMPLE = {}
for arm in args.arms.split(','):
    ARM[0] = arm; correct = 0; t0 = time.time(); GLEN = []; PERSAMPLE[arm] = {}
    if args.bs > 1 and DSET != 'longbench':                       # exact-length batched path (zero-pad only)
        from batch_exact import batched_generate_exact
        from collections import defaultdict as _dd
        _items = []
        for i, s in enumerate(ds):
            _sstr = tok.apply_chat_template([dict(role="user", content=mkmsg(s, i))], add_generation_prompt=True, tokenize=False)
            _items.append((i, s, tok(_sstr, return_tensors='pt', add_special_tokens=False)['input_ids'][0]))
        _buck = _dd(list)
        for _it in _items: _buck[int(_it[2].shape[0])].append(_it)
        _done = 0
        for _ln in sorted(_buck):
            _grp = _buck[_ln]
            for _j0 in range(0, len(_grp), args.bs):
                _ch = _grp[_j0:_j0+args.bs]
                _outs = batched_generate_exact(model, [c[2] for c in _ch], mask_id=MASK, gen_length=args.gen,
                    block_length=args.block, denoising_steps=args.steps, temperature=args.temp,
                    top_k=args.topk, remasking_strategy=args.remask, confidence_threshold=0.9, eos=tuple(EOS))
                for (_i, _s, _), _g in zip(_ch, _outs):
                    _gl = int(_g.shape[0]); GLEN.append(_gl)
                    _txt = tok.decode(_g, skip_special_tokens=True)
                    _c = float(score(_txt, _s, _i)); correct += _c; _done += 1
                    PERSAMPLE[arm][_i] = (_c, _gl, _g.tolist())   # +tokens (divergence probe)
                    if _done == 1: print(f"  [{arm} s0/bs{args.bs}] {show(_txt, _s, _i)}", flush=True)
                if _done > 1 and _done % 80 < args.bs:
                    print(f"  [{arm}/bs] {_done}/{len(ds)} acc~{correct/_done:.3f}  {(time.time()-t0)/_done:.1f}s/it", flush=True)
        res[arm] = correct/len(ds)
        if GLEN:
            _sg = sorted(GLEN); _n = len(_sg)
            print(f"  [{arm} genlen] mean={sum(GLEN)/_n:.0f} med={_sg[_n//2]} p90={_sg[int(_n*0.9)]}", flush=True)
        print(f"==> {arm}: acc={res[arm]:.3f} ({correct}/{len(ds)})  {time.time()-t0:.0f}s  [bs={args.bs}]", flush=True)
        continue
    for i, s in enumerate(ds):
        msg = [dict(role="user", content=mkmsg(s, i))]
        sstr = tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
        enc = tok.batch_encode_plus([sstr], return_tensors='pt', add_special_tokens=False)
        enc = {k: v.to('cuda') for k, v in enc.items()}; plen = enc['input_ids'].shape[1]
        if DSET == 'longbench' and plen > args.lbmax:            # LongBench middle truncation (standard; bounds long-context attention memory)
            ii = enc['input_ids'][0]; h = args.lbmax // 2
            ii = torch.cat([ii[:h], ii[-h:]])
            enc = {'input_ids': ii[None], 'attention_mask': torch.ones(1, ii.shape[0], dtype=torch.long, device='cuda')}; plen = ii.shape[0]
        if args.ctxnoise > 0:                                    # degrade base WITHOUT touching decode procedure or vocab stats
            _ci = enc['input_ids'][0].clone()
            _gen = torch.Generator().manual_seed(20260819 + i)
            _lo, _hi = 16, max(17, _ci.shape[0] - 160)           # spare chat-template head + trailing question/instruction
            _sel = torch.arange(_lo, _hi)[torch.rand(_hi - _lo, generator=_gen) < args.ctxnoise]
            _src = torch.randint(_lo, _hi, (_sel.shape[0],), generator=_gen)
            _ci[_sel.to(_ci.device)] = _ci[_src.to(_ci.device)]
            enc['input_ids'] = _ci[None]
        out = G.block_diffusion_generate(model, prompt=enc, mask_id=MASK, gen_length=args.gen,
            block_length=args.block, denoising_steps=args.steps, temperature=args.temp, top_k=args.topk, top_p=1.0,
            remasking_strategy=args.remask, confidence_threshold=0.9, eb_threshold=0.0,
            stopping_criteria_idx=EOS)
        txt = tok.decode(out[0, plen:], skip_special_tokens=True)
        _ge = out[0, plen:].tolist()                             # gen-length until first EOS (cap-hit diagnosis)
        GLEN.append(next((j for j, tk in enumerate(_ge) if tk in EOS), len(_ge)))
        PERSAMPLE[arm][i] = dict(score=float(score(txt, s, i)), pred=txt, gold=s.get('answers', s.get('answer')), tokens=_ge, plen=plen)
        if _os.environ.get('DUMP'):
            import json as _record_json
            with open(_os.environ['DUMP'], 'w') as _record_file:
                _record_json.dump(PERSAMPLE, _record_file)
        correct += float(score(txt, s, i))                       # float: bool -> 1.0, F1 -> continuous
        if i == 0: print(f"  [{arm} s0] {show(txt, s, i)}", flush=True)
        elif (i + 1) % 40 == 0: print(f"  [{arm}] {i+1}/{len(ds)} acc~{correct/(i+1):.3f}  {(time.time()-t0)/(i+1):.1f}s/it", flush=True)
    res[arm] = correct/len(ds)
    if GLEN:
        _sg = sorted(GLEN); _n = len(_sg); _cap = sum(1 for g in GLEN if g >= args.gen)
        print(f"  [{arm} genlen] mean={sum(GLEN)/_n:.0f} med={_sg[_n//2]} p90={_sg[int(_n*0.9)]} cap-hit={_cap}/{_n}={100*_cap/_n:.0f}%", flush=True)
    if SEALCNT[0]: print(f"  [{arm} sealed] {SEALCNT[0]} tokens sealed into 32-tok groups", flush=True)
    SEALCNT[0] = 0
    print(f"==> {arm}: acc={res[arm]:.3f} ({correct}/{len(ds)})  {time.time()-t0:.0f}s", flush=True)
print("RESULTS:", {k: round(v, 3) for k, v in res.items()}, flush=True)
import os as _os, json as _json
if _os.environ.get('DUMP'):
    _json.dump(PERSAMPLE, open(_os.environ['DUMP'], 'w'))
    print(f"dumped per-sample to {_os.environ['DUMP']}", flush=True)
