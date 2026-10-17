"""ctypes bindings for cuda/libdet.so.  Device memory is managed with torch tensors; pass .data_ptr()."""
import ctypes, os
import numpy as np
import torch

_raw = ctypes.CDLL(os.environ.get("DET_LIB", os.path.join(os.path.dirname(__file__), "..", "cuda", "libdet.so")))

class _Missing:
    def __init__(self, name): self.name = name
    def __call__(self, *a, **k): raise RuntimeError(f"{self.name} not implemented in libdet.so")
    def __setattr__(self, k, v):
        if k == "name": object.__setattr__(self, k, v)
def _fn(name):
    try: return getattr(_raw, name)
    except AttributeError: return _Missing(name)
class _L:
    def __getattr__(self, name): return _fn(name)
_lib = _L()
vp, fp, i8p, i16p, u8p, u16p, ip = (ctypes.c_void_p,) * 7
c_int, c_float, c_ll, c_size = ctypes.c_int, ctypes.c_float, ctypes.c_longlong, ctypes.c_size_t

class QLinear(ctypes.Structure):
    _fields_ = [("wq", vp), ("c", vp), ("m", vp), ("N", c_int), ("K", c_int), ("bits", c_int)]
class AttnLayer(ctypes.Structure):
    _fields_ = [("q_norm", vp), ("k_norm", vp), ("v_scale", vp), ("v_inv_scale", vp)]
class KVCache(ctypes.Structure):
    _fields_ = [("kq", vp), ("sk", vp), ("vq", vp), ("capacity", c_int)]
class GdnLayer(ctypes.Structure):
    _fields_ = [("conv", vp), ("A", vp), ("dt_bias", vp), ("norm", vp)]
class GdnState(ctypes.Structure):
    _fields_ = [("conv_state", vp), ("S", vp)]

_lib.det_rmsnorm_quant.argtypes = [vp, vp, c_float, c_int, c_int, vp, vp, vp, vp, vp]
_lib.det_quant.argtypes = [vp, c_int, c_int, vp, vp, vp, vp]
_lib.det_embed.argtypes = [vp, vp, c_int, c_int, vp, vp]
_lib.det_add.argtypes = [vp, vp, c_ll, vp]
_lib.det_silu_mul_quant.argtypes = [vp, vp, c_int, c_int, vp, vp, vp, vp, vp]
_lib.det_argmax.argtypes = [vp, c_int, c_int, vp, vp]
_lib.det_gemm.argtypes = [ctypes.POINTER(QLinear), vp, vp, vp, c_int, vp, vp]
_lib.det_gemm_scratch_bytes.argtypes = [c_int, c_int]; _lib.det_gemm_scratch_bytes.restype = c_size
_lib.det_gemm_set_scratch.argtypes = [vp, c_size]
_lib.det_attention.argtypes = [ctypes.POINTER(AttnLayer), ctypes.POINTER(KVCache), vp, vp, vp, vp, vp, c_int, c_float,
                               c_int, c_int, c_int, c_int, c_int, vp, vp, vp]
_lib.det_attn_scratch_bytes.argtypes = [c_int, c_int, c_int]; _lib.det_attn_scratch_bytes.restype = c_size
_lib.det_attention_dev.argtypes = [ctypes.POINTER(AttnLayer), ctypes.POINTER(KVCache), vp, vp, vp, vp, vp, c_int, c_float,
                                   c_int, vp, c_int, c_int, c_int, c_int, vp, vp, vp]
_lib.det_gdn.argtypes = [ctypes.POINTER(GdnLayer), ctypes.POINTER(GdnState), vp, vp, vp, vp, c_float,
                         c_int, c_int, c_int, c_int, c_int, vp, vp, vp]
_lib.det_gdn_scratch_bytes.argtypes = [c_int, c_int, c_int, c_int, c_int]; _lib.det_gdn_scratch_bytes.restype = c_size

def P(t):
    """device pointer of a torch tensor (or None)"""
    if t is None: return None
    assert t.is_cuda and t.is_contiguous(), "need contiguous cuda tensor"
    return t.data_ptr()

def stream_ptr(st=None):
    return (st or torch.cuda.current_stream()).cuda_stream

def rmsnorm_quant(x, w, eps, y=None, xq=None, sx=None, mx=None, st=None):
    M, n = x.shape
    _lib.det_rmsnorm_quant(P(x), P(w), eps, M, n, P(y), P(xq), P(sx), P(mx), stream_ptr(st))
def quant(x, xq, sx, mx=None, st=None):
    M, n = x.shape; _lib.det_quant(P(x), M, n, P(xq), P(sx), P(mx), stream_ptr(st))
def embed(table, ids, x, st=None):
    _lib.det_embed(P(table), P(ids), ids.shape[0], table.shape[1], P(x), stream_ptr(st))
def add(x, y, st=None):
    _lib.det_add(P(x), P(y), x.numel(), stream_ptr(st))
def silu_mul_quant(g, u, h, hq, sh, mh=None, st=None):
    M, n = g.shape; _lib.det_silu_mul_quant(P(g), P(u), M, n, P(h), P(hq), P(sh), P(mh), stream_ptr(st))
def argmax(logits, out, st=None):
    M, V = logits.shape; _lib.det_argmax(P(logits), M, V, P(out), stream_ptr(st))

def make_qlinear(wq_packed, c, m, N, K, bits):
    """tensors must stay alive while the QLinear is used"""
    q = QLinear(P(wq_packed), P(c), P(m), N, K, bits); q._keep = (wq_packed, c, m); return q
def gemm(L, xq, sx, mx, y, st=None):
    _lib.det_gemm(ctypes.byref(L), P(xq), P(sx), P(mx), xq.shape[0], P(y), stream_ptr(st))
def gemm_scratch_bytes(M, K): return _lib.det_gemm_scratch_bytes(M, K)
def gemm_set_scratch(buf):
    """buf: cuda uint8 tensor (must stay alive while set; None unsets) used by det_gemm for M > 64 W4/W6"""
    _lib.det_gemm_set_scratch(P(buf), 0 if buf is None else buf.numel() * buf.element_size())

def attn_scratch_bytes(M, n_h, hd): return _lib.det_attn_scratch_bytes(M, n_h, hd)
def attention(L, kv, qg, k, v, rope_cos, rope_sin, rot, eps, pos0, n_h, n_kv, hd, o, scratch, st=None):
    M = qg.shape[0]
    _lib.det_attention(ctypes.byref(L), ctypes.byref(kv), P(qg), P(k), P(v), P(rope_cos), P(rope_sin), rot, eps,
                       M, pos0, n_h, n_kv, hd, P(o), P(scratch), stream_ptr(st))
def attention_dev(L, kv, qg, k, v, rope_cos, rope_sin, rot, eps, pos0_dev, max_pos, n_h, n_kv, hd, o, scratch, st=None):
    """graph-capturable attention: pos0 read from the device int32 tensor pos0_dev[1]; sized for pos0 <= max_pos"""
    M = qg.shape[0]
    _lib.det_attention_dev(ctypes.byref(L), ctypes.byref(kv), P(qg), P(k), P(v), P(rope_cos), P(rope_sin), rot, eps,
                           M, P(pos0_dev), max_pos, n_h, n_kv, hd, P(o), P(scratch), stream_ptr(st))
def gdn_scratch_bytes(M, n_k, n_v, dk, dv): return _lib.det_gdn_scratch_bytes(M, n_k, n_v, dk, dv)
def gdn(L, S, qkv, z, a, b, eps, n_k, n_v, dk, dv, on, scratch, st=None):
    M = qkv.shape[0]
    _lib.det_gdn(ctypes.byref(L), ctypes.byref(S), P(qkv), P(z), P(a), P(b), eps, M, n_k, n_v, dk, dv, P(on), P(scratch), stream_ptr(st))

_lib.det_sample.argtypes = [vp, c_int, c_int, c_float, vp, vp, vp]
def sample(logits, inv_temp, z, out, st=None):
    """logits [M,V] f32, z [M] int64 (raw splitmix64 outputs), out [M] int32"""
    M, V = logits.shape; _lib.det_sample(P(logits), M, V, inv_temp, P(z), P(out), stream_ptr(st))
# ---- top-k / top-p sampling
_lib.det_sample_topkp.argtypes = [vp, c_int, c_int, c_float, c_int, c_float, vp, vp, vp, vp]
_lib.det_sample_scratch_bytes.argtypes = [c_int, c_int]; _lib.det_sample_scratch_bytes.restype = c_size
def sample_scratch_bytes(M, V): return _lib.det_sample_scratch_bytes(M, V)
def sample_topkp(logits, inv_temp, top_k, top_p, z, out, scratch, st=None):
    """as sample(), with top_k (0 = off) and top_p (1.0 = off) per detref/sample.py; scratch: uint8 cuda tensor >= sample_scratch_bytes(M, V)"""
    M, V = logits.shape
    assert scratch.numel() >= sample_scratch_bytes(M, V), "sampling scratch too small"
    _lib.det_sample_topkp(P(logits), M, V, inv_temp, int(top_k), float(top_p), P(z), P(out), P(scratch), stream_ptr(st))

def to_dev(a):
    """numpy -> cuda torch tensor (bf16 bits as uint16 ok; int16/uint16 ok)"""
    if a.dtype == np.uint16: t = torch.from_numpy(a.view(np.int16)).cuda()
    else: t = torch.from_numpy(np.ascontiguousarray(a)).cuda()
    return t
