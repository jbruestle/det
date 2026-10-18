"""ctypes bindings for metal/libdet_metal.dylib.  Device memory = shared (unified) buffers from det_malloc, wrapped in
`Buf` objects that expose a numpy view (`.np`); the C API takes the host pointers.  Mirrors detcuda.py's function
names so that the model runner / tests can be ported almost verbatim.  Always call sync() before reading results."""
import ctypes, os
import numpy as np

_path = os.environ.get("DET_LIB", os.path.join(os.path.dirname(__file__), "..", "metal", "libdet_metal.dylib"))
_lib = ctypes.CDLL(_path)
vp = ctypes.c_void_p
c_int, c_float, c_ll, c_size, c_ull = ctypes.c_int, ctypes.c_float, ctypes.c_longlong, ctypes.c_size_t, ctypes.c_ulonglong

_lib.det_init.restype = c_int
_lib.det_device_name.restype = ctypes.c_char_p
_lib.det_malloc.argtypes = [c_size]; _lib.det_malloc.restype = vp
_lib.det_free.argtypes = [vp]
_lib.det_allocated_bytes.restype = c_size
_lib.det_memset.argtypes = [vp, c_int, c_size, vp]
_lib.det_memcpy.argtypes = [vp, vp, c_size, vp]
_lib.det_test_prim.argtypes = [c_int, vp, c_int, c_int, vp, vp]

class QLinear(ctypes.Structure):
    _fields_ = [("wq", vp), ("c", vp), ("m", vp), ("N", c_int), ("K", c_int), ("bits", c_int)]
class AttnLayer(ctypes.Structure):
    _fields_ = [("q_norm", vp), ("k_norm", vp), ("v_scale", vp), ("v_inv_scale", vp)]
class KVCache(ctypes.Structure):
    _fields_ = [("kq", vp), ("sk", vp), ("vq", vp), ("capacity", c_int), ("vlo", vp), ("vhi", vp)]
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
_lib.det_copy_cols.argtypes = [vp, c_int, c_int, c_int, c_int, vp, vp]
_lib.det_split_cols.argtypes = [vp, c_int, c_int, c_int, ctypes.POINTER(c_int), ctypes.POINTER(c_int), ctypes.POINTER(vp), vp]
_lib.det_gemm.argtypes = [ctypes.POINTER(QLinear), vp, vp, vp, c_int, vp, vp]
_lib.det_gemm_ex.argtypes = [ctypes.POINTER(QLinear), vp, vp, vp, c_int, vp, c_int, c_int, ctypes.POINTER(c_int), ctypes.POINTER(vp), vp]
_lib.det_gemm_scratch_bytes.argtypes = [c_int, c_int]; _lib.det_gemm_scratch_bytes.restype = c_size
_lib.det_gemm_set_scratch.argtypes = [vp, c_size]
_lib.det_attention.argtypes = [ctypes.POINTER(AttnLayer), ctypes.POINTER(KVCache), vp, vp, vp, vp, vp, c_int, c_float,
                               c_int, c_int, c_int, c_int, c_int, vp, vp, vp]
_lib.det_attn_scratch_bytes.argtypes = [c_int, c_int, c_int]; _lib.det_attn_scratch_bytes.restype = c_size
_lib.det_gdn.argtypes = [ctypes.POINTER(GdnLayer), ctypes.POINTER(GdnState), vp, vp, vp, vp, c_float,
                         c_int, c_int, c_int, c_int, c_int, vp, vp, vp]
_lib.det_gdn_scratch_bytes.argtypes = [c_int, c_int, c_int, c_int, c_int]; _lib.det_gdn_scratch_bytes.restype = c_size
_lib.det_sample.argtypes = [vp, c_int, c_int, c_float, vp, vp, vp]
_lib.det_sample_topkp.argtypes = [vp, c_int, c_int, c_float, c_int, c_float, vp, vp, vp, vp]
_lib.det_sample_scratch_bytes.argtypes = [c_int, c_int]; _lib.det_sample_scratch_bytes.restype = c_size


class Buf:
    """A shared GPU buffer with a numpy view.  `Buf(shape, dtype)` allocates (zeroed); slicing along the first axis
    (`b[a:b]`, `b[i]`) returns a view Buf (same memory).  `.ptr` is the host pointer the C API takes."""
    def __init__(self, shape, dtype, _parent=None, _ptr=None, _np=None):
        self.shape = tuple(shape) if not isinstance(shape, int) else (shape,)
        self.dtype = np.dtype(dtype)
        if _parent is None:
            self.nbytes = int(np.prod(self.shape)) * self.dtype.itemsize
            self.ptr = _lib.det_malloc(max(self.nbytes, 16))
            if not self.ptr: raise MemoryError(f"det_malloc({self.nbytes}) failed")
            self._owner = True
            raw = (ctypes.c_uint8 * max(self.nbytes, 16)).from_address(self.ptr)
            self.np = np.frombuffer(raw, dtype=np.uint8)[:self.nbytes].view(self.dtype).reshape(self.shape)
        else:
            self._owner = False; self._parent = _parent; self.ptr = _ptr; self.np = _np; self.nbytes = _np.nbytes
    def __del__(self):
        if getattr(self, "_owner", False) and self.ptr:
            try: _lib.det_free(self.ptr)
            except Exception: pass
    def __getitem__(self, idx):
        v = self.np[idx]
        if not isinstance(idx, slice):   # single row -> keep a leading axis of 1
            v = v[None] if v.ndim < self.np.ndim else v
        off = v.ctypes.data - self.np.ctypes.data
        return Buf(v.shape, self.dtype, _parent=self, _ptr=self.ptr + off, _np=v)
    def numel(self): return int(np.prod(self.shape))
    def element_size(self): return self.dtype.itemsize
    def copy_(self, a):
        """host -> buffer copy (the caller guarantees no GPU work is reading the buffer)"""
        a = np.asarray(a)
        if a.dtype == np.uint16 and self.dtype == np.int16: a = a.view(np.int16)
        self.np[...] = a.reshape(self.shape); return self
    def zero_(self): self.np[...] = 0; return self
    def fill_(self, v): self.np[...] = v; return self
    def cpu(self): return self   # torch-compat shim: .cpu().numpy() after sync()
    def numpy(self): sync(); return self.np
    def item(self): sync(); return self.np.item()


def sync(): _lib.det_sync()
def flush(): _lib.det_flush()
def init(): return _lib.det_init()
def device_name(): return _lib.det_device_name().decode()
def allocated_bytes(): return _lib.det_allocated_bytes()

def P(b):
    """host pointer of a Buf (or None)"""
    if b is None: return None
    return b.ptr

def empty(shape, dtype=np.float32): return Buf(shape, dtype)
def zeros(shape, dtype=np.float32): return Buf(shape, dtype)
def to_dev(a):
    """numpy -> Buf copy (uint16 is kept as uint16)"""
    a = np.ascontiguousarray(a)
    b = Buf(a.shape, a.dtype); b.np[...] = a; return b

def memset(b, value=0): _lib.det_memset(b.ptr, value, b.nbytes, None)
def memcpy(dst, src, nbytes=None): _lib.det_memcpy(dst.ptr, src.ptr, dst.nbytes if nbytes is None else nbytes, None)
def test_prim(op, x, out, M, n): _lib.det_test_prim(op, P(x), M, n, P(out), None)

def rmsnorm_quant(x, w, eps, y=None, xq=None, sx=None, mx=None, st=None):
    M, n = x.shape
    _lib.det_rmsnorm_quant(P(x), P(w), eps, M, n, P(y), P(xq), P(sx), P(mx), None)
def quant(x, xq, sx, mx=None, st=None):
    M, n = x.shape; _lib.det_quant(P(x), M, n, P(xq), P(sx), P(mx), None)
def embed(table, ids, x, st=None):
    _lib.det_embed(P(table), P(ids), ids.shape[0], table.shape[1], P(x), None)
def add(x, y, st=None):
    _lib.det_add(P(x), P(y), x.numel(), None)
def silu_mul_quant(g, u, h, hq, sh, mh=None, st=None):
    M, n = g.shape; _lib.det_silu_mul_quant(P(g), P(u), M, n, P(h), P(hq), P(sh), P(mh), None)
def split_cols(src, N, T, offs, ns, dsts, st=None):
    """dsts[i][t][:ns[i]] = src[t][offs[i]:offs[i]+ns[i]] for all parts in one dispatch (<= 4 parts)"""
    k = len(ns)
    _lib.det_split_cols(P(src), N, T, k, (c_int * k)(*offs), (c_int * k)(*ns), (vp * k)(*[P(d) for d in dsts]), None)
def argmax(logits, out, st=None):
    M, V = logits.shape; _lib.det_argmax(P(logits), M, V, P(out), None)

def tile_weights(packed, N, K, bits):
    """standard row-major packing [N][K/32*bpl] (numpy uint8) -> the Metal tiled layout [ceil(N/64)][K/128][64][4*bpl]
    (zero rows beyond N) as a Buf"""
    packed = np.asarray(packed, dtype=np.uint8).reshape(N, -1)
    bpl = {8: 32, 6: 24, 4: 16}[bits]; G = K // 128
    assert packed.shape[1] == K // 32 * bpl, packed.shape
    NB = (N + 63) // 64
    out = Buf((NB, G, 64, 4 * bpl), np.uint8)
    view = out.np
    src = packed.reshape(N, G, 4 * bpl)
    for nb in range(NB):           # chunked transpose (bounded temporaries for the 27B lm_head)
        rows = src[nb * 64:(nb + 1) * 64]
        view[nb, :, :rows.shape[0], :] = rows.transpose(1, 0, 2)
    return out

def make_qlinear(wq_packed, c, m, N, K, bits):
    """wq_packed: standard packed codes (numpy or Buf, row-major) -> retiled on the device; c, m: numpy or Buf.
    The returned QLinear keeps its buffers alive."""
    wq_t = tile_weights(wq_packed.np if isinstance(wq_packed, Buf) else wq_packed, N, K, bits)
    if not isinstance(c, Buf): c = to_dev(np.asarray(c, dtype=np.float32))
    if not isinstance(m, Buf): m = to_dev(np.asarray(m, dtype=np.uint16))
    q = QLinear(P(wq_t), P(c), P(m), N, K, bits); q._keep = (wq_t, c, m); return q
def gemm(L, xq, sx, mx, y, st=None, accumulate=False, parts=None):
    """y = L(xq) (or y += ... with accumulate).  parts = (offs, dsts): route the output columns into the dst buffers
    (column block i = [offs[i], offs[i+1])) instead of y (Metal fused epilogue, see det_gemm_ex)."""
    if not accumulate and parts is None:
        _lib.det_gemm(ctypes.byref(L), P(xq), P(sx), P(mx), xq.shape[0], P(y), None); return
    offs, dsts = parts if parts is not None else ([], [])
    k = len(offs)
    _lib.det_gemm_ex(ctypes.byref(L), P(xq), P(sx), P(mx), xq.shape[0], P(y), int(accumulate), k,
                     (c_int * max(k, 1))(*offs), (vp * max(k, 1))(*[P(d) for d in dsts]), None)
def gemm_scratch_bytes(M, K): return _lib.det_gemm_scratch_bytes(M, K)
def gemm_set_scratch(buf): _lib.det_gemm_set_scratch(P(buf), 0 if buf is None else buf.nbytes)

def attn_scratch_bytes(M, n_h, hd): return _lib.det_attn_scratch_bytes(M, n_h, hd)
def attention(L, kv, qg, k, v, rope_cos, rope_sin, rot, eps, pos0, n_h, n_kv, hd, o, scratch, st=None):
    M = qg.shape[0]
    _lib.det_attention(ctypes.byref(L), ctypes.byref(kv), P(qg), P(k), P(v), P(rope_cos), P(rope_sin), rot, eps,
                       M, pos0, n_h, n_kv, hd, P(o), P(scratch), None)
def gdn_scratch_bytes(M, n_k, n_v, dk, dv): return _lib.det_gdn_scratch_bytes(M, n_k, n_v, dk, dv)
def gdn(L, S, qkv, z, a, b, eps, n_k, n_v, dk, dv, on, scratch, st=None):
    M = qkv.shape[0]
    _lib.det_gdn(ctypes.byref(L), ctypes.byref(S), P(qkv), P(z), P(a), P(b), eps, M, n_k, n_v, dk, dv, P(on), P(scratch), None)

def sample(logits, inv_temp, z, out, st=None):
    M, V = logits.shape; _lib.det_sample(P(logits), M, V, inv_temp, P(z), P(out), None)
def sample_scratch_bytes(M, V): return _lib.det_sample_scratch_bytes(M, V)
def sample_topkp(logits, inv_temp, top_k, top_p, z, out, scratch, st=None):
    M, V = logits.shape
    assert scratch.nbytes >= sample_scratch_bytes(M, V), "sampling scratch too small"
    _lib.det_sample_topkp(P(logits), M, V, inv_temp, int(top_k), float(top_p), P(z), P(out), P(scratch), None)
