"""Weight quantization + packing (part of the SPEC for the model file format).

A quantized linear W[N,K] (y = W x) is stored as:
  wq : integer codes, symmetric, |wq| <= qmax = 2^(bits-1)-1, packed per 32 weights (see pack/unpack)
  c  : fp32 [N]           per-output-row scale
  m  : uint16 [N, K/128]  per-group integer multiplier (>= 1)
  dequantized weight = c[n] * m[n, k//128] * wq[n, k]
GEMM semantics (exact integer arithmetic, see model.linear):
  acc[t,n] = sum_g m[n,g] * sum_{k in g} wq[n,k] * xq[t,k]   (int64, exact)
  y[t,n]   = (float32(acc[t,n]) * c[n]) * sx[t]
"""
import numpy as np
f32 = np.float32
GROUP = 128

def qmax_for(bits):
    return (1 << (bits - 1)) - 1

def quantize_linear(W, bits, search=True, row_chunk=4096):
    """W: float array [N, K]. Returns dict(wq int8 [N,K] (unpacked), c f32 [N], m u16 [N,G], bits).
    Rows are processed in chunks to bound memory (the search needs several float64 temporaries)."""
    N = W.shape[0]
    if N > row_chunk:
        parts = [quantize_linear(W[i:i + row_chunk], bits, search, row_chunk) for i in range(0, N, row_chunk)]
        return dict(wq=np.concatenate([p["wq"] for p in parts]), c=np.concatenate([p["c"] for p in parts]),
                    m=np.concatenate([p["m"] for p in parts]), bits=bits)
    W = np.asarray(W, dtype=np.float64)
    N, K = W.shape
    assert K % GROUP == 0, K
    G = K // GROUP
    qmax = qmax_for(bits)
    Wg = W.reshape(N, G, GROUP)
    amax = np.abs(Wg).max(-1)                       # [N,G]
    amax = np.where(amax == 0, 1e-12, amax)
    base = amax / qmax
    if search:
        best_err = None; best_s = None
        for f in np.linspace(0.80, 1.0, 11):
            s = base * f
            q = np.clip(np.rint(Wg / s[..., None]), -qmax, qmax)
            err = ((q * s[..., None] - Wg) ** 2).sum(-1)
            if best_err is None:
                best_err, best_s = err, s
            else:
                better = err < best_err
                best_s = np.where(better, s, best_s); best_err = np.where(better, err, best_err)
        s = best_s
    else:
        s = base
    # row scale c and integer group multipliers m (16-bit)
    c = (s.max(1) / 65535.0).astype(f32)              # [N]
    m = np.clip(np.rint(s / c.astype(np.float64)[:, None]), 1, 65535).astype(np.uint16)
    seff = c.astype(np.float64)[:, None] * m.astype(np.float64)   # effective scale
    wq = np.clip(np.rint(Wg / seff[..., None]), -qmax, qmax).astype(np.int8).reshape(N, K)
    return dict(wq=wq, c=c, m=m, bits=bits)

def dequant(q):
    N, K = q["wq"].shape
    seff = q["c"].astype(np.float64)[:, None] * q["m"].astype(np.float64)
    return (q["wq"].reshape(N, -1, GROUP).astype(np.float64) * seff[..., None]).reshape(N, K)

# ---------------------------------------------------------------- packing
def pack(wq, bits):
    """wq int8 [N,K] -> uint8 packed [N, K/32 * bytes_per_32]."""
    N, K = wq.shape
    assert K % 32 == 0
    if bits == 8:
        return wq.view(np.uint8).copy()
    u = (wq.astype(np.int16) + (1 << (bits - 1))).astype(np.uint8).reshape(N, K // 32, 32)
    if bits == 4:
        return ((u[:, :, :16] & 0xF) | (u[:, :, 16:] << 4)).reshape(N, K // 2)
    if bits == 6:
        ql = (u[:, :, :16] & 0xF) | ((u[:, :, 16:] & 0xF) << 4)           # 16 bytes
        hi = u >> 4                                                          # 2 bits each
        qh = hi[:, :, 0:8] | (hi[:, :, 8:16] << 2) | (hi[:, :, 16:24] << 4) | (hi[:, :, 24:32] << 6)  # 8 bytes
        return np.concatenate([ql, qh], axis=2).reshape(N, K // 32 * 24)
    raise ValueError(bits)

def unpack(p, bits, K):
    N = p.shape[0]
    if bits == 8:
        return p.view(np.int8).reshape(N, K)
    if bits == 4:
        b = p.reshape(N, K // 32, 16)
        u = np.concatenate([b & 0xF, b >> 4], axis=2)
    elif bits == 6:
        b = p.reshape(N, K // 32, 24)
        ql, qh = b[:, :, :16], b[:, :, 16:]
        lo = np.concatenate([ql & 0xF, ql >> 4], axis=2)                    # 32 low nibbles
        hi = np.concatenate([(qh >> 0) & 3, (qh >> 2) & 3, (qh >> 4) & 3, (qh >> 6) & 3], axis=2)
        u = lo | (hi << 4)
    else:
        raise ValueError(bits)
    return (u.astype(np.int16) - (1 << (bits - 1))).astype(np.int8).reshape(N, K)

if __name__ == "__main__":
    rng = np.random.RandomState(0)
    W = rng.randn(64, 512) * 0.02
    for bits in (4, 6, 8):
        q = quantize_linear(W, bits)
        err = np.abs(dequant(q) - W).max() / np.abs(W).max()
        rt = unpack(pack(q["wq"], bits), bits, 512)
        assert np.array_equal(rt, q["wq"]), bits
        print(bits, "bits: max rel err", err, "pack ok")
