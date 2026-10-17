"""Deterministic sampling (SPEC).  Given fp32 logits[V], temperature, top_k, top_p and a 64-bit RNG state:
  1. l_i = logits_i * inv_temp   (inv_temp = fp32(1/temperature) computed on the host, exact mul)
  2. mx = max_i l_i;  p_i = exp_f(l_i - mx)                       (fp32, elementwise)
  3. w_i = rint(p_i * 2^32)  as uint64  (integer weights; w of the argmax = 2^32)
  4. optional top-k: keep the k largest w_i (ties -> lower index), others 0
     optional top-p: sort by (w desc, index asc); keep the smallest prefix whose integer sum >= ceil(top_p * total); others 0
  5. total = sum_i w_i (exact int); u = rng_next() mod total (unsigned 64-bit); choose smallest i with prefix_sum(i) > u
RNG: splitmix64 (pure 64-bit integer arithmetic), seeded by the user; one draw per token.
"""
import numpy as np
from detmath import exp_f, f32

MASK = (1 << 64) - 1
def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    z = z ^ (z >> 31)
    return state, z

def sample(logits, rng_state, temperature=1.0, top_k=0, top_p=1.0):
    logits = np.asarray(logits, dtype=f32)
    if temperature <= 0:
        return int(np.argmax(logits)), rng_state       # greedy: lowest index on ties (np.argmax does this)
    l = logits * f32(1.0 / temperature)
    p = exp_f(l - l.max())
    w = np.rint(p * f32(4294967296.0)).astype(np.float64).astype(np.uint64)   # exact: p*2^32 <= 2^32 representable in f32
    V = len(w)
    if (top_k and top_k < V) or top_p < 1.0:
        order = np.lexsort((np.arange(V), -w.astype(np.int64)))   # w desc, index asc
        keep = np.ones(V, dtype=bool)
        if top_k and top_k < V:
            keep[order[top_k:]] = False
        if top_p < 1.0:
            ws = w[order].astype(object)
            total = sum(ws)
            # integer threshold: ceil(top_p * total) computed in integers from the fp32 top_p (the C API takes a float):
            # thresh = (total * round_half_even(top_p * 2^24) + 2^24 - 1) >> 24
            thresh = (int(total) * int(round(float(f32(top_p)) * 2**24)) + (1 << 24) - 1) >> 24
            acc = 0; cut = V
            for j in range(V):
                acc += int(ws[j])
                if acc >= thresh: cut = j + 1; break
            keep[order[cut:]] = False
        w = np.where(keep, w, np.uint64(0))
    total = int(w.astype(object).sum())
    rng_state, z = splitmix64(rng_state)
    u = z % total
    cs = np.cumsum(w.astype(object))
    idx = int(np.searchsorted(np.array(cs, dtype=object), u, side="right"))
    return idx, rng_state

if __name__ == "__main__":
    rng = np.random.RandomState(0)
    lg = (rng.randn(1000) * 3).astype(f32)
    st = 12345; counts = np.zeros(1000)
    for _ in range(2000):
        i, st = sample(lg, st, temperature=0.8, top_k=50, top_p=0.95); counts[i] += 1
    print("top sampled:", np.argsort(-counts)[:5], "argmax:", lg.argmax())
    print("greedy:", sample(lg, 0, temperature=0)[0])
