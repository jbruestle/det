// Deterministic sampling (detref/sample.py): temperature, top-k, top-p, multinomial.  Sort-free parallel structure:
//   l_i = mul(logit_i, inv_temp); mx = max_i l_i; w_i = (u64) rint( mul( exp_f( sub(l_i, mx) ), 2^32 ) )
//   key_i = (w_i << 20) | (0xFFFFF - i)   (w desc, index asc order == key desc order; keys are unique)
//   top-k: the cut key = the k-th largest key; top-p: the cut key = the largest key K with sum_{key >= K} w >= thresh,
//   thresh = (total * p24 + 2^24 - 1) >> 24.  Both are found by 7 radix-select passes over 8-bit digits (count and
//   weight histograms per pass; k_smp_hist writes per-block partial histograms, k_smp_scan folds them and picks the
//   digit).  kept(i) <=> key_i >= max(cut keys).  Selection: total = sum of kept w, u = z mod total, the smallest i
//   with kept-prefix sum > u (segment sums in k_smp_part, scan + walk in k_smp_select).  All integer arithmetic after
//   exp_f is exact, so the decomposition is free to differ from the reference's loops.
using namespace det;

#define SNT 1024
#define SNB 32                 // segments per row for the selection passes
#define SKEY_SHIFT 20
struct smp_state { ulong prefix_k, prefix_p, remk, remthresh, total, cutkey; };

kernel void k_smp_init(constant det_p_smp& p [[buffer(0)]], uint row [[thread_position_in_grid]]) {
    device smp_state* st = (device smp_state*)p.state + row;
    st->prefix_k = 0; st->prefix_p = 0; st->remk = (ulong)p.top_k; st->remthresh = 0; st->total = 0; st->cutkey = 0;
}
inline ulong smp_key(ulong w, int i) { return (w << SKEY_SHIFT) | (ulong)(0xFFFFF - i); }

kernel void k_smp_max(constant det_p_smp& p [[buffer(0)]], uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float smem[32];
    device const float* lg = (device const float*)p.logits + (ulong)row * p.V;
    float m = -INFINITY;
    for (int i = tid; i < p.V; i += SNT) m = fmax_(m, mul(lg[i], p.inv_temp));
    m = block_max<32>(m, smem, tid);
    if (tid == 0) ((device float*)p.mx)[row] = m;
}

kernel void k_smp_weights(constant det_p_smp& p [[buffer(0)]], uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int row = tg.y, V = p.V;
    device const float* lg = (device const float*)p.logits + (ulong)row * V;
    device ulong* keys = (device ulong*)p.keys + (ulong)row * V;
    const float m = ((device const float*)p.mx)[row];
    for (int i = tg.x * SNT + tid; i < V; i += p.nblk * SNT) {
        const float w = mul(exp_f(sub(mul(lg[i], p.inv_temp), m)), 4294967296.0f);
        keys[i] = smp_key((ulong)rint_(w), i);
    }
}

// per-block partial histograms of digit `digit` (8 bits) for keys matching the current prefixes:
//   count (top-k select, prefix_k) and weight sum as lo/hi 32-bit words with carries (top-p select, prefix_p)
kernel void k_smp_hist(constant det_p_smp& p [[buffer(0)]], uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint hc[256], hlo[256], hhi[256];
    const int row = tg.y, V = p.V, blk = tg.x;
    for (int i = tid; i < 256; i += SNT) { atomic_store_explicit(&hc[i], 0u, memory_order_relaxed); atomic_store_explicit(&hlo[i], 0u, memory_order_relaxed); atomic_store_explicit(&hhi[i], 0u, memory_order_relaxed); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const ulong* keys = (device const ulong*)p.keys + (ulong)row * V;
    device const smp_state* st = (device const smp_state*)p.state + row;
    const int sh = p.digit * 8;
    const ulong pk = st->prefix_k, pp = st->prefix_p;
    for (int i = blk * SNT + tid; i < V; i += p.nblk * SNT) {
        const ulong key = keys[i];
        const ulong above = key >> (sh + 8);
        const uint d = (uint)((key >> sh) & 0xFFul);
        if (p.use_k && above == pk) atomic_fetch_add_explicit(&hc[d], 1u, memory_order_relaxed);
        if (p.use_p && above == pp) {
            const ulong w = key >> SKEY_SHIFT;
            const uint lo = (uint)w, hi = (uint)(w >> 32);
            const uint old = atomic_fetch_add_explicit(&hlo[d], lo, memory_order_relaxed);
            uint carry = (old + lo < old) ? 1u : 0u;
            if (hi + carry) atomic_fetch_add_explicit(&hhi[d], hi + carry, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device uint* out = (device uint*)p.hist + ((ulong)row * p.nblk + blk) * 768;
    for (int i = tid; i < 256; i += SNT) {
        out[i] = atomic_load_explicit(&hc[i], memory_order_relaxed);
        out[256 + i] = atomic_load_explicit(&hlo[i], memory_order_relaxed);
        out[512 + i] = atomic_load_explicit(&hhi[i], memory_order_relaxed);
    }
}

// fold the partial histograms and pick the digit of each select (one threadgroup of 256 per row)
kernel void k_smp_scan(constant det_p_smp& p [[buffer(0)]], uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup ulong cnt[256], sum[256];
    device const uint* h = (device const uint*)p.hist + (ulong)row * p.nblk * 768;
    ulong c = 0, s = 0;
    for (int b = 0; b < p.nblk; b++) {
        c += h[b * 768 + tid];
        s += (ulong)h[b * 768 + 256 + tid] + ((ulong)h[b * 768 + 512 + tid] << 32);
    }
    cnt[tid] = c; sum[tid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        device smp_state* st = (device smp_state*)p.state + row;
        const int digit = p.digit;
        if (digit == 6 && p.use_p) {      // first pass: all keys match the empty prefix -> total of all weights, threshold
            ulong total = 0;
            for (int d = 0; d < 256; d++) total += sum[d];
            const ulong th = total >> 32, tl = total & 0xFFFFFFFFul;
            const ulong A = th * p.p24, B = tl * p.p24 + ((1ul << 24) - 1ul);
            st->remthresh = (A << 8) + (B >> 24);
        }
        if (p.use_k) {
            ulong need = st->remk, acc = 0; int dsel = 0;
            for (int d = 255; d >= 0; d--) { if (acc + cnt[d] >= need) { dsel = d; break; } acc += cnt[d]; }
            st->remk = need - acc;
            st->prefix_k = (st->prefix_k << 8) | (ulong)dsel;
        }
        if (p.use_p) {
            ulong need = st->remthresh, acc = 0; int dsel = 0;
            for (int d = 255; d >= 0; d--) { if (acc + sum[d] >= need) { dsel = d; break; } acc += sum[d]; }
            st->remthresh = need - acc;
            st->prefix_p = (st->prefix_p << 8) | (ulong)dsel;
        }
        if (digit == 0) {
            ulong ck = p.use_k ? st->prefix_k : 0ul, cp = p.use_p ? st->prefix_p : 0ul;
            st->cutkey = ck > cp ? ck : cp;
        }
    }
}

// segment sums of the kept weights (index order) -> part[row][SNB]
kernel void k_smp_part(constant det_p_smp& p [[buffer(0)]], uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup ulong sw[32];
    const int row = tg.y, V = p.V, seg = (((V + SNB - 1) / SNB) + 31) & ~31;
    const int b = tg.x * seg, e = min(V, b + seg);
    device const ulong* keys = (device const ulong*)p.keys + (ulong)row * V;
    const ulong cut = ((device const smp_state*)p.state)[row].cutkey;
    ulong acc = 0;
    for (int i = b + tid; i < e; i += SNT) { const ulong key = keys[i]; if (key >= cut) acc += key >> SKEY_SHIFT; }
    for (int o = 16; o > 0; o >>= 1) {
        uint lo = (uint)acc, hi = (uint)(acc >> 32);
        lo = simd_shuffle_xor(lo, (ushort)o); hi = simd_shuffle_xor(hi, (ushort)o);
        acc += ((ulong)hi << 32) | lo;
    }
    if ((tid & 31) == 0) sw[tid >> 5] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { ulong t = 0; for (int i = 0; i < 32; i++) t += sw[i]; ((device ulong*)p.part)[row * SNB + tg.x] = t; }
}

// u = z mod total; find the segment holding the crossing, then walk it: chunk sums per thread, scan, locate
kernel void k_smp_select(constant det_p_smp& p [[buffer(0)]], uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup ulong parts[SNB + 1];
    threadgroup ulong csum[SNT];
    threadgroup int res;
    const int V = p.V, seg = (((V + SNB - 1) / SNB) + 31) & ~31;
    device const ulong* part = (device const ulong*)p.part + row * SNB;
    if (tid == 0) {
        ulong acc = 0;
        for (int s = 0; s < SNB; s++) { parts[s] = acc; acc += part[s]; }
        parts[SNB] = acc;
        res = -1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ulong total = parts[SNB];
    const ulong u = ((device const ulong*)p.z)[row] % total;
    int s = 0;
    for (int i = 0; i < SNB; i++) if (parts[i + 1] > u) { s = i; break; }
    const ulong u_local = u - parts[s];
    const int b = s * seg, e = min(V, b + seg);
    device const ulong* keys = (device const ulong*)p.keys + (ulong)row * V;
    const ulong cut = ((device const smp_state*)p.state)[row].cutkey;
    // thread t owns [b + t*cw, b + (t+1)*cw)
    const int cw = (e - b + SNT - 1) / SNT;
    const int tb = b + tid * cw, te = min(e, tb + cw);
    ulong acc = 0;
    for (int i = tb; i < te; i++) { const ulong key = keys[i]; if (key >= cut) acc += key >> SKEY_SHIFT; }
    csum[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { ulong a = 0; for (int i = 0; i < SNT; i++) { ulong v = csum[i]; csum[i] = a; a += v; } }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ulong before = csum[tid];
    if (before <= u_local && u_local < before + acc) {
        ulong a = before;
        for (int i = tb; i < te; i++) {
            const ulong key = keys[i];
            if (key >= cut) { a += key >> SKEY_SHIFT; if (a > u_local) { res = i; break; } }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) ((device int*)p.out)[row] = res;
}
