// Gated DeltaNet layer core (bit-exact w.r.t. detref/model.py: gdn_core).  Port of cuda/gdn.cu:
//   1. k_gdn_prep  : conv1d(4-tap, state = previous 3 raw inputs) + silu, l2norm of q/k heads (q *= QSCALE),
//                    beta = sigmoid(b), decay = exp(-(A * softplus(a + dt_bias)))        -> scratch
//   2. k_gdn_recur : the strictly sequential recurrence (hardware fast path with per-result checks, exact fallback per
//                    simdgroup, see below).  One threadgroup owns JB=32 columns (v dims) of one v-head; a
//                    column S[:, j] (128 k dims) is split over P=4 lanes of one simdgroup (R=32 registers each).  The
//                    spec tree over i is tree_local<R> in registers followed by a butterfly with ascending xor offsets.
//                    k/q of the current token are broadcast through (double buffered) threadgroup memory.
//   3. k_gdn_out   : on = rmsnorm(o, norm, eps) * silu(z)   (+ conv_state update, last chunk)
using namespace det;

#define GDN_HD 128
constant float QSCALE_GDN = 0.08838834764831845f;   // fp32(1/sqrt(128)) = 0x3DB504F3

// grid = (ceil(conv_dim/512), M_chunk) threadgroups of 128 (4 simdgroups; one per 128 channels, 4 contiguous per lane)
kernel void k_gdn_prep(constant det_p_gdn_prep& p [[buffer(0)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    device const float* qkv = (device const float*)p.qkv;
    device const float* conv_state = (device const float*)p.conv_state;
    device const float* conv_w = (device const float*)p.conv_w;
    device const float* a = (device const float*)p.a;
    device const float* b = (device const float*)p.b;
    device const float* A = (device const float*)p.A;
    device const float* dt_bias = (device const float*)p.dt_bias;
    device float* y = (device float*)p.y;
    device float* beta = (device float*)p.beta;
    device float* decay = (device float*)p.decay;
    const int t = tgid.y, tg = p.t0 + t;
    const int lane = tid & 31, w = tid >> 5;
    const int conv_dim = p.conv_dim, kd = p.kd, n_v = p.n_v;
    const int base = (tgid.x * 4 + w) * GDN_HD;
    if (base < conv_dim) {
        const int c0 = base + lane * 4;
        float4 x[4];
        DET_U_GDN for (int m = 0; m < 4; m++) {
            const int tt = tg - 3 + m;
            device const float* src = (tt >= 0) ? (qkv + (ulong)tt * conv_dim) : (conv_state + (ulong)(tt + 3) * conv_dim);
            x[m] = *(device const float4*)(src + c0);
        }
        float yv[4];
        DET_U_GDN for (int i = 0; i < 4; i++) {
            const float4 wc = *(device const float4*)(conv_w + (ulong)(c0 + i) * 4);
            const float x0 = x[0][i], x1 = x[1][i], x2 = x[2][i], x3 = x[3][i];
            float v = add(mul(wc.x, x0), mul(wc.y, x1));
            v = add(v, mul(wc.z, x2));
            v = add(v, mul(wc.w, x3));
            yv[i] = silu_f(v);
        }
        if (base < 2 * kd) {
            float sq[4];
            DET_U_GDN for (int i = 0; i < 4; i++) sq[i] = mul(yv[i], yv[i]);
            const float ss = warp_tree(tree_local<4>(sq));
            const float inv = div(1.0f, sqrt_(add(ss, p.eps)));
            DET_U_GDN for (int i = 0; i < 4; i++) yv[i] = mul(yv[i], inv);
            if (base < kd) { DET_U_GDN for (int i = 0; i < 4; i++) yv[i] = mul(yv[i], QSCALE_GDN); }
        }
        *(device float4*)(y + (ulong)t * conv_dim + c0) = float4(yv[0], yv[1], yv[2], yv[3]);
    }
    if (tgid.x == 0 && (int)tid < n_v) {
        const int h = tid;
        beta[(ulong)t * n_v + h] = sigmoid_f(b[(ulong)tg * n_v + h]);
        const float sp = softplus_f(add(a[(ulong)tg * n_v + h], dt_bias[h]));
        decay[(ulong)t * n_v + h] = exp_f(-(mul(A[h], sp)));
    }
}

// JB columns of one v-head per threadgroup, P lanes per column (R = 128/P k dims each), NT = JB*P = 128 threads.
#define JB 32
#define PP 4
#define RR 32
#define LPP 8
#define KS 36
// Fast path: the token step with raw hardware ops, every result checked with a 2-op test for zero / denormal /
// minimum-normal-binade values (the only cases where the flushing hardware result can differ from IEEE); if any lane
// of the simdgroup flags a value (or the token's k/q/v/decay/beta inputs are flagged), the whole simdgroup redoes the
// step with the exact det:: ops from the saved state.  The trees use shuffles across lanes, hence the simdgroup-wide
// decision (simd_any).
inline bool susp(float x) { return (as_type<uint>(x) & 0x7fffffffu) < 0x01000000u; }
#define CHK(x) (bad = bad | susp(x))
inline float tree_local_chk32(thread const float* v, thread bool& bad) {
    float t[32];
    DET_U_TREE for (int i = 0; i < 32; i++) t[i] = v[i];
    DET_U_TREE for (int s = 1; s < 32; s <<= 1) {
        DET_U_TREE for (int i = 0; i < 32; i += 2 * s) { t[i] = t[i] + t[i + s]; CHK(t[i]); }
    }
    return t[0];
}
kernel void k_gdn_recur(constant det_p_gdn_recur& p [[buffer(0)]], uint bid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float skq[2][2][PP * KS];
    threadgroup int badkq[3];   // ring of 3: slot (t+2)%3 is reset at iteration t, filled at t+1, read at t+2
    device const float* y = (device const float*)p.y;
    device const float* beta = (device const float*)p.beta;
    device const float* decay = (device const float*)p.decay;
    device float* S = (device float*)p.S;
    device float* o = (device float*)p.o;
    const int M = p.M, conv_dim = p.conv_dim, kd = p.kd, n_v = p.n_v, rep = p.rep;
    const int lane = tid & 31, w = tid >> 5;
    const int pp = lane / LPP;
    const int j = w * LPP + (lane % LPP);
    const int h = bid / (GDN_HD / JB);
    const int jg = (bid % (GDN_HD / JB)) * JB + j;
    const int kh = h / rep;
    device float* Sp = S + ((ulong)h * GDN_HD + (ulong)pp * RR) * GDN_HD + jg;
    float s[RR];
    DET_U_GDN for (int r = 0; r < RR; r++) s[r] = Sp[(ulong)r * GDN_HD];
    device const float* yq = y + (ulong)kh * GDN_HD;
    device const float* yk = y + kd + (ulong)kh * GDN_HD;
    device const float* yv = y + 2 * kd + (ulong)h * GDN_HD + jg;
    device const float* bh = beta + h;
    device const float* dh = decay + h;
    if (tid < 3) badkq[tid] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tid; i < 2 * GDN_HD; i += 128) {
        const int ii = i & (GDN_HD - 1);
        const float val = (i < GDN_HD) ? yk[ii] : yq[ii];
        skq[0][i >= GDN_HD][(ii / RR) * KS + (ii % RR)] = val;
        if (susp(val)) badkq[0] = 1;
    }
    float vcur = yv[0], bcur = bh[0], dcur = dh[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int t = 0; t < M; t++) {
        float nload[2];
        float vn = 0.0f, bn = 0.0f, dn = 0.0f;
        if (t + 1 < M) {
            const ulong off = (ulong)(t + 1) * conv_dim;
            for (int l = 0; l < 2; l++) {
                const int i = tid + l * 128;
                const int ii = i & (GDN_HD - 1);
                nload[l] = (i < GDN_HD) ? yk[off + ii] : yq[off + ii];
            }
            vn = yv[off]; bn = bh[(ulong)(t + 1) * n_v]; dn = dh[(ulong)(t + 1) * n_v];
        }
        threadgroup const float* kb = &skq[t & 1][0][pp * KS];
        threadgroup const float* qb = &skq[t & 1][1][pp * KS];
        bool bad = badkq[t % 3] != 0;
        bad = bad | susp(vcur) | susp(bcur) | susp(dcur);
        float sold[RR];
        DET_U_GDN for (int r = 0; r < RR; r++) sold[r] = s[r];
        // ---- fast path (hardware ops + checks)
        float prod[RR];
        DET_U_GDN for (int r = 0; r < RR; r += 4) {
            const float4 kk = *(threadgroup const float4*)(kb + r);
            s[r] = s[r] * dcur; s[r + 1] = s[r + 1] * dcur; s[r + 2] = s[r + 2] * dcur; s[r + 3] = s[r + 3] * dcur;
            CHK(s[r]); CHK(s[r + 1]); CHK(s[r + 2]); CHK(s[r + 3]);
            prod[r] = s[r] * kk.x; prod[r + 1] = s[r + 1] * kk.y; prod[r + 2] = s[r + 2] * kk.z; prod[r + 3] = s[r + 3] * kk.w;
            CHK(prod[r]); CHK(prod[r + 1]); CHK(prod[r + 2]); CHK(prod[r + 3]);
        }
        float kv = tree_local_chk32(prod, bad);
        for (int off = LPP; off < 32; off <<= 1) { kv = kv + simd_shuffle_xor(kv, (ushort)off); CHK(kv); }
        float delta = vcur - kv; CHK(delta);
        delta = delta * bcur; CHK(delta);
        DET_U_GDN for (int r = 0; r < RR; r += 4) {
            const float4 kk = *(threadgroup const float4*)(kb + r);
            const float4 qq = *(threadgroup const float4*)(qb + r);
            float u0 = kk.x * delta, u1 = kk.y * delta, u2 = kk.z * delta, u3 = kk.w * delta;
            CHK(u0); CHK(u1); CHK(u2); CHK(u3);
            s[r] = s[r] + u0; s[r + 1] = s[r + 1] + u1; s[r + 2] = s[r + 2] + u2; s[r + 3] = s[r + 3] + u3;
            CHK(s[r]); CHK(s[r + 1]); CHK(s[r + 2]); CHK(s[r + 3]);
            prod[r] = s[r] * qq.x; prod[r + 1] = s[r + 1] * qq.y; prod[r + 2] = s[r + 2] * qq.z; prod[r + 3] = s[r + 3] * qq.w;
            CHK(prod[r]); CHK(prod[r + 1]); CHK(prod[r + 2]); CHK(prod[r + 3]);
        }
        float ov = tree_local_chk32(prod, bad);
        for (int off = LPP; off < 32; off <<= 1) { ov = ov + simd_shuffle_xor(ov, (ushort)off); CHK(ov); }
        if (simd_any(bad)) {
            // ---- exact path (det:: ops, IEEE denormals) from the saved state
            DET_U_GDN for (int r = 0; r < RR; r++) s[r] = mul(sold[r], dcur);
            DET_U_GDN for (int r = 0; r < RR; r += 4) {
                const float4 kk = *(threadgroup const float4*)(kb + r);
                prod[r] = mul(s[r], kk.x); prod[r + 1] = mul(s[r + 1], kk.y);
                prod[r + 2] = mul(s[r + 2], kk.z); prod[r + 3] = mul(s[r + 3], kk.w);
            }
            kv = tree_local<RR>(prod);
            for (int off = LPP; off < 32; off <<= 1) kv = add(kv, simd_shuffle_xor(kv, (ushort)off));
            delta = mul(sub(vcur, kv), bcur);
            DET_U_GDN for (int r = 0; r < RR; r += 4) {
                const float4 kk = *(threadgroup const float4*)(kb + r);
                const float4 qq = *(threadgroup const float4*)(qb + r);
                s[r] = add(s[r], mul(kk.x, delta));         prod[r] = mul(s[r], qq.x);
                s[r + 1] = add(s[r + 1], mul(kk.y, delta)); prod[r + 1] = mul(s[r + 1], qq.y);
                s[r + 2] = add(s[r + 2], mul(kk.z, delta)); prod[r + 2] = mul(s[r + 2], qq.z);
                s[r + 3] = add(s[r + 3], mul(kk.w, delta)); prod[r + 3] = mul(s[r + 3], qq.w);
            }
            ov = tree_local<RR>(prod);
            for (int off = LPP; off < 32; off <<= 1) ov = add(ov, simd_shuffle_xor(ov, (ushort)off));
        }
        if (pp == 0) o[((ulong)t * n_v + h) * GDN_HD + jg] = ov;
        if (t + 1 < M) {
            if (tid == 0) badkq[(t + 2) % 3] = 0;
            for (int l = 0; l < 2; l++) {
                const int i = tid + l * 128;
                const int ii = i & (GDN_HD - 1);
                skq[(t + 1) & 1][i >= GDN_HD][(ii / RR) * KS + (ii % RR)] = nload[l];
                if (susp(nload[l])) badkq[(t + 1) % 3] = 1;
            }
            vcur = vn; bcur = bn; dcur = dn;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    DET_U_GDN for (int r = 0; r < RR; r++) Sp[(ulong)r * GDN_HD] = s[r];
}

// blocks [0, nblk_out): 4 rows (of 128) per block, one simdgroup per row: on = rmsnorm(o,norm,eps) * silu(z)
// blocks [nblk_out, ...): conv_state <- last 3 raw inputs (oldest first); each thread owns one channel
kernel void k_gdn_out(constant det_p_gdn_out& p [[buffer(0)]], uint bid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int lane = tid & 31, w = tid >> 5;
    if ((int)bid < p.nblk_out) {
        const int row = bid * 4 + w;
        if (row >= p.nrows) return;
        device const float* o = (device const float*)p.o;
        device const float* z = (device const float*)p.z;
        device const float* norm = (device const float*)p.norm;
        device float* on = (device float*)p.on;
        const ulong ro = (ulong)row * GDN_HD + lane * 4;
        float x[4], sq[4];
        DET_U_GDN for (int i = 0; i < 4; i++) { x[i] = o[ro + i]; sq[i] = mul(x[i], x[i]); }
        const float ss = warp_tree(tree_local<4>(sq));
        const float mean = div(ss, (float)GDN_HD);
        const float inv = div(1.0f, sqrt_(add(mean, p.eps)));
        DET_U_GDN for (int i = 0; i < 4; i++) {
            const float yv = mul(mul(x[i], inv), norm[lane * 4 + i]);
            on[ro + i] = mul(yv, silu_f(z[ro + i]));
        }
    } else {
        const int c = (bid - p.nblk_out) * 128 + tid;
        if (c >= p.conv_dim) return;
        device const float* qkv = (device const float*)p.qkv;
        device float* conv_state = (device float*)p.conv_state;
        float nv[3];
        DET_U_GDN for (int r = 0; r < 3; r++) {
            const int u = p.M - 3 + r;
            nv[r] = (u >= 0) ? qkv[(ulong)u * p.conv_dim + c] : conv_state[(ulong)(u + 3) * p.conv_dim + c];
        }
        DET_U_GDN for (int r = 0; r < 3; r++) conv_state[(ulong)r * p.conv_dim + c] = nv[r];
    }
}
