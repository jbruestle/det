// det CUDA runtime (see det_cuda.h): device init, device buffers, synchronization, test hooks.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mutex>
#include <unordered_map>
#include "common.cuh"
#include "det_cuda.h"
using namespace det;

namespace {

struct Ctx {
    int dev = 0;
    char name[256] = {0};
    std::mutex mu;
    std::unordered_map<void*, size_t> allocs;
    size_t allocated = 0;
};
Ctx* g = nullptr;
std::once_flag g_once;

void init_once() {
    g = new Ctx();
    DET_CHECK(cudaGetDevice(&g->dev));
    cudaDeviceProp p;
    DET_CHECK(cudaGetDeviceProperties(&p, g->dev));
    snprintf(g->name, sizeof g->name, "%s", p.name);
}
inline void ensure_init() { std::call_once(g_once, init_once); }

void* track(void* p, size_t bytes) {
    std::lock_guard<std::mutex> lock(g->mu);
    g->allocs[p] = bytes;
    g->allocated += bytes;
    return p;
}

// ---- test hook kernels: the elementwise fp32 primitives (op numbers in det_cuda.h) and the spec tree sum
__global__ void k_prim(int op, const float* __restrict__ x, int n, float* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    float r = 0.0f;
    switch (op) {
        case 0: r = exp_f(v); break;
        case 1: r = log_f(v); break;
        case 2: r = softplus_f(v); break;
        case 3: r = sigmoid_f(v); break;
        case 4: r = silu_f(v); break;
        case 5: { if (i & 1) return; unsigned long long lo = __float_as_uint(x[i]), hi = __float_as_uint(x[i + 1]); r = i64_to_f((long long)((hi << 32) | lo)); break; }
        case 6: r = rint_(v); break;
        case 7: r = div(1.0f, v); break;
        case 8: r = sqrt_(v); break;
        case 9: r = mul(v, x[n - 1 - i]); break;
        case 10: r = add(v, x[n - 1 - i]); break;
        case 12: { if (i & 1) return; unsigned long long lo = __float_as_uint(x[i]), hi = __float_as_uint(x[i + 1]); r = (float)((long long)((hi << 32) | lo)); break; }
        case 13: r = (float)rint_i(v); break;
        case 14: r = div(v, x[n - 1 - i]); break;
        case 15: r = sub(v, x[n - 1 - i]); break;
        case 16: r = fmaxf(v, x[n - 1 - i]); break;
        default: r = v;
    }
    out[i] = r;
}

// treesum of rows of n (n <= 8192): 1024 threads, 8 contiguous values per thread, zero padded
__global__ void __launch_bounds__(1024) k_prim_treesum(const float* __restrict__ x, int n, float* __restrict__ out) {
    __shared__ float smem[32];
    const float* row = x + (size_t)blockIdx.x * n;
    float v[8];
#pragma unroll
    for (int i = 0; i < 8; i++) { const int idx = threadIdx.x * 8 + i; v[i] = idx < n ? row[idx] : 0.0f; }
    float s = warp_tree(tree_local<8>(v));
    s = block_tree<32>(s, smem);
    if (threadIdx.x == 0) out[blockIdx.x] = s;
}

} // namespace

extern "C" {

int det_init(void) { ensure_init(); return g ? 0 : 1; }
const char* det_device_name(void) { ensure_init(); return g->name; }
void* det_cuda_stream(void) { ensure_init(); return (void*)(cudaStream_t)0; }

void* det_malloc(size_t bytes) {
    ensure_init();
    if (bytes == 0) bytes = 16;
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) { fprintf(stderr, "det_malloc(%zu) failed\n", bytes); return NULL; }
    DET_CHECK(cudaMemsetAsync(p, 0, bytes, 0));
    return track(p, bytes);
}

void det_free(void* p) {
    if (!p || !g) return;
    size_t bytes;
    {
        std::lock_guard<std::mutex> lock(g->mu);
        auto it = g->allocs.find(p);
        if (it == g->allocs.end()) { fprintf(stderr, "det_free: %p is not a det_malloc pointer\n", p); abort(); }
        bytes = it->second;
        g->allocs.erase(it);
        g->allocated -= bytes;
    }
    DET_CHECK(cudaDeviceSynchronize());   // pending work may still read the buffer
    DET_CHECK(cudaFree(p));
}

size_t det_allocated_bytes(void) { return g ? g->allocated : 0; }

void det_sync(void) {
    ensure_init();
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { fprintf(stderr, "det (cuda) fatal: %s\n", cudaGetErrorString(e)); abort(); }
}
void det_flush(void) {}

void det_memset(void* p, int value, size_t bytes, cudaStream_t st) { DET_CHECK(cudaMemsetAsync(p, value, bytes, st)); }
void det_memcpy(void* dst, const void* src, size_t bytes, cudaStream_t st) { DET_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, st)); }

void det_test_prim(int op, const float* x, int M, int n, float* out, cudaStream_t st) {
    if (op == 11) k_prim_treesum<<<M, 1024, 0, st>>>(x, n, out);
    else k_prim<<<(unsigned)(((size_t)M * n + 255) / 256), 256, 0, st>>>(op, x, n, out);
    DET_CHECK(cudaGetLastError());
}

}
