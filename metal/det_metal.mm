// det Metal runtime (see runtime.h / det_metal.h).
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include "det_metal.h"
#include "runtime.h"
#include "params.h"

namespace det {

struct Alloc { id<MTLBuffer> buf; size_t bytes; };

struct Ctx {
    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> q = nil;
    id<MTLResidencySet> rs = nil;
    id<MTLLibrary> lib = nil;
    id<MTLCommandBuffer> cb = nil;
    id<MTLComputeCommandEncoder> enc = nil;
    id<MTLCommandBuffer> last = nil;             // most recently committed command buffer (the queue runs in order)
    std::map<std::string, id<MTLComputePipelineState>> psos;
    std::map<uintptr_t, Alloc> allocs;          // host base address -> buffer
    NSMutableArray<id<MTLBuffer>>* all_bufs = nil;   // every live buffer (declared to each encoder via useResources)
    size_t allocated = 0;
    size_t ndispatch = 0, ndispatch_cb = 0;
    bool inited = false;
    int cores = 0;
};
static Ctx* g = nullptr;
static std::once_flag g_once;

[[noreturn]] void fail(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "det (metal) fatal: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    abort();
}

static std::string shader_dir() {
    const char* env = getenv("DET_METAL_SHADERS");
    if (env) return env;
    Dl_info info;
    if (dladdr((const void*)&det_init, &info) && info.dli_fname) {
        std::string p = info.dli_fname;
        size_t s = p.rfind('/');
        p = (s == std::string::npos) ? "." : p.substr(0, s);
        return p + "/shaders";
    }
    return "shaders";
}

static std::string read_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) fail("cannot open shader source %s", path.c_str());
    std::string s;
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
    fclose(f);
    return s;
}

static void init_once() {
    @autoreleasepool {
        g = new Ctx();
        g->dev = MTLCreateSystemDefaultDevice();
        if (!g->dev) fail("no Metal device");
        g->q = [g->dev newCommandQueue];
        g->all_bufs = [NSMutableArray new];
        NSError* err = nil;
        MTLResidencySetDescriptor* rd = [MTLResidencySetDescriptor new];
        rd.initialCapacity = 4096;
        g->rs = [g->dev newResidencySetWithDescriptor:rd error:&err];
        if (!g->rs) fail("residency set: %s", err.localizedDescription.UTF8String);
        [g->q addResidencySet:g->rs];
        // shader library: params.h + common.metal + every other *.metal (sorted) as one translation unit
        std::string dir = shader_dir();
        std::string src = "#define __DET_METAL__ 1\n";
        if (const char* defs = getenv("DET_METAL_DEFINES")) {   // "A=1,B=2" -> #define lines (experiments)
            std::string d = defs; size_t pos = 0;
            while (pos <= d.size()) {
                size_t e = d.find(',', pos); if (e == std::string::npos) e = d.size();
                std::string kv = d.substr(pos, e - pos); size_t eq = kv.find('=');
                if (!kv.empty()) src += "#define " + (eq == std::string::npos ? kv : kv.substr(0, eq) + " " + kv.substr(eq + 1)) + "\n";
                pos = e + 1;
            }
        }
        src += read_file(dir + "/../params.h") + "\n" + read_file(dir + "/common.metal") + "\n";
        NSArray* files = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:[NSString stringWithUTF8String:dir.c_str()] error:&err];
        if (!files) fail("cannot list shader dir %s", dir.c_str());
        std::vector<std::string> names;
        for (NSString* f in files) if ([f hasSuffix:@".metal"] && ![f isEqualToString:@"common.metal"]) names.push_back(f.UTF8String);
        std::sort(names.begin(), names.end());
        for (auto& n : names) src += "\n// ===== " + n + "\n" + read_file(dir + "/" + n) + "\n";
        MTLCompileOptions* opt = [MTLCompileOptions new];
        opt.mathMode = MTLMathModeSafe;                                    // no fast math, no contraction
        opt.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        opt.languageVersion = MTLLanguageVersion4_0;
        g->lib = [g->dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()] options:opt error:&err];
        if (!g->lib) fail("shader compilation failed:\n%s", err.localizedDescription.UTF8String);
        if (getenv("DET_METAL_VERBOSE")) fprintf(stderr, "det: shaders compiled from %s (%zu bytes)\n", dir.c_str(), src.size());
        g->inited = true;
    }
}

static inline void ensure_init() { std::call_once(g_once, init_once); }

int num_cores() { ensure_init(); return g->cores; }
size_t dispatch_count() { return g ? g->ndispatch : 0; }

uint64_t gpu_or_zero(const void* host) { return host ? gpu(host) : 0; }
uint64_t gpu(const void* host) {
    ensure_init();
    uintptr_t h = (uintptr_t)host;
    auto it = g->allocs.upper_bound(h);
    if (it == g->allocs.begin()) fail("pointer %p is not inside a det_malloc buffer", host);
    --it;
    if (h >= it->first + it->second.bytes) fail("pointer %p is not inside a det_malloc buffer (nearest base %p, %zu bytes)", host, (void*)it->first, it->second.bytes);
    return it->second.buf.gpuAddress + (h - it->first);
}

static id<MTLComputePipelineState> pso_for(const char* name, const std::vector<FnConst>& consts) {
    std::string key = name;
    for (auto& c : consts) key += "|" + std::to_string(c.index) + "=" + std::to_string(c.value);
    auto it = g->psos.find(key);
    if (it != g->psos.end()) return it->second;
    @autoreleasepool {
        NSError* err = nil;
        id<MTLFunction> fn;
        if (consts.empty()) {
            fn = [g->lib newFunctionWithName:[NSString stringWithUTF8String:name]];
        } else {
            MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
            for (auto& c : consts) { int v = c.value; [cv setConstantValue:&v type:MTLDataTypeInt atIndex:c.index]; }
            fn = [g->lib newFunctionWithName:[NSString stringWithUTF8String:name] constantValues:cv error:&err];
        }
        if (!fn) fail("kernel %s not found (%s)", key.c_str(), err ? err.localizedDescription.UTF8String : "");
        id<MTLComputePipelineState> p = [g->dev newComputePipelineStateWithFunction:fn error:&err];
        if (!p) fail("pipeline %s: %s", key.c_str(), err.localizedDescription.UTF8String);
        g->psos[key] = p;
        return p;
    }
}

static void ensure_encoder() {
    if (!g->cb) { g->cb = [g->q commandBuffer]; g->ndispatch_cb = 0; }
    // Bindless access (raw GPU addresses) is invisible to Metal's hazard tracking unless the buffers are declared with
    // useResources: every live buffer is declared read+write on each (serial) encoder, so Metal orders every dispatch
    // after the previous one.  DET_METAL_BARRIER=1 additionally inserts an explicit memory barrier after each dispatch
    // (2x slower decode; debugging aid).
    if (!g->enc) {
        static const bool explicit_barrier = getenv("DET_METAL_BARRIER") != nullptr;
        g->enc = explicit_barrier ? [g->cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent] : [g->cb computeCommandEncoder];
        // every dispatch may read or write any buffer: declaring them all lets Metal's hazard tracking serialise
        // consecutive dispatches (residency is handled by the residency set; this call is about tracking)
        if (g->all_bufs.count) {
            std::vector<id<MTLBuffer>> v(g->all_bufs.count);
            for (NSUInteger i = 0; i < g->all_bufs.count; i++) v[i] = g->all_bufs[i];
            [g->enc useResources:v.data() count:v.size() usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        }
    }
}

static void commit(bool wait) {
    if (g->enc) { [g->enc endEncoding]; g->enc = nil; }
    if (g->cb) {
        [g->cb commit];
        g->last = g->cb;
        g->cb = nil;
    }
    if (wait && g->last) {
        [g->last waitUntilCompleted];
        if (g->last.error) fail("command buffer failed: %s", g->last.error.localizedDescription.UTF8String);
        g->last = nil;
    }
}

void dispatch(const char* name, const std::vector<FnConst>& consts, const void* params, size_t psize,
              MTLSize grid_tg, MTLSize tg, size_t tgmem) {
    ensure_init();
    if (grid_tg.width == 0 || grid_tg.height == 0 || grid_tg.depth == 0) return;
    id<MTLComputePipelineState> p = pso_for(name, consts);
    ensure_encoder();
    [g->enc setComputePipelineState:p];
    [g->enc setBytes:params length:psize atIndex:0];
    if (tgmem) [g->enc setThreadgroupMemoryLength:tgmem atIndex:0];
    [g->enc dispatchThreadgroups:grid_tg threadsPerThreadgroup:tg];
    static const bool explicit_barrier = getenv("DET_METAL_BARRIER") != nullptr;
    if (explicit_barrier) [g->enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    g->ndispatch++;
    static const bool sync_every = getenv("DET_METAL_SYNC_EVERY") != nullptr;   // debugging aid
    if (sync_every) { commit(true); return; }
    // keep the GPU busy during long encodes: commit every so often (command buffers on one queue run in order)
    if (++g->ndispatch_cb >= 256) commit(false);
}

void fill_u32(uint64_t addr, uint32_t value, size_t n) {
    det_p_fill p = { addr, value, (uint64_t)n };
    const size_t per = 256 * 8;   // 8 words per thread
    dispatch("k_fill_u32", &p, sizeof p, MTLSizeMake((n + per - 1) / per, 1, 1), MTLSizeMake(256, 1, 1));
}

} // namespace det
using namespace det;

extern "C" {

int det_init(void) { ensure_init(); return g->inited ? 0 : 1; }
const char* det_device_name(void) { ensure_init(); return g->dev.name.UTF8String; }

void* det_malloc(size_t bytes) {
    ensure_init();
    if (bytes == 0) bytes = 16;
    @autoreleasepool {
        id<MTLBuffer> b = [g->dev newBufferWithLength:bytes options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked];
        if (!b) { fprintf(stderr, "det_malloc(%zu) failed\n", bytes); return NULL; }
        memset(b.contents, 0, bytes);
        [g->rs addAllocation:b];
        [g->rs commit];
        g->allocs[(uintptr_t)b.contents] = Alloc{ b, bytes };
        [g->all_bufs addObject:b];
        if (g->enc) [g->enc useResource:b usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        g->allocated += bytes;
        return b.contents;
    }
}

void det_free(void* p) {
    if (!p || !g) return;
    auto it = g->allocs.find((uintptr_t)p);
    if (it == g->allocs.end()) fail("det_free: %p is not a det_malloc base pointer", p);
    commit(true);   // the buffer may be referenced by pending work
    @autoreleasepool {
        [g->all_bufs removeObjectIdenticalTo:it->second.buf];
        [g->rs removeAllocation:it->second.buf];
        [g->rs commit];
    }
    g->allocated -= it->second.bytes;
    g->allocs.erase(it);
}

size_t det_allocated_bytes(void) { return g ? g->allocated : 0; }
void det_sync(void) { ensure_init(); commit(true); }
void det_flush(void) { ensure_init(); commit(false); }

void det_memset(void* p, int value, size_t bytes, det_stream_t) {
    if (bytes % 4) fail("det_memset: bytes must be a multiple of 4");
    uint32_t v = (uint32_t)(value & 0xFF) * 0x01010101u;
    fill_u32(gpu(p), v, bytes / 4);
}

void det_memcpy(void* dst, const void* src, size_t bytes, det_stream_t) {
    if (bytes % 4) fail("det_memcpy: bytes must be a multiple of 4");
    det_p_copy p = { gpu(dst), gpu(src), (uint64_t)(bytes / 4) };
    const size_t per = 256 * 4;
    dispatch("k_copy_u32", &p, sizeof p, MTLSizeMake((bytes / 4 + per - 1) / per, 1, 1), MTLSizeMake(256, 1, 1));
}

void det_test_prim(int op, const float* x, int M, int n, float* out, det_stream_t) {
    det_p_prim p = { gpu(x), gpu(out), op, n };
    if (op == 11) dispatch("k_prim_treesum", &p, sizeof p, MTLSizeMake(M, 1, 1), MTLSizeMake(1024, 1, 1));
    else dispatch("k_prim", &p, sizeof p, MTLSizeMake(((size_t)M * n + 255) / 256, 1, 1), MTLSizeMake(256, 1, 1));
}

} // extern "C"
