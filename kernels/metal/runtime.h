// det Metal runtime: device, queue, residency set, buffer registry, shader library, pipeline cache, dispatch.
#pragma once
#import <Metal/Metal.h>
#include <stdint.h>
#include <stddef.h>
#include <string>
#include <vector>

namespace det {

struct FnConst { int index; int value; };   // integer function constants select kernel variants

// Translate a host pointer (from det_malloc, possibly offset) into its GPU address.  Aborts on unknown pointers.
uint64_t gpu(const void* host);
uint64_t gpu_or_zero(const void* host);     // NULL -> 0

// Encode one compute dispatch of `name` (with the given function constants) into the current command buffer.
// `params` (size `psize`) is passed as buffer(0) via setBytes; `tgmem` bytes of threadgroup memory at index 0 (optional).
void dispatch(const char* name, const std::vector<FnConst>& consts, const void* params, size_t psize,
              MTLSize grid_tg, MTLSize tg, size_t tgmem = 0);
inline void dispatch(const char* name, const void* params, size_t psize, MTLSize grid_tg, MTLSize tg, size_t tgmem = 0) {
    dispatch(name, {}, params, psize, grid_tg, tg, tgmem);
}
void fill_u32(uint64_t gpu_addr, uint32_t value, size_t n_words);   // GPU-side fill (stream ordered)
size_t dispatch_count();
[[noreturn]] void fail(const char* fmt, ...);
int num_cores();   // GPU core count (best effort; 0 if unknown)

}
