/**
 * Nanobind Metal extension for natten-mps.
 *
 * Loads precompiled .metallib shaders and dispatches neighborhood attention
 * kernels via the Metal API. Currently supports 1D QK/AV forward.
 *
 * Design: Since nanobind lacks native torch::Tensor type casters (unlike
 * pybind11), we accept raw buffer pointers (as uint64_t from
 * tensor.data_ptr()) and shape parameters from the Python side. The Python
 * wrapper in nanobind.py handles tensor creation and layout conversion.
 *
 * The key advantage over torch.mps.compile_shader() is that metallib
 * loading + PSO creation is a one-time cost, whereas compile_shader()
 * re-parses MSL source each time (even if cached internally).
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <Metal/Metal.h>
#include <Foundation/Foundation.h>

#include <dlfcn.h>
#include <unordered_map>
#include <mutex>

namespace nb = nanobind;

// ---------------------------------------------------------------------------
// Metal state
// ---------------------------------------------------------------------------

static id<MTLDevice> g_device = nil;
static id<MTLLibrary> g_library = nil;
static std::unordered_map<std::string, id<MTLComputePipelineState>> g_pipelines;
static std::mutex g_mutex;

// ---------------------------------------------------------------------------
// Locate .metallib relative to this shared object
// ---------------------------------------------------------------------------

static std::string get_module_dir() {
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&get_module_dir), &info) && info.dli_fname) {
        std::string path(info.dli_fname);
        auto pos = path.rfind('/');
        if (pos != std::string::npos) {
            return path.substr(0, pos);
        }
    }
    return ".";
}

static bool ensure_metal() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_device) return true;

    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) return false;

    @autoreleasepool {
        std::string dir = get_module_dir();
        std::string libPath = dir + "/shaders/natten_nb_1d.metallib";

        NSString* nsPath = [NSString stringWithUTF8String:libPath.c_str()];
        NSURL* url = [NSURL fileURLWithPath:nsPath];
        NSError* error = nil;
        g_library = [g_device newLibraryWithURL:url error:&error];

        if (!g_library) {
            libPath = dir + "/natten_nb_1d.metallib";
            nsPath = [NSString stringWithUTF8String:libPath.c_str()];
            url = [NSURL fileURLWithPath:nsPath];
            g_library = [g_device newLibraryWithURL:url error:&error];
        }

        if (!g_library) {
            g_device = nil;
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Pipeline state cache
// ---------------------------------------------------------------------------

static id<MTLComputePipelineState> get_pipeline(const std::string& name) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_pipelines.find(name);
    if (it != g_pipelines.end()) return it->second;

    @autoreleasepool {
        NSString* nsName = [NSString stringWithUTF8String:name.c_str()];
        id<MTLFunction> fn = [g_library newFunctionWithName:nsName];
        if (!fn) {
            throw std::runtime_error("Metal function '" + name + "' not found in metallib");
        }

        NSError* error = nil;
        id<MTLComputePipelineState> pso =
            [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!pso) {
            throw std::runtime_error("Failed to create pipeline state for '" + name + "'");
        }

        g_pipelines[name] = pso;
        return pso;
    }
}

// ---------------------------------------------------------------------------
// Dispatch a 1D kernel
//
// Accepts raw MTLBuffer pointers (as uint64_t from Python) plus shape info.
// The Python wrapper creates output tensors and passes their data_ptr() too.
// ---------------------------------------------------------------------------

static void dispatch_1d_kernel(
    const std::string& kernel_name,
    uint64_t buf0_ptr, uint64_t buf1_ptr, uint64_t buf2_ptr,
    int batch_size, int heads, int length, int dim,
    int kernel_size, int dilation)
{
    if (!g_device) {
        throw std::runtime_error("Metal not initialized — call is_available() first");
    }

    auto pso = get_pipeline(kernel_name);

    @autoreleasepool {
        id<MTLCommandQueue> queue = [g_device newCommandQueue];
        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

        // Buffers are raw id<MTLBuffer> pointers passed as uint64_t
        id<MTLBuffer> buf0 = (__bridge id<MTLBuffer>)(void*)buf0_ptr;
        id<MTLBuffer> buf1 = (__bridge id<MTLBuffer>)(void*)buf1_ptr;
        id<MTLBuffer> buf2 = (__bridge id<MTLBuffer>)(void*)buf2_ptr;

        [encoder setComputePipelineState:pso];
        [encoder setBuffer:buf0 offset:0 atIndex:0];
        [encoder setBuffer:buf1 offset:0 atIndex:1];
        [encoder setBuffer:buf2 offset:0 atIndex:2];
        [encoder setBytes:&batch_size length:sizeof(int) atIndex:3];
        [encoder setBytes:&heads length:sizeof(int) atIndex:4];
        [encoder setBytes:&length length:sizeof(int) atIndex:5];
        [encoder setBytes:&dim length:sizeof(int) atIndex:6];
        [encoder setBytes:&kernel_size length:sizeof(int) atIndex:7];
        [encoder setBytes:&dilation length:sizeof(int) atIndex:8];

        MTLSize grid = MTLSizeMake(length, heads, batch_size);
        NSUInteger maxTpg = pso.maxTotalThreadsPerThreadgroup;
        MTLSize tgSize = MTLSizeMake(
            std::min((NSUInteger)length, maxTpg), 1, 1);

        [encoder dispatchThreads:grid threadsPerThreadgroup:tgSize];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
    }
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

NB_MODULE(_nanobind_ext, m) {
    m.doc() = "natten-mps nanobind Metal extension (Tier 2)";

    m.def("is_available", []() { return ensure_metal(); },
          "Check if the nanobind Metal backend is available");

    m.def("dispatch_1d", &dispatch_1d_kernel,
          "Dispatch a 1D Metal kernel by name.\n"
          "buf0/buf1/buf2 are raw MTLBuffer pointers (tensor.data_ptr()).\n"
          "The caller is responsible for creating output tensors and syncing.");
}
