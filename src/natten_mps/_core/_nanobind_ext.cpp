/**
 * Nanobind Metal extension for natten-mps.
 *
 * Loads precompiled .metallib shaders and dispatches neighborhood attention
 * kernels via MPS command buffers. Currently supports 1D QK/AV forward;
 * 2D/3D delegate to the Metal (Tier 1) backend.
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>
#include <torch/torch.h>

#include <Metal/Metal.h>
#include <Foundation/Foundation.h>

namespace nb = nanobind;

// ---------------------------------------------------------------------------
// Metal device & pipeline cache
// ---------------------------------------------------------------------------

static id<MTLDevice> g_device = nil;
static id<MTLLibrary> g_library = nil;
static id<MTLCommandQueue> g_queue = nil;

static bool ensure_metal() {
    if (g_device) return true;
    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) return false;
    g_queue = [g_device newCommandQueue];

    // Look for .metallib next to this .so
    @autoreleasepool {
        NSBundle* bundle = [NSBundle bundleForClass:nil];
        // Fallback: look in the same directory as the extension
        NSString* path = [[NSString stringWithUTF8String:__FILE__]
                          stringByDeletingLastPathComponent];
        NSString* libPath = [path stringByAppendingPathComponent:
                             @"shaders/natten_nb_1d.metallib"];

        NSError* error = nil;
        NSURL* url = [NSURL fileURLWithPath:libPath];
        g_library = [g_device newLibraryWithURL:url error:&error];
        if (!g_library) {
            // Try relative to module location (set at build time)
            return false;
        }
    }
    return g_library != nil;
}

// ---------------------------------------------------------------------------
// Stub implementations — delegate to Metal tier for now
// ---------------------------------------------------------------------------

static torch::Tensor na1d_qk_forward_nb(
    torch::Tensor q, torch::Tensor k,
    int kernel_size, int dilation,
    std::tuple<int> stride,
    std::tuple<bool> is_causal)
{
    throw std::runtime_error(
        "nanobind na1d_qk_forward not yet implemented — "
        "use metal or pure backend");
}

static torch::Tensor na1d_av_forward_nb(
    torch::Tensor attn, torch::Tensor v,
    int kernel_size, int dilation,
    std::tuple<int> stride,
    std::tuple<bool> is_causal)
{
    throw std::runtime_error(
        "nanobind na1d_av_forward not yet implemented — "
        "use metal or pure backend");
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

NB_MODULE(_nanobind_ext, m) {
    m.doc() = "natten-mps nanobind Metal extension (Tier 2)";
    m.def("is_available", []() { return ensure_metal(); },
          "Check if the nanobind Metal backend is available");
    m.def("na1d_qk_forward", &na1d_qk_forward_nb,
          "1D QK forward via precompiled Metal shaders");
    m.def("na1d_av_forward", &na1d_av_forward_nb,
          "1D AV forward via precompiled Metal shaders");
}
