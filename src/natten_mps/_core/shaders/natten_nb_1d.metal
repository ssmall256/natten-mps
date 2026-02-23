// Precompiled 1D neighborhood attention Metal kernels for nanobind backend.
// Compiled to .metallib at build time via:
//   xcrun -sdk macosx metal -c natten_nb_1d.metal -o natten_nb_1d.air
//   xcrun -sdk macosx metallib natten_nb_1d.air -o natten_nb_1d.metallib

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

inline int get_window_start(int index, int length, int kernel_size,
                            int neighborhood_size, int dilation) {
    if (dilation <= 1) {
        int start = max(index - neighborhood_size, 0);
        if (index + neighborhood_size >= length) {
            start += (length - index - neighborhood_size - 1);
        }
        return start;
    }
    int ni = index - neighborhood_size * dilation;
    if (ni < 0) {
        return index % dilation;
    }
    if (index + neighborhood_size * dilation >= length) {
        int imodd = index % dilation;
        int a = (length / dilation) * dilation;
        int b = length - a;
        if (imodd < b) {
            return length - b + imodd - 2 * neighborhood_size * dilation;
        }
        return a + imodd - kernel_size * dilation;
    }
    return ni;
}

inline float dot_f32(device const float* a, device const float* b, int len) {
    float sum = 0.0f;
    int i = 0;
    for (; i + 3 < len; i += 4) {
        float4 va = *reinterpret_cast<device const float4*>(a + i);
        float4 vb = *reinterpret_cast<device const float4*>(b + i);
        sum += dot(va, vb);
    }
    for (; i < len; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// ---------------------------------------------------------------------------
// 1D QK forward  –  grid: (L, H, B)
// Input:  query, key  [B, H, L, D]
// Output: attn        [B, H, L, K]
// ---------------------------------------------------------------------------
kernel void natten1d_qk_forward(
    device const float* query    [[buffer(0)]],
    device const float* key      [[buffer(1)]],
    device float*       attn     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       length      [[buffer(5)]],
    constant int&       dim         [[buffer(6)]],
    constant int&       kernel_size [[buffer(7)]],
    constant int&       dilation    [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
    int i = gid.x;
    int h = gid.y;
    int b = gid.z;
    if (b >= batch_size || h >= heads || i >= length) return;

    int neighborhood_size = kernel_size / 2;
    int ni = get_window_start(i, length, kernel_size, neighborhood_size, dilation);

    int q_base = ((b * heads + h) * length + i) * dim;
    for (int ki = 0; ki < kernel_size; ki++) {
        float sum = 0.0f;
        int key_i = ki * dilation + ni;
        if (key_i >= 0 && key_i < length) {
            int k_base = ((b * heads + h) * length + key_i) * dim;
            sum = dot_f32(query + q_base, key + k_base, dim);
        }
        int attn_idx = ((b * heads + h) * length + i) * kernel_size + ki;
        attn[attn_idx] = sum;
    }
}

// ---------------------------------------------------------------------------
// 1D AV forward  –  grid: (L, H, B)
// Input:  attn  [B, H, L, K],  value [B, H, L, D]
// Output: out   [B, H, L, D]
// ---------------------------------------------------------------------------
kernel void natten1d_av_forward(
    device const float* attn    [[buffer(0)]],
    device const float* value   [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       length      [[buffer(5)]],
    constant int&       dim         [[buffer(6)]],
    constant int&       kernel_size [[buffer(7)]],
    constant int&       dilation    [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
    int i = gid.x;
    int h = gid.y;
    int b = gid.z;
    if (b >= batch_size || h >= heads || i >= length) return;

    int neighborhood_size = kernel_size / 2;
    int ni = get_window_start(i, length, kernel_size, neighborhood_size, dilation);

    for (int d = 0; d < dim; d++) {
        float sum = 0.0f;
        for (int ki = 0; ki < kernel_size; ki++) {
            int value_i = ki * dilation + ni;
            if (value_i >= 0 && value_i < length) {
                int attn_idx = ((b * heads + h) * length + i) * kernel_size + ki;
                int val_idx  = ((b * heads + h) * length + value_i) * dim + d;
                sum += attn[attn_idx] * value[val_idx];
            }
        }
        int out_idx = ((b * heads + h) * length + i) * dim + d;
        out[out_idx] = sum;
    }
}
