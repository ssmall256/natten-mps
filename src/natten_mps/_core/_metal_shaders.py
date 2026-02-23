"""
Metal Shading Language source for neighborhood attention kernels.

Kernels operate on heads-first layout: [B, H, L, D] (1D), [B, H, Hi, Wi, D] (2D),
or [B, H, Dp, Hi, Wi, D] (3D). The Python dispatch layer handles permutation
from/to the spatial-first layout used by the rest of natten-mps.

Adapted from natten_metal_gem kernels with proven numerical parity against
NATTEN's CUDA reference implementation.
"""

NATTEN_METAL_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

inline int get_causal_window_start(int index, int kernel_size, int dilation) {
    // Do NOT clamp to 0 — we need negative values to match the pure backend's
    // ordering where ki=0 is the farthest-back key position. The kernel loops
    // check validity (>= 0, <= index) before accessing memory.
    return index - (kernel_size - 1) * dilation;
}

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

// ---------------------------------------------------------------------------
// SIMD vec4 helpers — use float4/half4 for inner-loop vectorization
// ---------------------------------------------------------------------------

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

inline float dot_f16(device const half* a, device const half* b, int len) {
    float sum = 0.0f;
    int i = 0;
    for (; i + 3 < len; i += 4) {
        half4 va = *reinterpret_cast<device const half4*>(a + i);
        half4 vb = *reinterpret_cast<device const half4*>(b + i);
        sum += dot(float4(va), float4(vb));
    }
    for (; i < len; i++) {
        sum += float(a[i]) * float(b[i]);
    }
    return sum;
}

inline void weighted_add_f32(device const float* src, float weight,
                              thread float* dst, int len) {
    int i = 0;
    for (; i + 3 < len; i += 4) {
        float4 vs = *reinterpret_cast<device const float4*>(src + i);
        dst[i]   += weight * vs.x;
        dst[i+1] += weight * vs.y;
        dst[i+2] += weight * vs.z;
        dst[i+3] += weight * vs.w;
    }
    for (; i < len; i++) {
        dst[i] += weight * src[i];
    }
}

inline void weighted_add_f16(device const half* src, float weight,
                              thread float* dst, int len) {
    int i = 0;
    for (; i + 3 < len; i += 4) {
        half4 vs = *reinterpret_cast<device const half4*>(src + i);
        float4 fs = float4(vs);
        dst[i]   += weight * fs.x;
        dst[i+1] += weight * fs.y;
        dst[i+2] += weight * fs.z;
        dst[i+3] += weight * fs.w;
    }
    for (; i < len; i++) {
        dst[i] += weight * float(src[i]);
    }
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

// ---------------------------------------------------------------------------
// 2D QK forward  –  grid: (W, Hi, B*H)
// Input:  query, key  [B, H, Hi, Wi, D]
// Output: attn        [B, H, Hi, Wi, K*K]
// ---------------------------------------------------------------------------
kernel void natten2d_qk_forward(
    device const float* query    [[buffer(0)]],
    device const float* key      [[buffer(1)]],
    device float*       attn     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       height      [[buffer(5)]],
    constant int&       width       [[buffer(6)]],
    constant int&       dim         [[buffer(7)]],
    constant int&       kernel_size_h [[buffer(8)]],
    constant int&       kernel_size_w [[buffer(9)]],
    constant int&       dilation_h    [[buffer(10)]],
    constant int&       dilation_w    [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]])
{
    int j = gid.x;
    int i = gid.y;
    int bh = gid.z;
    int b = bh / heads;
    int h = bh % heads;
    if (b >= batch_size || i >= height || j >= width) return;

    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int ni = get_window_start(i, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = get_window_start(j, width, kernel_size_w, nh_size_w, dilation_w);

    int q_base = (((b * heads + h) * height + i) * width + j) * dim;
    for (int ki = 0; ki < kernel_size_h; ki++) {
        for (int kj = 0; kj < kernel_size_w; kj++) {
            float sum = 0.0f;
            int key_i = ki * dilation_h + ni;
            int key_j = kj * dilation_w + nj;
            if (key_i >= 0 && key_i < height && key_j >= 0 && key_j < width) {
                int k_base = (((b * heads + h) * height + key_i) * width + key_j) * dim;
                sum = dot_f32(query + q_base, key + k_base, dim);
            }
            int attn_idx = (((b * heads + h) * height + i) * width + j) * (kernel_size_h * kernel_size_w) + ki * kernel_size_w + kj;
            attn[attn_idx] = sum;
        }
    }
}

// ---------------------------------------------------------------------------
// 2D AV forward  –  grid: (W, Hi, B*H)
// Input:  attn  [B, H, Hi, Wi, K*K],  value [B, H, Hi, Wi, D]
// Output: out   [B, H, Hi, Wi, D]
// ---------------------------------------------------------------------------
kernel void natten2d_av_forward(
    device const float* attn    [[buffer(0)]],
    device const float* value   [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       height      [[buffer(5)]],
    constant int&       width       [[buffer(6)]],
    constant int&       dim         [[buffer(7)]],
    constant int&       kernel_size_h [[buffer(8)]],
    constant int&       kernel_size_w [[buffer(9)]],
    constant int&       dilation_h    [[buffer(10)]],
    constant int&       dilation_w    [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]])
{
    int j = gid.x;
    int i = gid.y;
    int bh = gid.z;
    int b = bh / heads;
    int h = bh % heads;
    if (b >= batch_size || i >= height || j >= width) return;

    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int ni = get_window_start(i, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = get_window_start(j, width, kernel_size_w, nh_size_w, dilation_w);

    for (int d = 0; d < dim; d++) {
        float sum = 0.0f;
        for (int ki = 0; ki < kernel_size_h; ki++) {
            for (int kj = 0; kj < kernel_size_w; kj++) {
                int value_i = ki * dilation_h + ni;
                int value_j = kj * dilation_w + nj;
                if (value_i >= 0 && value_i < height && value_j >= 0 && value_j < width) {
                    int attn_idx = (((b * heads + h) * height + i) * width + j) * (kernel_size_h * kernel_size_w) + ki * kernel_size_w + kj;
                    int val_idx  = (((b * heads + h) * height + value_i) * width + value_j) * dim + d;
                    sum += attn[attn_idx] * value[val_idx];
                }
            }
        }
        int out_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
        out[out_idx] = sum;
    }
}

// ---------------------------------------------------------------------------
// 3D QK forward  –  grid: (W, H, B*heads*Dp)
// Input:  query, key  [B, heads, Dp, Hi, Wi, D]
// Output: attn        [B, heads, Dp, Hi, Wi, K*K*K]
// ---------------------------------------------------------------------------
kernel void natten3d_qk_forward(
    device const float* query    [[buffer(0)]],
    device const float* key      [[buffer(1)]],
    device float*       attn     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       depth       [[buffer(5)]],
    constant int&       height      [[buffer(6)]],
    constant int&       width       [[buffer(7)]],
    constant int&       dim         [[buffer(8)]],
    constant int&       kernel_size_d [[buffer(9)]],
    constant int&       kernel_size_h [[buffer(10)]],
    constant int&       kernel_size_w [[buffer(11)]],
    constant int&       dilation_d    [[buffer(12)]],
    constant int&       dilation_h    [[buffer(13)]],
    constant int&       dilation_w    [[buffer(14)]],
    uint3 gid [[thread_position_in_grid]])
{
    int j = gid.x;   // width
    int i = gid.y;   // height
    int bhd = gid.z; // batch * heads * depth
    int b  = bhd / (heads * depth);
    int rem = bhd % (heads * depth);
    int h  = rem / depth;
    int dp = rem % depth;
    if (b >= batch_size || i >= height || j >= width || dp >= depth) return;

    int nh_size_d = kernel_size_d / 2;
    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int nd = get_window_start(dp, depth, kernel_size_d, nh_size_d, dilation_d);
    int ni = get_window_start(i, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = get_window_start(j, width, kernel_size_w, nh_size_w, dilation_w);

    int k_vol = kernel_size_d * kernel_size_h * kernel_size_w;

    int q_base = ((((b*heads+h)*depth+dp)*height+i)*width+j)*dim;
    for (int kd = 0; kd < kernel_size_d; kd++) {
        int key_d = kd * dilation_d + nd;
        for (int ki = 0; ki < kernel_size_h; ki++) {
            int key_i = ki * dilation_h + ni;
            for (int kj = 0; kj < kernel_size_w; kj++) {
                int key_j = kj * dilation_w + nj;
                float sum = 0.0f;
                if (key_d >= 0 && key_d < depth &&
                    key_i >= 0 && key_i < height &&
                    key_j >= 0 && key_j < width) {
                    int k_base = ((((b*heads+h)*depth+key_d)*height+key_i)*width+key_j)*dim;
                    sum = dot_f32(query + q_base, key + k_base, dim);
                }
                int attn_idx = ((((b*heads+h)*depth+dp)*height+i)*width+j)*k_vol
                               + (kd*kernel_size_h+ki)*kernel_size_w+kj;
                attn[attn_idx] = sum;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3D AV forward  –  grid: (W, H, B*heads*Dp)
// Input:  attn  [B, heads, Dp, Hi, Wi, K*K*K],  value [B, heads, Dp, Hi, Wi, D]
// Output: out   [B, heads, Dp, Hi, Wi, D]
// ---------------------------------------------------------------------------
kernel void natten3d_av_forward(
    device const float* attn    [[buffer(0)]],
    device const float* value   [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       depth       [[buffer(5)]],
    constant int&       height      [[buffer(6)]],
    constant int&       width       [[buffer(7)]],
    constant int&       dim         [[buffer(8)]],
    constant int&       kernel_size_d [[buffer(9)]],
    constant int&       kernel_size_h [[buffer(10)]],
    constant int&       kernel_size_w [[buffer(11)]],
    constant int&       dilation_d    [[buffer(12)]],
    constant int&       dilation_h    [[buffer(13)]],
    constant int&       dilation_w    [[buffer(14)]],
    uint3 gid [[thread_position_in_grid]])
{
    int j = gid.x;   // width
    int i = gid.y;   // height
    int bhd = gid.z; // batch * heads * depth
    int b  = bhd / (heads * depth);
    int rem = bhd % (heads * depth);
    int h  = rem / depth;
    int dp = rem % depth;
    if (b >= batch_size || i >= height || j >= width || dp >= depth) return;

    int nh_size_d = kernel_size_d / 2;
    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int nd = get_window_start(dp, depth, kernel_size_d, nh_size_d, dilation_d);
    int ni = get_window_start(i, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = get_window_start(j, width, kernel_size_w, nh_size_w, dilation_w);

    int k_vol = kernel_size_d * kernel_size_h * kernel_size_w;

    for (int d = 0; d < dim; d++) {
        float sum = 0.0f;
        for (int kd = 0; kd < kernel_size_d; kd++) {
            int val_d = kd * dilation_d + nd;
            for (int ki = 0; ki < kernel_size_h; ki++) {
                int val_i = ki * dilation_h + ni;
                for (int kj = 0; kj < kernel_size_w; kj++) {
                    int val_j = kj * dilation_w + nj;
                    if (val_d >= 0 && val_d < depth &&
                        val_i >= 0 && val_i < height &&
                        val_j >= 0 && val_j < width) {
                        int attn_idx = ((((b*heads+h)*depth+dp)*height+i)*width+j)*k_vol
                                       + (kd*kernel_size_h+ki)*kernel_size_w+kj;
                        int val_idx  = ((((b*heads+h)*depth+val_d)*height+val_i)*width+val_j)*dim+d;
                        sum += attn[attn_idx] * value[val_idx];
                    }
                }
            }
        }
        int out_idx = ((((b*heads+h)*depth+dp)*height+i)*width+j)*dim+d;
        out[out_idx] = sum;
    }
}

// ===========================================================================
// Float16 (half) kernel variants – float accumulators, half storage
// ===========================================================================

// ---------------------------------------------------------------------------
// 1D QK forward (f16)
// ---------------------------------------------------------------------------
kernel void natten1d_qk_forward_f16(
    device const half*  query    [[buffer(0)]],
    device const half*  key      [[buffer(1)]],
    device half*        attn     [[buffer(2)]],
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
            sum = dot_f16(query + q_base, key + k_base, dim);
        }
        int attn_idx = ((b * heads + h) * length + i) * kernel_size + ki;
        attn[attn_idx] = half(sum);
    }
}

// ---------------------------------------------------------------------------
// 1D AV forward (f16)
// ---------------------------------------------------------------------------
kernel void natten1d_av_forward_f16(
    device const half*  attn    [[buffer(0)]],
    device const half*  value   [[buffer(1)]],
    device half*        out     [[buffer(2)]],
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
                sum += float(attn[attn_idx]) * float(value[val_idx]);
            }
        }
        int out_idx = ((b * heads + h) * length + i) * dim + d;
        out[out_idx] = half(sum);
    }
}

// ---------------------------------------------------------------------------
// 2D QK forward (f16)
// ---------------------------------------------------------------------------
kernel void natten2d_qk_forward_f16(
    device const half*  query    [[buffer(0)]],
    device const half*  key      [[buffer(1)]],
    device half*        attn     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       height      [[buffer(5)]],
    constant int&       width       [[buffer(6)]],
    constant int&       dim         [[buffer(7)]],
    constant int&       kernel_size_h [[buffer(8)]],
    constant int&       kernel_size_w [[buffer(9)]],
    constant int&       dilation_h    [[buffer(10)]],
    constant int&       dilation_w    [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]])
{
    int j = gid.x;
    int i = gid.y;
    int bh = gid.z;
    int b = bh / heads;
    int h = bh % heads;
    if (b >= batch_size || i >= height || j >= width) return;

    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int ni = get_window_start(i, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = get_window_start(j, width, kernel_size_w, nh_size_w, dilation_w);

    int q_base = (((b * heads + h) * height + i) * width + j) * dim;
    for (int ki = 0; ki < kernel_size_h; ki++) {
        for (int kj = 0; kj < kernel_size_w; kj++) {
            float sum = 0.0f;
            int key_i = ki * dilation_h + ni;
            int key_j = kj * dilation_w + nj;
            if (key_i >= 0 && key_i < height && key_j >= 0 && key_j < width) {
                int k_base = (((b * heads + h) * height + key_i) * width + key_j) * dim;
                sum = dot_f16(query + q_base, key + k_base, dim);
            }
            int attn_idx = (((b * heads + h) * height + i) * width + j) * (kernel_size_h * kernel_size_w) + ki * kernel_size_w + kj;
            attn[attn_idx] = half(sum);
        }
    }
}

// ---------------------------------------------------------------------------
// 2D AV forward (f16)
// ---------------------------------------------------------------------------
kernel void natten2d_av_forward_f16(
    device const half*  attn    [[buffer(0)]],
    device const half*  value   [[buffer(1)]],
    device half*        out     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       height      [[buffer(5)]],
    constant int&       width       [[buffer(6)]],
    constant int&       dim         [[buffer(7)]],
    constant int&       kernel_size_h [[buffer(8)]],
    constant int&       kernel_size_w [[buffer(9)]],
    constant int&       dilation_h    [[buffer(10)]],
    constant int&       dilation_w    [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]])
{
    int j = gid.x;
    int i = gid.y;
    int bh = gid.z;
    int b = bh / heads;
    int h = bh % heads;
    if (b >= batch_size || i >= height || j >= width) return;

    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int ni = get_window_start(i, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = get_window_start(j, width, kernel_size_w, nh_size_w, dilation_w);

    for (int d = 0; d < dim; d++) {
        float sum = 0.0f;
        for (int ki = 0; ki < kernel_size_h; ki++) {
            for (int kj = 0; kj < kernel_size_w; kj++) {
                int value_i = ki * dilation_h + ni;
                int value_j = kj * dilation_w + nj;
                if (value_i >= 0 && value_i < height && value_j >= 0 && value_j < width) {
                    int attn_idx = (((b * heads + h) * height + i) * width + j) * (kernel_size_h * kernel_size_w) + ki * kernel_size_w + kj;
                    int val_idx  = (((b * heads + h) * height + value_i) * width + value_j) * dim + d;
                    sum += float(attn[attn_idx]) * float(value[val_idx]);
                }
            }
        }
        int out_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
        out[out_idx] = half(sum);
    }
}

// ---------------------------------------------------------------------------
// 3D QK forward (f16)
// ---------------------------------------------------------------------------
kernel void natten3d_qk_forward_f16(
    device const half*  query    [[buffer(0)]],
    device const half*  key      [[buffer(1)]],
    device half*        attn     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       depth       [[buffer(5)]],
    constant int&       height      [[buffer(6)]],
    constant int&       width       [[buffer(7)]],
    constant int&       dim         [[buffer(8)]],
    constant int&       kernel_size_d [[buffer(9)]],
    constant int&       kernel_size_h [[buffer(10)]],
    constant int&       kernel_size_w [[buffer(11)]],
    constant int&       dilation_d    [[buffer(12)]],
    constant int&       dilation_h    [[buffer(13)]],
    constant int&       dilation_w    [[buffer(14)]],
    uint3 gid [[thread_position_in_grid]])
{
    int j = gid.x;
    int i = gid.y;
    int bhd = gid.z;
    int b  = bhd / (heads * depth);
    int rem = bhd % (heads * depth);
    int h  = rem / depth;
    int dp = rem % depth;
    if (b >= batch_size || i >= height || j >= width || dp >= depth) return;

    int nh_size_d = kernel_size_d / 2;
    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int nd = get_window_start(dp, depth, kernel_size_d, nh_size_d, dilation_d);
    int ni = get_window_start(i, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = get_window_start(j, width, kernel_size_w, nh_size_w, dilation_w);

    int k_vol = kernel_size_d * kernel_size_h * kernel_size_w;

    int q_base = ((((b*heads+h)*depth+dp)*height+i)*width+j)*dim;
    for (int kd = 0; kd < kernel_size_d; kd++) {
        int key_d = kd * dilation_d + nd;
        for (int ki = 0; ki < kernel_size_h; ki++) {
            int key_i = ki * dilation_h + ni;
            for (int kj = 0; kj < kernel_size_w; kj++) {
                int key_j = kj * dilation_w + nj;
                float sum = 0.0f;
                if (key_d >= 0 && key_d < depth &&
                    key_i >= 0 && key_i < height &&
                    key_j >= 0 && key_j < width) {
                    int k_base = ((((b*heads+h)*depth+key_d)*height+key_i)*width+key_j)*dim;
                    sum = dot_f16(query + q_base, key + k_base, dim);
                }
                int attn_idx = ((((b*heads+h)*depth+dp)*height+i)*width+j)*k_vol
                               + (kd*kernel_size_h+ki)*kernel_size_w+kj;
                attn[attn_idx] = half(sum);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3D AV forward (f16)
// ---------------------------------------------------------------------------
kernel void natten3d_av_forward_f16(
    device const half*  attn    [[buffer(0)]],
    device const half*  value   [[buffer(1)]],
    device half*        out     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       depth       [[buffer(5)]],
    constant int&       height      [[buffer(6)]],
    constant int&       width       [[buffer(7)]],
    constant int&       dim         [[buffer(8)]],
    constant int&       kernel_size_d [[buffer(9)]],
    constant int&       kernel_size_h [[buffer(10)]],
    constant int&       kernel_size_w [[buffer(11)]],
    constant int&       dilation_d    [[buffer(12)]],
    constant int&       dilation_h    [[buffer(13)]],
    constant int&       dilation_w    [[buffer(14)]],
    uint3 gid [[thread_position_in_grid]])
{
    int j = gid.x;
    int i = gid.y;
    int bhd = gid.z;
    int b  = bhd / (heads * depth);
    int rem = bhd % (heads * depth);
    int h  = rem / depth;
    int dp = rem % depth;
    if (b >= batch_size || i >= height || j >= width || dp >= depth) return;

    int nh_size_d = kernel_size_d / 2;
    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int nd = get_window_start(dp, depth, kernel_size_d, nh_size_d, dilation_d);
    int ni = get_window_start(i, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = get_window_start(j, width, kernel_size_w, nh_size_w, dilation_w);

    int k_vol = kernel_size_d * kernel_size_h * kernel_size_w;

    for (int d = 0; d < dim; d++) {
        float sum = 0.0f;
        for (int kd = 0; kd < kernel_size_d; kd++) {
            int val_d = kd * dilation_d + nd;
            for (int ki = 0; ki < kernel_size_h; ki++) {
                int val_i = ki * dilation_h + ni;
                for (int kj = 0; kj < kernel_size_w; kj++) {
                    int val_j = kj * dilation_w + nj;
                    if (val_d >= 0 && val_d < depth &&
                        val_i >= 0 && val_i < height &&
                        val_j >= 0 && val_j < width) {
                        int attn_idx = ((((b*heads+h)*depth+dp)*height+i)*width+j)*k_vol
                                       + (kd*kernel_size_h+ki)*kernel_size_w+kj;
                        int val_idx  = ((((b*heads+h)*depth+val_d)*height+val_i)*width+val_j)*dim+d;
                        sum += float(attn[attn_idx]) * float(value[val_idx]);
                    }
                }
            }
        }
        int out_idx = ((((b*heads+h)*depth+dp)*height+i)*width+j)*dim+d;
        out[out_idx] = half(sum);
    }
}

// ===========================================================================
// Causal kernel variants – per-axis causal masking
// ===========================================================================

// ---------------------------------------------------------------------------
// 1D QK causal forward (float32)  –  grid: (L, H, B)
// ---------------------------------------------------------------------------
kernel void natten1d_qk_causal_forward(
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

    int ni = get_causal_window_start(i, kernel_size, dilation);

    int q_base = ((b * heads + h) * length + i) * dim;
    for (int ki = 0; ki < kernel_size; ki++) {
        int key_i = ki * dilation + ni;
        float sum = 0.0f;
        bool valid = (key_i >= 0 && key_i < length && key_i <= i);
        if (valid) {
            int k_base = ((b * heads + h) * length + key_i) * dim;
            sum = dot_f32(query + q_base, key + k_base, dim);
        }
        int attn_idx = ((b * heads + h) * length + i) * kernel_size + ki;
        attn[attn_idx] = valid ? sum : -INFINITY;
    }
}

// ---------------------------------------------------------------------------
// 1D AV causal forward (float32)  –  grid: (L, H, B)
// ---------------------------------------------------------------------------
kernel void natten1d_av_causal_forward(
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

    int ni = get_causal_window_start(i, kernel_size, dilation);

    for (int d = 0; d < dim; d++) {
        float sum = 0.0f;
        for (int ki = 0; ki < kernel_size; ki++) {
            int value_i = ki * dilation + ni;
            if (value_i >= 0 && value_i < length && value_i <= i) {
                int attn_idx = ((b * heads + h) * length + i) * kernel_size + ki;
                int val_idx  = ((b * heads + h) * length + value_i) * dim + d;
                sum += attn[attn_idx] * value[val_idx];
            }
        }
        int out_idx = ((b * heads + h) * length + i) * dim + d;
        out[out_idx] = sum;
    }
}

// ---------------------------------------------------------------------------
// 2D QK causal forward (float32)  –  grid: (W, Hi, B*H)
// Per-axis causal flags: causal_h, causal_w (0 or 1)
// ---------------------------------------------------------------------------
kernel void natten2d_qk_causal_forward(
    device const float* query    [[buffer(0)]],
    device const float* key      [[buffer(1)]],
    device float*       attn     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       height      [[buffer(5)]],
    constant int&       width       [[buffer(6)]],
    constant int&       dim         [[buffer(7)]],
    constant int&       kernel_size_h [[buffer(8)]],
    constant int&       kernel_size_w [[buffer(9)]],
    constant int&       dilation_h    [[buffer(10)]],
    constant int&       dilation_w    [[buffer(11)]],
    constant int&       causal_h    [[buffer(12)]],
    constant int&       causal_w    [[buffer(13)]],
    uint3 gid [[thread_position_in_grid]])
{
    int j = gid.x;
    int i = gid.y;
    int bh = gid.z;
    int b = bh / heads;
    int h = bh % heads;
    if (b >= batch_size || i >= height || j >= width) return;

    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int ni = causal_h ? get_causal_window_start(i, kernel_size_h, dilation_h)
                      : get_window_start(i, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = causal_w ? get_causal_window_start(j, kernel_size_w, dilation_w)
                      : get_window_start(j, width, kernel_size_w, nh_size_w, dilation_w);

    int q_base = (((b * heads + h) * height + i) * width + j) * dim;
    for (int ki = 0; ki < kernel_size_h; ki++) {
        int key_i = ki * dilation_h + ni;
        bool valid_i = (key_i >= 0 && key_i < height);
        if (causal_h) valid_i = valid_i && (key_i <= i);
        for (int kj = 0; kj < kernel_size_w; kj++) {
            int key_j = kj * dilation_w + nj;
            bool valid_j = (key_j >= 0 && key_j < width);
            if (causal_w) valid_j = valid_j && (key_j <= j);
            bool valid = valid_i && valid_j;
            float sum = 0.0f;
            if (valid) {
                int k_base = (((b * heads + h) * height + key_i) * width + key_j) * dim;
                sum = dot_f32(query + q_base, key + k_base, dim);
            }
            int attn_idx = (((b * heads + h) * height + i) * width + j) * (kernel_size_h * kernel_size_w) + ki * kernel_size_w + kj;
            attn[attn_idx] = valid ? sum : -INFINITY;
        }
    }
}

// ---------------------------------------------------------------------------
// 2D AV causal forward (float32)  –  grid: (W, Hi, B*H)
// ---------------------------------------------------------------------------
kernel void natten2d_av_causal_forward(
    device const float* attn    [[buffer(0)]],
    device const float* value   [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       height      [[buffer(5)]],
    constant int&       width       [[buffer(6)]],
    constant int&       dim         [[buffer(7)]],
    constant int&       kernel_size_h [[buffer(8)]],
    constant int&       kernel_size_w [[buffer(9)]],
    constant int&       dilation_h    [[buffer(10)]],
    constant int&       dilation_w    [[buffer(11)]],
    constant int&       causal_h    [[buffer(12)]],
    constant int&       causal_w    [[buffer(13)]],
    uint3 gid [[thread_position_in_grid]])
{
    int j = gid.x;
    int i = gid.y;
    int bh = gid.z;
    int b = bh / heads;
    int h = bh % heads;
    if (b >= batch_size || i >= height || j >= width) return;

    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int ni = causal_h ? get_causal_window_start(i, kernel_size_h, dilation_h)
                      : get_window_start(i, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = causal_w ? get_causal_window_start(j, kernel_size_w, dilation_w)
                      : get_window_start(j, width, kernel_size_w, nh_size_w, dilation_w);

    for (int d = 0; d < dim; d++) {
        float sum = 0.0f;
        for (int ki = 0; ki < kernel_size_h; ki++) {
            int value_i = ki * dilation_h + ni;
            bool valid_i = (value_i >= 0 && value_i < height);
            if (causal_h) valid_i = valid_i && (value_i <= i);
            for (int kj = 0; kj < kernel_size_w; kj++) {
                int value_j = kj * dilation_w + nj;
                bool valid_j = (value_j >= 0 && value_j < width);
                if (causal_w) valid_j = valid_j && (value_j <= j);
                if (valid_i && valid_j) {
                    int attn_idx = (((b * heads + h) * height + i) * width + j) * (kernel_size_h * kernel_size_w) + ki * kernel_size_w + kj;
                    int val_idx  = (((b * heads + h) * height + value_i) * width + value_j) * dim + d;
                    sum += attn[attn_idx] * value[val_idx];
                }
            }
        }
        int out_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
        out[out_idx] = sum;
    }
}

// ---------------------------------------------------------------------------
// 3D QK causal forward (float32)  –  grid: (W, H, B*heads*Dp)
// Per-axis causal flags: causal_d, causal_h, causal_w
// ---------------------------------------------------------------------------
kernel void natten3d_qk_causal_forward(
    device const float* query    [[buffer(0)]],
    device const float* key      [[buffer(1)]],
    device float*       attn     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       depth       [[buffer(5)]],
    constant int&       height      [[buffer(6)]],
    constant int&       width       [[buffer(7)]],
    constant int&       dim         [[buffer(8)]],
    constant int&       kernel_size_d [[buffer(9)]],
    constant int&       kernel_size_h [[buffer(10)]],
    constant int&       kernel_size_w [[buffer(11)]],
    constant int&       dilation_d    [[buffer(12)]],
    constant int&       dilation_h    [[buffer(13)]],
    constant int&       dilation_w    [[buffer(14)]],
    constant int&       causal_d    [[buffer(15)]],
    constant int&       causal_h    [[buffer(16)]],
    constant int&       causal_w    [[buffer(17)]],
    uint3 gid [[thread_position_in_grid]])
{
    int j = gid.x;
    int i = gid.y;
    int bhd = gid.z;
    int b  = bhd / (heads * depth);
    int rem = bhd % (heads * depth);
    int h  = rem / depth;
    int dp = rem % depth;
    if (b >= batch_size || i >= height || j >= width || dp >= depth) return;

    int nh_size_d = kernel_size_d / 2;
    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int nd = causal_d ? get_causal_window_start(dp, kernel_size_d, dilation_d)
                      : get_window_start(dp, depth, kernel_size_d, nh_size_d, dilation_d);
    int ni = causal_h ? get_causal_window_start(i, kernel_size_h, dilation_h)
                      : get_window_start(i, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = causal_w ? get_causal_window_start(j, kernel_size_w, dilation_w)
                      : get_window_start(j, width, kernel_size_w, nh_size_w, dilation_w);

    int k_vol = kernel_size_d * kernel_size_h * kernel_size_w;

    int q_base = ((((b*heads+h)*depth+dp)*height+i)*width+j)*dim;
    for (int kd = 0; kd < kernel_size_d; kd++) {
        int key_d = kd * dilation_d + nd;
        bool valid_d = (key_d >= 0 && key_d < depth);
        if (causal_d) valid_d = valid_d && (key_d <= dp);
        for (int ki = 0; ki < kernel_size_h; ki++) {
            int key_i = ki * dilation_h + ni;
            bool valid_i = (key_i >= 0 && key_i < height);
            if (causal_h) valid_i = valid_i && (key_i <= i);
            for (int kj = 0; kj < kernel_size_w; kj++) {
                int key_j = kj * dilation_w + nj;
                bool valid_j = (key_j >= 0 && key_j < width);
                if (causal_w) valid_j = valid_j && (key_j <= j);
                bool valid = valid_d && valid_i && valid_j;
                float sum = 0.0f;
                if (valid) {
                    int k_base = ((((b*heads+h)*depth+key_d)*height+key_i)*width+key_j)*dim;
                    sum = dot_f32(query + q_base, key + k_base, dim);
                }
                int attn_idx = ((((b*heads+h)*depth+dp)*height+i)*width+j)*k_vol
                               + (kd*kernel_size_h+ki)*kernel_size_w+kj;
                attn[attn_idx] = valid ? sum : -INFINITY;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3D AV causal forward (float32)  –  grid: (W, H, B*heads*Dp)
// ---------------------------------------------------------------------------
kernel void natten3d_av_causal_forward(
    device const float* attn    [[buffer(0)]],
    device const float* value   [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       depth       [[buffer(5)]],
    constant int&       height      [[buffer(6)]],
    constant int&       width       [[buffer(7)]],
    constant int&       dim         [[buffer(8)]],
    constant int&       kernel_size_d [[buffer(9)]],
    constant int&       kernel_size_h [[buffer(10)]],
    constant int&       kernel_size_w [[buffer(11)]],
    constant int&       dilation_d    [[buffer(12)]],
    constant int&       dilation_h    [[buffer(13)]],
    constant int&       dilation_w    [[buffer(14)]],
    constant int&       causal_d    [[buffer(15)]],
    constant int&       causal_h    [[buffer(16)]],
    constant int&       causal_w    [[buffer(17)]],
    uint3 gid [[thread_position_in_grid]])
{
    int j = gid.x;
    int i = gid.y;
    int bhd = gid.z;
    int b  = bhd / (heads * depth);
    int rem = bhd % (heads * depth);
    int h  = rem / depth;
    int dp = rem % depth;
    if (b >= batch_size || i >= height || j >= width || dp >= depth) return;

    int nh_size_d = kernel_size_d / 2;
    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int nd = causal_d ? get_causal_window_start(dp, kernel_size_d, dilation_d)
                      : get_window_start(dp, depth, kernel_size_d, nh_size_d, dilation_d);
    int ni = causal_h ? get_causal_window_start(i, kernel_size_h, dilation_h)
                      : get_window_start(i, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = causal_w ? get_causal_window_start(j, kernel_size_w, dilation_w)
                      : get_window_start(j, width, kernel_size_w, nh_size_w, dilation_w);

    int k_vol = kernel_size_d * kernel_size_h * kernel_size_w;

    for (int d = 0; d < dim; d++) {
        float sum = 0.0f;
        for (int kd = 0; kd < kernel_size_d; kd++) {
            int val_d = kd * dilation_d + nd;
            bool valid_d = (val_d >= 0 && val_d < depth);
            if (causal_d) valid_d = valid_d && (val_d <= dp);
            for (int ki = 0; ki < kernel_size_h; ki++) {
                int val_i = ki * dilation_h + ni;
                bool valid_i = (val_i >= 0 && val_i < height);
                if (causal_h) valid_i = valid_i && (val_i <= i);
                for (int kj = 0; kj < kernel_size_w; kj++) {
                    int val_j = kj * dilation_w + nj;
                    bool valid_j = (val_j >= 0 && val_j < width);
                    if (causal_w) valid_j = valid_j && (val_j <= j);
                    if (valid_d && valid_i && valid_j) {
                        int attn_idx = ((((b*heads+h)*depth+dp)*height+i)*width+j)*k_vol
                                       + (kd*kernel_size_h+ki)*kernel_size_w+kj;
                        int val_idx  = ((((b*heads+h)*depth+val_d)*height+val_i)*width+val_j)*dim+d;
                        sum += attn[attn_idx] * value[val_idx];
                    }
                }
            }
        }
        int out_idx = ((((b*heads+h)*depth+dp)*height+i)*width+j)*dim+d;
        out[out_idx] = sum;
    }
}

// ===========================================================================
// Causal kernel variants – float16
// ===========================================================================

// ---------------------------------------------------------------------------
// 1D QK causal forward (f16)
// ---------------------------------------------------------------------------
kernel void natten1d_qk_causal_forward_f16(
    device const half*  query    [[buffer(0)]],
    device const half*  key      [[buffer(1)]],
    device half*        attn     [[buffer(2)]],
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

    int ni = get_causal_window_start(i, kernel_size, dilation);

    int q_base = ((b * heads + h) * length + i) * dim;
    for (int ki = 0; ki < kernel_size; ki++) {
        int key_i = ki * dilation + ni;
        float sum = 0.0f;
        bool valid = (key_i >= 0 && key_i < length && key_i <= i);
        if (valid) {
            int k_base = ((b * heads + h) * length + key_i) * dim;
            sum = dot_f16(query + q_base, key + k_base, dim);
        }
        int attn_idx = ((b * heads + h) * length + i) * kernel_size + ki;
        attn[attn_idx] = valid ? half(sum) : half(-INFINITY);
    }
}

// ---------------------------------------------------------------------------
// 1D AV causal forward (f16)
// ---------------------------------------------------------------------------
kernel void natten1d_av_causal_forward_f16(
    device const half*  attn    [[buffer(0)]],
    device const half*  value   [[buffer(1)]],
    device half*        out     [[buffer(2)]],
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

    int ni = get_causal_window_start(i, kernel_size, dilation);

    for (int d = 0; d < dim; d++) {
        float sum = 0.0f;
        for (int ki = 0; ki < kernel_size; ki++) {
            int value_i = ki * dilation + ni;
            if (value_i >= 0 && value_i < length && value_i <= i) {
                int attn_idx = ((b * heads + h) * length + i) * kernel_size + ki;
                int val_idx  = ((b * heads + h) * length + value_i) * dim + d;
                sum += float(attn[attn_idx]) * float(value[val_idx]);
            }
        }
        int out_idx = ((b * heads + h) * length + i) * dim + d;
        out[out_idx] = half(sum);
    }
}

// ---------------------------------------------------------------------------
// 2D QK causal forward (f16)
// ---------------------------------------------------------------------------
kernel void natten2d_qk_causal_forward_f16(
    device const half*  query    [[buffer(0)]],
    device const half*  key      [[buffer(1)]],
    device half*        attn     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       height      [[buffer(5)]],
    constant int&       width       [[buffer(6)]],
    constant int&       dim         [[buffer(7)]],
    constant int&       kernel_size_h [[buffer(8)]],
    constant int&       kernel_size_w [[buffer(9)]],
    constant int&       dilation_h    [[buffer(10)]],
    constant int&       dilation_w    [[buffer(11)]],
    constant int&       causal_h    [[buffer(12)]],
    constant int&       causal_w    [[buffer(13)]],
    uint3 gid [[thread_position_in_grid]])
{
    int j = gid.x;
    int i = gid.y;
    int bh = gid.z;
    int b = bh / heads;
    int h = bh % heads;
    if (b >= batch_size || i >= height || j >= width) return;

    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int ni = causal_h ? get_causal_window_start(i, kernel_size_h, dilation_h)
                      : get_window_start(i, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = causal_w ? get_causal_window_start(j, kernel_size_w, dilation_w)
                      : get_window_start(j, width, kernel_size_w, nh_size_w, dilation_w);

    int q_base = (((b * heads + h) * height + i) * width + j) * dim;
    for (int ki = 0; ki < kernel_size_h; ki++) {
        int key_i = ki * dilation_h + ni;
        bool valid_i = (key_i >= 0 && key_i < height);
        if (causal_h) valid_i = valid_i && (key_i <= i);
        for (int kj = 0; kj < kernel_size_w; kj++) {
            int key_j = kj * dilation_w + nj;
            bool valid_j = (key_j >= 0 && key_j < width);
            if (causal_w) valid_j = valid_j && (key_j <= j);
            bool valid = valid_i && valid_j;
            float sum = 0.0f;
            if (valid) {
                int k_base = (((b * heads + h) * height + key_i) * width + key_j) * dim;
                sum = dot_f16(query + q_base, key + k_base, dim);
            }
            int attn_idx = (((b * heads + h) * height + i) * width + j) * (kernel_size_h * kernel_size_w) + ki * kernel_size_w + kj;
            attn[attn_idx] = valid ? half(sum) : half(-INFINITY);
        }
    }
}

// ---------------------------------------------------------------------------
// 2D AV causal forward (f16)
// ---------------------------------------------------------------------------
kernel void natten2d_av_causal_forward_f16(
    device const half*  attn    [[buffer(0)]],
    device const half*  value   [[buffer(1)]],
    device half*        out     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       height      [[buffer(5)]],
    constant int&       width       [[buffer(6)]],
    constant int&       dim         [[buffer(7)]],
    constant int&       kernel_size_h [[buffer(8)]],
    constant int&       kernel_size_w [[buffer(9)]],
    constant int&       dilation_h    [[buffer(10)]],
    constant int&       dilation_w    [[buffer(11)]],
    constant int&       causal_h    [[buffer(12)]],
    constant int&       causal_w    [[buffer(13)]],
    uint3 gid [[thread_position_in_grid]])
{
    int j = gid.x;
    int i = gid.y;
    int bh = gid.z;
    int b = bh / heads;
    int h = bh % heads;
    if (b >= batch_size || i >= height || j >= width) return;

    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int ni = causal_h ? get_causal_window_start(i, kernel_size_h, dilation_h)
                      : get_window_start(i, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = causal_w ? get_causal_window_start(j, kernel_size_w, dilation_w)
                      : get_window_start(j, width, kernel_size_w, nh_size_w, dilation_w);

    for (int d = 0; d < dim; d++) {
        float sum = 0.0f;
        for (int ki = 0; ki < kernel_size_h; ki++) {
            int value_i = ki * dilation_h + ni;
            bool valid_i = (value_i >= 0 && value_i < height);
            if (causal_h) valid_i = valid_i && (value_i <= i);
            for (int kj = 0; kj < kernel_size_w; kj++) {
                int value_j = kj * dilation_w + nj;
                bool valid_j = (value_j >= 0 && value_j < width);
                if (causal_w) valid_j = valid_j && (value_j <= j);
                if (valid_i && valid_j) {
                    int attn_idx = (((b * heads + h) * height + i) * width + j) * (kernel_size_h * kernel_size_w) + ki * kernel_size_w + kj;
                    int val_idx  = (((b * heads + h) * height + value_i) * width + value_j) * dim + d;
                    sum += float(attn[attn_idx]) * float(value[val_idx]);
                }
            }
        }
        int out_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
        out[out_idx] = half(sum);
    }
}

// ---------------------------------------------------------------------------
// 3D QK causal forward (f16)
// ---------------------------------------------------------------------------
kernel void natten3d_qk_causal_forward_f16(
    device const half*  query    [[buffer(0)]],
    device const half*  key      [[buffer(1)]],
    device half*        attn     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       depth       [[buffer(5)]],
    constant int&       height      [[buffer(6)]],
    constant int&       width       [[buffer(7)]],
    constant int&       dim         [[buffer(8)]],
    constant int&       kernel_size_d [[buffer(9)]],
    constant int&       kernel_size_h [[buffer(10)]],
    constant int&       kernel_size_w [[buffer(11)]],
    constant int&       dilation_d    [[buffer(12)]],
    constant int&       dilation_h    [[buffer(13)]],
    constant int&       dilation_w    [[buffer(14)]],
    constant int&       causal_d    [[buffer(15)]],
    constant int&       causal_h    [[buffer(16)]],
    constant int&       causal_w    [[buffer(17)]],
    uint3 gid [[thread_position_in_grid]])
{
    int j = gid.x;
    int i = gid.y;
    int bhd = gid.z;
    int b  = bhd / (heads * depth);
    int rem = bhd % (heads * depth);
    int h  = rem / depth;
    int dp = rem % depth;
    if (b >= batch_size || i >= height || j >= width || dp >= depth) return;

    int nh_size_d = kernel_size_d / 2;
    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int nd = causal_d ? get_causal_window_start(dp, kernel_size_d, dilation_d)
                      : get_window_start(dp, depth, kernel_size_d, nh_size_d, dilation_d);
    int ni = causal_h ? get_causal_window_start(i, kernel_size_h, dilation_h)
                      : get_window_start(i, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = causal_w ? get_causal_window_start(j, kernel_size_w, dilation_w)
                      : get_window_start(j, width, kernel_size_w, nh_size_w, dilation_w);

    int k_vol = kernel_size_d * kernel_size_h * kernel_size_w;

    int q_base = ((((b*heads+h)*depth+dp)*height+i)*width+j)*dim;
    for (int kd = 0; kd < kernel_size_d; kd++) {
        int key_d = kd * dilation_d + nd;
        bool valid_d = (key_d >= 0 && key_d < depth);
        if (causal_d) valid_d = valid_d && (key_d <= dp);
        for (int ki = 0; ki < kernel_size_h; ki++) {
            int key_i = ki * dilation_h + ni;
            bool valid_i = (key_i >= 0 && key_i < height);
            if (causal_h) valid_i = valid_i && (key_i <= i);
            for (int kj = 0; kj < kernel_size_w; kj++) {
                int key_j = kj * dilation_w + nj;
                bool valid_j = (key_j >= 0 && key_j < width);
                if (causal_w) valid_j = valid_j && (key_j <= j);
                bool valid = valid_d && valid_i && valid_j;
                float sum = 0.0f;
                if (valid) {
                    int k_base = ((((b*heads+h)*depth+key_d)*height+key_i)*width+key_j)*dim;
                    sum = dot_f16(query + q_base, key + k_base, dim);
                }
                int attn_idx = ((((b*heads+h)*depth+dp)*height+i)*width+j)*k_vol
                               + (kd*kernel_size_h+ki)*kernel_size_w+kj;
                attn[attn_idx] = valid ? half(sum) : half(-INFINITY);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3D AV causal forward (f16)
// ---------------------------------------------------------------------------
kernel void natten3d_av_causal_forward_f16(
    device const half*  attn    [[buffer(0)]],
    device const half*  value   [[buffer(1)]],
    device half*        out     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       depth       [[buffer(5)]],
    constant int&       height      [[buffer(6)]],
    constant int&       width       [[buffer(7)]],
    constant int&       dim         [[buffer(8)]],
    constant int&       kernel_size_d [[buffer(9)]],
    constant int&       kernel_size_h [[buffer(10)]],
    constant int&       kernel_size_w [[buffer(11)]],
    constant int&       dilation_d    [[buffer(12)]],
    constant int&       dilation_h    [[buffer(13)]],
    constant int&       dilation_w    [[buffer(14)]],
    constant int&       causal_d    [[buffer(15)]],
    constant int&       causal_h    [[buffer(16)]],
    constant int&       causal_w    [[buffer(17)]],
    uint3 gid [[thread_position_in_grid]])
{
    int j = gid.x;
    int i = gid.y;
    int bhd = gid.z;
    int b  = bhd / (heads * depth);
    int rem = bhd % (heads * depth);
    int h  = rem / depth;
    int dp = rem % depth;
    if (b >= batch_size || i >= height || j >= width || dp >= depth) return;

    int nh_size_d = kernel_size_d / 2;
    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int nd = causal_d ? get_causal_window_start(dp, kernel_size_d, dilation_d)
                      : get_window_start(dp, depth, kernel_size_d, nh_size_d, dilation_d);
    int ni = causal_h ? get_causal_window_start(i, kernel_size_h, dilation_h)
                      : get_window_start(i, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = causal_w ? get_causal_window_start(j, kernel_size_w, dilation_w)
                      : get_window_start(j, width, kernel_size_w, nh_size_w, dilation_w);

    int k_vol = kernel_size_d * kernel_size_h * kernel_size_w;

    for (int d = 0; d < dim; d++) {
        float sum = 0.0f;
        for (int kd = 0; kd < kernel_size_d; kd++) {
            int val_d = kd * dilation_d + nd;
            bool valid_d = (val_d >= 0 && val_d < depth);
            if (causal_d) valid_d = valid_d && (val_d <= dp);
            for (int ki = 0; ki < kernel_size_h; ki++) {
                int val_i = ki * dilation_h + ni;
                bool valid_i = (val_i >= 0 && val_i < height);
                if (causal_h) valid_i = valid_i && (val_i <= i);
                for (int kj = 0; kj < kernel_size_w; kj++) {
                    int val_j = kj * dilation_w + nj;
                    bool valid_j = (val_j >= 0 && val_j < width);
                    if (causal_w) valid_j = valid_j && (val_j <= j);
                    if (valid_d && valid_i && valid_j) {
                        int attn_idx = ((((b*heads+h)*depth+dp)*height+i)*width+j)*k_vol
                                       + (kd*kernel_size_h+ki)*kernel_size_w+kj;
                        int val_idx  = ((((b*heads+h)*depth+val_d)*height+val_i)*width+val_j)*dim+d;
                        sum += float(attn[attn_idx]) * float(value[val_idx]);
                    }
                }
            }
        }
        int out_idx = ((((b*heads+h)*depth+dp)*height+i)*width+j)*dim+d;
        out[out_idx] = half(sum);
    }
}

// ===========================================================================
// Strided kernel variants – thread grid covers output positions only
// ===========================================================================

// ---------------------------------------------------------------------------
// 1D QK strided forward (float32)  –  grid: (L_out, H, B)
// query/key [B, H, L, D],  attn [B, H, L_out, K]
// ---------------------------------------------------------------------------
kernel void natten1d_qk_strided_forward(
    device const float* query    [[buffer(0)]],
    device const float* key      [[buffer(1)]],
    device float*       attn     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       length      [[buffer(5)]],
    constant int&       dim         [[buffer(6)]],
    constant int&       kernel_size [[buffer(7)]],
    constant int&       dilation    [[buffer(8)]],
    constant int&       stride_val  [[buffer(9)]],
    constant int&       l_out       [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]])
{
    int oi = gid.x;  // output index
    int h = gid.y;
    int b = gid.z;
    if (b >= batch_size || h >= heads || oi >= l_out) return;

    int i = oi * stride_val;  // query position in input space
    int neighborhood_size = kernel_size / 2;
    int ni = get_window_start(i, length, kernel_size, neighborhood_size, dilation);

    for (int ki = 0; ki < kernel_size; ki++) {
        float sum = 0.0f;
        int key_i = ki * dilation + ni;
        if (key_i >= 0 && key_i < length) {
            for (int d = 0; d < dim; d++) {
                int q_idx = ((b * heads + h) * length + i) * dim + d;
                int k_idx = ((b * heads + h) * length + key_i) * dim + d;
                sum += query[q_idx] * key[k_idx];
            }
        }
        int attn_idx = ((b * heads + h) * l_out + oi) * kernel_size + ki;
        attn[attn_idx] = sum;
    }
}

// ---------------------------------------------------------------------------
// 1D AV strided forward (float32)  –  grid: (L_out, H, B)
// attn [B, H, L_out, K],  value [B, H, L, D],  out [B, H, L_out, D]
// ---------------------------------------------------------------------------
kernel void natten1d_av_strided_forward(
    device const float* attn    [[buffer(0)]],
    device const float* value   [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       length      [[buffer(5)]],
    constant int&       dim         [[buffer(6)]],
    constant int&       kernel_size [[buffer(7)]],
    constant int&       dilation    [[buffer(8)]],
    constant int&       stride_val  [[buffer(9)]],
    constant int&       l_out       [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]])
{
    int oi = gid.x;
    int h = gid.y;
    int b = gid.z;
    if (b >= batch_size || h >= heads || oi >= l_out) return;

    int i = oi * stride_val;
    int neighborhood_size = kernel_size / 2;
    int ni = get_window_start(i, length, kernel_size, neighborhood_size, dilation);

    for (int d = 0; d < dim; d++) {
        float sum = 0.0f;
        for (int ki = 0; ki < kernel_size; ki++) {
            int value_i = ki * dilation + ni;
            if (value_i >= 0 && value_i < length) {
                int attn_idx = ((b * heads + h) * l_out + oi) * kernel_size + ki;
                int val_idx  = ((b * heads + h) * length + value_i) * dim + d;
                sum += attn[attn_idx] * value[val_idx];
            }
        }
        int out_idx = ((b * heads + h) * l_out + oi) * dim + d;
        out[out_idx] = sum;
    }
}

// ---------------------------------------------------------------------------
// 2D QK strided forward (float32)  –  grid: (W_out, H_out, B*H)
// ---------------------------------------------------------------------------
kernel void natten2d_qk_strided_forward(
    device const float* query    [[buffer(0)]],
    device const float* key      [[buffer(1)]],
    device float*       attn     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       height      [[buffer(5)]],
    constant int&       width       [[buffer(6)]],
    constant int&       dim         [[buffer(7)]],
    constant int&       kernel_size_h [[buffer(8)]],
    constant int&       kernel_size_w [[buffer(9)]],
    constant int&       dilation_h    [[buffer(10)]],
    constant int&       dilation_w    [[buffer(11)]],
    constant int&       stride_h    [[buffer(12)]],
    constant int&       stride_w    [[buffer(13)]],
    constant int&       h_out       [[buffer(14)]],
    constant int&       w_out       [[buffer(15)]],
    uint3 gid [[thread_position_in_grid]])
{
    int oj = gid.x;
    int oi = gid.y;
    int bh = gid.z;
    int b = bh / heads;
    int h = bh % heads;
    if (b >= batch_size || oi >= h_out || oj >= w_out) return;

    int i = oi * stride_h;
    int j = oj * stride_w;
    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int ni = get_window_start(i, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = get_window_start(j, width, kernel_size_w, nh_size_w, dilation_w);

    for (int ki = 0; ki < kernel_size_h; ki++) {
        for (int kj = 0; kj < kernel_size_w; kj++) {
            float sum = 0.0f;
            int key_i = ki * dilation_h + ni;
            int key_j = kj * dilation_w + nj;
            if (key_i >= 0 && key_i < height && key_j >= 0 && key_j < width) {
                for (int d = 0; d < dim; d++) {
                    int q_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
                    int k_idx = (((b * heads + h) * height + key_i) * width + key_j) * dim + d;
                    sum += query[q_idx] * key[k_idx];
                }
            }
            int attn_idx = (((b * heads + h) * h_out + oi) * w_out + oj) * (kernel_size_h * kernel_size_w) + ki * kernel_size_w + kj;
            attn[attn_idx] = sum;
        }
    }
}

// ---------------------------------------------------------------------------
// 2D AV strided forward (float32)  –  grid: (W_out, H_out, B*H)
// ---------------------------------------------------------------------------
kernel void natten2d_av_strided_forward(
    device const float* attn    [[buffer(0)]],
    device const float* value   [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       height      [[buffer(5)]],
    constant int&       width       [[buffer(6)]],
    constant int&       dim         [[buffer(7)]],
    constant int&       kernel_size_h [[buffer(8)]],
    constant int&       kernel_size_w [[buffer(9)]],
    constant int&       dilation_h    [[buffer(10)]],
    constant int&       dilation_w    [[buffer(11)]],
    constant int&       stride_h    [[buffer(12)]],
    constant int&       stride_w    [[buffer(13)]],
    constant int&       h_out       [[buffer(14)]],
    constant int&       w_out       [[buffer(15)]],
    uint3 gid [[thread_position_in_grid]])
{
    int oj = gid.x;
    int oi = gid.y;
    int bh = gid.z;
    int b = bh / heads;
    int h = bh % heads;
    if (b >= batch_size || oi >= h_out || oj >= w_out) return;

    int i = oi * stride_h;
    int j = oj * stride_w;
    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int ni = get_window_start(i, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = get_window_start(j, width, kernel_size_w, nh_size_w, dilation_w);

    for (int d = 0; d < dim; d++) {
        float sum = 0.0f;
        for (int ki = 0; ki < kernel_size_h; ki++) {
            for (int kj = 0; kj < kernel_size_w; kj++) {
                int value_i = ki * dilation_h + ni;
                int value_j = kj * dilation_w + nj;
                if (value_i >= 0 && value_i < height && value_j >= 0 && value_j < width) {
                    int attn_idx = (((b * heads + h) * h_out + oi) * w_out + oj) * (kernel_size_h * kernel_size_w) + ki * kernel_size_w + kj;
                    int val_idx  = (((b * heads + h) * height + value_i) * width + value_j) * dim + d;
                    sum += attn[attn_idx] * value[val_idx];
                }
            }
        }
        int out_idx = (((b * heads + h) * h_out + oi) * w_out + oj) * dim + d;
        out[out_idx] = sum;
    }
}

// ---------------------------------------------------------------------------
// 3D QK strided forward (float32)  –  grid: (W_out, H_out, B*heads*D_out)
// ---------------------------------------------------------------------------
kernel void natten3d_qk_strided_forward(
    device const float* query    [[buffer(0)]],
    device const float* key      [[buffer(1)]],
    device float*       attn     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       depth       [[buffer(5)]],
    constant int&       height      [[buffer(6)]],
    constant int&       width       [[buffer(7)]],
    constant int&       dim         [[buffer(8)]],
    constant int&       kernel_size_d [[buffer(9)]],
    constant int&       kernel_size_h [[buffer(10)]],
    constant int&       kernel_size_w [[buffer(11)]],
    constant int&       dilation_d    [[buffer(12)]],
    constant int&       dilation_h    [[buffer(13)]],
    constant int&       dilation_w    [[buffer(14)]],
    constant int&       stride_d    [[buffer(15)]],
    constant int&       stride_h    [[buffer(16)]],
    constant int&       stride_w    [[buffer(17)]],
    constant int&       d_out       [[buffer(18)]],
    constant int&       h_out       [[buffer(19)]],
    constant int&       w_out       [[buffer(20)]],
    uint3 gid [[thread_position_in_grid]])
{
    int oj = gid.x;
    int oi = gid.y;
    int bhd = gid.z;
    int b  = bhd / (heads * d_out);
    int rem = bhd % (heads * d_out);
    int h  = rem / d_out;
    int od = rem % d_out;
    if (b >= batch_size || oi >= h_out || oj >= w_out || od >= d_out) return;

    int dp = od * stride_d;
    int i  = oi * stride_h;
    int j  = oj * stride_w;
    int nh_size_d = kernel_size_d / 2;
    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int nd = get_window_start(dp, depth, kernel_size_d, nh_size_d, dilation_d);
    int ni = get_window_start(i, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = get_window_start(j, width, kernel_size_w, nh_size_w, dilation_w);

    int k_vol = kernel_size_d * kernel_size_h * kernel_size_w;

    for (int kd = 0; kd < kernel_size_d; kd++) {
        int key_d = kd * dilation_d + nd;
        for (int ki = 0; ki < kernel_size_h; ki++) {
            int key_i = ki * dilation_h + ni;
            for (int kj = 0; kj < kernel_size_w; kj++) {
                int key_j = kj * dilation_w + nj;
                float sum = 0.0f;
                if (key_d >= 0 && key_d < depth &&
                    key_i >= 0 && key_i < height &&
                    key_j >= 0 && key_j < width) {
                    for (int d = 0; d < dim; d++) {
                        int q_idx = ((((b*heads+h)*depth+dp)*height+i)*width+j)*dim+d;
                        int k_idx = ((((b*heads+h)*depth+key_d)*height+key_i)*width+key_j)*dim+d;
                        sum += query[q_idx] * key[k_idx];
                    }
                }
                int attn_idx = ((((b*heads+h)*d_out+od)*h_out+oi)*w_out+oj)*k_vol
                               + (kd*kernel_size_h+ki)*kernel_size_w+kj;
                attn[attn_idx] = sum;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3D AV strided forward (float32)  –  grid: (W_out, H_out, B*heads*D_out)
// ---------------------------------------------------------------------------
kernel void natten3d_av_strided_forward(
    device const float* attn    [[buffer(0)]],
    device const float* value   [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       depth       [[buffer(5)]],
    constant int&       height      [[buffer(6)]],
    constant int&       width       [[buffer(7)]],
    constant int&       dim         [[buffer(8)]],
    constant int&       kernel_size_d [[buffer(9)]],
    constant int&       kernel_size_h [[buffer(10)]],
    constant int&       kernel_size_w [[buffer(11)]],
    constant int&       dilation_d    [[buffer(12)]],
    constant int&       dilation_h    [[buffer(13)]],
    constant int&       dilation_w    [[buffer(14)]],
    constant int&       stride_d    [[buffer(15)]],
    constant int&       stride_h    [[buffer(16)]],
    constant int&       stride_w    [[buffer(17)]],
    constant int&       d_out       [[buffer(18)]],
    constant int&       h_out       [[buffer(19)]],
    constant int&       w_out       [[buffer(20)]],
    uint3 gid [[thread_position_in_grid]])
{
    int oj = gid.x;
    int oi = gid.y;
    int bhd = gid.z;
    int b  = bhd / (heads * d_out);
    int rem = bhd % (heads * d_out);
    int h  = rem / d_out;
    int od = rem % d_out;
    if (b >= batch_size || oi >= h_out || oj >= w_out || od >= d_out) return;

    int dp = od * stride_d;
    int i  = oi * stride_h;
    int j  = oj * stride_w;
    int nh_size_d = kernel_size_d / 2;
    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int nd = get_window_start(dp, depth, kernel_size_d, nh_size_d, dilation_d);
    int ni = get_window_start(i, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = get_window_start(j, width, kernel_size_w, nh_size_w, dilation_w);

    int k_vol = kernel_size_d * kernel_size_h * kernel_size_w;

    for (int d = 0; d < dim; d++) {
        float sum = 0.0f;
        for (int kd = 0; kd < kernel_size_d; kd++) {
            int val_d = kd * dilation_d + nd;
            for (int ki = 0; ki < kernel_size_h; ki++) {
                int val_i = ki * dilation_h + ni;
                for (int kj = 0; kj < kernel_size_w; kj++) {
                    int val_j = kj * dilation_w + nj;
                    if (val_d >= 0 && val_d < depth &&
                        val_i >= 0 && val_i < height &&
                        val_j >= 0 && val_j < width) {
                        int attn_idx = ((((b*heads+h)*d_out+od)*h_out+oi)*w_out+oj)*k_vol
                                       + (kd*kernel_size_h+ki)*kernel_size_w+kj;
                        int val_idx  = ((((b*heads+h)*depth+val_d)*height+val_i)*width+val_j)*dim+d;
                        sum += attn[attn_idx] * value[val_idx];
                    }
                }
            }
        }
        int out_idx = ((((b*heads+h)*d_out+od)*h_out+oi)*w_out+oj)*dim+d;
        out[out_idx] = sum;
    }
}

// ===========================================================================
// Strided kernel variants – float16
// ===========================================================================

// ---------------------------------------------------------------------------
// 1D QK strided forward (f16)
// ---------------------------------------------------------------------------
kernel void natten1d_qk_strided_forward_f16(
    device const half*  query    [[buffer(0)]],
    device const half*  key      [[buffer(1)]],
    device half*        attn     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       length      [[buffer(5)]],
    constant int&       dim         [[buffer(6)]],
    constant int&       kernel_size [[buffer(7)]],
    constant int&       dilation    [[buffer(8)]],
    constant int&       stride_val  [[buffer(9)]],
    constant int&       l_out       [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]])
{
    int oi = gid.x;
    int h = gid.y;
    int b = gid.z;
    if (b >= batch_size || h >= heads || oi >= l_out) return;

    int i = oi * stride_val;
    int neighborhood_size = kernel_size / 2;
    int ni = get_window_start(i, length, kernel_size, neighborhood_size, dilation);

    for (int ki = 0; ki < kernel_size; ki++) {
        float sum = 0.0f;
        int key_i = ki * dilation + ni;
        if (key_i >= 0 && key_i < length) {
            for (int d = 0; d < dim; d++) {
                int q_idx = ((b * heads + h) * length + i) * dim + d;
                int k_idx = ((b * heads + h) * length + key_i) * dim + d;
                sum += float(query[q_idx]) * float(key[k_idx]);
            }
        }
        int attn_idx = ((b * heads + h) * l_out + oi) * kernel_size + ki;
        attn[attn_idx] = half(sum);
    }
}

// ---------------------------------------------------------------------------
// 1D AV strided forward (f16)
// ---------------------------------------------------------------------------
kernel void natten1d_av_strided_forward_f16(
    device const half*  attn    [[buffer(0)]],
    device const half*  value   [[buffer(1)]],
    device half*        out     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       length      [[buffer(5)]],
    constant int&       dim         [[buffer(6)]],
    constant int&       kernel_size [[buffer(7)]],
    constant int&       dilation    [[buffer(8)]],
    constant int&       stride_val  [[buffer(9)]],
    constant int&       l_out       [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]])
{
    int oi = gid.x;
    int h = gid.y;
    int b = gid.z;
    if (b >= batch_size || h >= heads || oi >= l_out) return;

    int i = oi * stride_val;
    int neighborhood_size = kernel_size / 2;
    int ni = get_window_start(i, length, kernel_size, neighborhood_size, dilation);

    for (int d = 0; d < dim; d++) {
        float sum = 0.0f;
        for (int ki = 0; ki < kernel_size; ki++) {
            int value_i = ki * dilation + ni;
            if (value_i >= 0 && value_i < length) {
                int attn_idx = ((b * heads + h) * l_out + oi) * kernel_size + ki;
                int val_idx  = ((b * heads + h) * length + value_i) * dim + d;
                sum += float(attn[attn_idx]) * float(value[val_idx]);
            }
        }
        int out_idx = ((b * heads + h) * l_out + oi) * dim + d;
        out[out_idx] = half(sum);
    }
}

// ---------------------------------------------------------------------------
// 2D QK strided forward (f16)
// ---------------------------------------------------------------------------
kernel void natten2d_qk_strided_forward_f16(
    device const half*  query    [[buffer(0)]],
    device const half*  key      [[buffer(1)]],
    device half*        attn     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       height      [[buffer(5)]],
    constant int&       width       [[buffer(6)]],
    constant int&       dim         [[buffer(7)]],
    constant int&       kernel_size_h [[buffer(8)]],
    constant int&       kernel_size_w [[buffer(9)]],
    constant int&       dilation_h    [[buffer(10)]],
    constant int&       dilation_w    [[buffer(11)]],
    constant int&       stride_h    [[buffer(12)]],
    constant int&       stride_w    [[buffer(13)]],
    constant int&       h_out       [[buffer(14)]],
    constant int&       w_out       [[buffer(15)]],
    uint3 gid [[thread_position_in_grid]])
{
    int oj = gid.x;
    int oi = gid.y;
    int bh = gid.z;
    int b = bh / heads;
    int h = bh % heads;
    if (b >= batch_size || oi >= h_out || oj >= w_out) return;

    int i = oi * stride_h;
    int j = oj * stride_w;
    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int ni = get_window_start(i, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = get_window_start(j, width, kernel_size_w, nh_size_w, dilation_w);

    for (int ki = 0; ki < kernel_size_h; ki++) {
        for (int kj = 0; kj < kernel_size_w; kj++) {
            float sum = 0.0f;
            int key_i = ki * dilation_h + ni;
            int key_j = kj * dilation_w + nj;
            if (key_i >= 0 && key_i < height && key_j >= 0 && key_j < width) {
                for (int d = 0; d < dim; d++) {
                    int q_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
                    int k_idx = (((b * heads + h) * height + key_i) * width + key_j) * dim + d;
                    sum += float(query[q_idx]) * float(key[k_idx]);
                }
            }
            int attn_idx = (((b * heads + h) * h_out + oi) * w_out + oj) * (kernel_size_h * kernel_size_w) + ki * kernel_size_w + kj;
            attn[attn_idx] = half(sum);
        }
    }
}

// ---------------------------------------------------------------------------
// 2D AV strided forward (f16)
// ---------------------------------------------------------------------------
kernel void natten2d_av_strided_forward_f16(
    device const half*  attn    [[buffer(0)]],
    device const half*  value   [[buffer(1)]],
    device half*        out     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       height      [[buffer(5)]],
    constant int&       width       [[buffer(6)]],
    constant int&       dim         [[buffer(7)]],
    constant int&       kernel_size_h [[buffer(8)]],
    constant int&       kernel_size_w [[buffer(9)]],
    constant int&       dilation_h    [[buffer(10)]],
    constant int&       dilation_w    [[buffer(11)]],
    constant int&       stride_h    [[buffer(12)]],
    constant int&       stride_w    [[buffer(13)]],
    constant int&       h_out       [[buffer(14)]],
    constant int&       w_out       [[buffer(15)]],
    uint3 gid [[thread_position_in_grid]])
{
    int oj = gid.x;
    int oi = gid.y;
    int bh = gid.z;
    int b = bh / heads;
    int h = bh % heads;
    if (b >= batch_size || oi >= h_out || oj >= w_out) return;

    int i = oi * stride_h;
    int j = oj * stride_w;
    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int ni = get_window_start(i, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = get_window_start(j, width, kernel_size_w, nh_size_w, dilation_w);

    for (int d = 0; d < dim; d++) {
        float sum = 0.0f;
        for (int ki = 0; ki < kernel_size_h; ki++) {
            for (int kj = 0; kj < kernel_size_w; kj++) {
                int value_i = ki * dilation_h + ni;
                int value_j = kj * dilation_w + nj;
                if (value_i >= 0 && value_i < height && value_j >= 0 && value_j < width) {
                    int attn_idx = (((b * heads + h) * h_out + oi) * w_out + oj) * (kernel_size_h * kernel_size_w) + ki * kernel_size_w + kj;
                    int val_idx  = (((b * heads + h) * height + value_i) * width + value_j) * dim + d;
                    sum += float(attn[attn_idx]) * float(value[val_idx]);
                }
            }
        }
        int out_idx = (((b * heads + h) * h_out + oi) * w_out + oj) * dim + d;
        out[out_idx] = half(sum);
    }
}

// ---------------------------------------------------------------------------
// 3D QK strided forward (f16)
// ---------------------------------------------------------------------------
kernel void natten3d_qk_strided_forward_f16(
    device const half*  query    [[buffer(0)]],
    device const half*  key      [[buffer(1)]],
    device half*        attn     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       depth       [[buffer(5)]],
    constant int&       height      [[buffer(6)]],
    constant int&       width       [[buffer(7)]],
    constant int&       dim         [[buffer(8)]],
    constant int&       kernel_size_d [[buffer(9)]],
    constant int&       kernel_size_h [[buffer(10)]],
    constant int&       kernel_size_w [[buffer(11)]],
    constant int&       dilation_d    [[buffer(12)]],
    constant int&       dilation_h    [[buffer(13)]],
    constant int&       dilation_w    [[buffer(14)]],
    constant int&       stride_d    [[buffer(15)]],
    constant int&       stride_h    [[buffer(16)]],
    constant int&       stride_w    [[buffer(17)]],
    constant int&       d_out       [[buffer(18)]],
    constant int&       h_out       [[buffer(19)]],
    constant int&       w_out       [[buffer(20)]],
    uint3 gid [[thread_position_in_grid]])
{
    int oj = gid.x;
    int oi = gid.y;
    int bhd = gid.z;
    int b  = bhd / (heads * d_out);
    int rem = bhd % (heads * d_out);
    int h  = rem / d_out;
    int od = rem % d_out;
    if (b >= batch_size || oi >= h_out || oj >= w_out || od >= d_out) return;

    int dp = od * stride_d;
    int i  = oi * stride_h;
    int j  = oj * stride_w;
    int nh_size_d = kernel_size_d / 2;
    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int nd = get_window_start(dp, depth, kernel_size_d, nh_size_d, dilation_d);
    int ni = get_window_start(i, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = get_window_start(j, width, kernel_size_w, nh_size_w, dilation_w);

    int k_vol = kernel_size_d * kernel_size_h * kernel_size_w;

    for (int kd = 0; kd < kernel_size_d; kd++) {
        int key_d = kd * dilation_d + nd;
        for (int ki = 0; ki < kernel_size_h; ki++) {
            int key_i = ki * dilation_h + ni;
            for (int kj = 0; kj < kernel_size_w; kj++) {
                int key_j = kj * dilation_w + nj;
                float sum = 0.0f;
                if (key_d >= 0 && key_d < depth &&
                    key_i >= 0 && key_i < height &&
                    key_j >= 0 && key_j < width) {
                    for (int d = 0; d < dim; d++) {
                        int q_idx = ((((b*heads+h)*depth+dp)*height+i)*width+j)*dim+d;
                        int k_idx = ((((b*heads+h)*depth+key_d)*height+key_i)*width+key_j)*dim+d;
                        sum += float(query[q_idx]) * float(key[k_idx]);
                    }
                }
                int attn_idx = ((((b*heads+h)*d_out+od)*h_out+oi)*w_out+oj)*k_vol
                               + (kd*kernel_size_h+ki)*kernel_size_w+kj;
                attn[attn_idx] = half(sum);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3D AV strided forward (f16)
// ---------------------------------------------------------------------------
kernel void natten3d_av_strided_forward_f16(
    device const half*  attn    [[buffer(0)]],
    device const half*  value   [[buffer(1)]],
    device half*        out     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       depth       [[buffer(5)]],
    constant int&       height      [[buffer(6)]],
    constant int&       width       [[buffer(7)]],
    constant int&       dim         [[buffer(8)]],
    constant int&       kernel_size_d [[buffer(9)]],
    constant int&       kernel_size_h [[buffer(10)]],
    constant int&       kernel_size_w [[buffer(11)]],
    constant int&       dilation_d    [[buffer(12)]],
    constant int&       dilation_h    [[buffer(13)]],
    constant int&       dilation_w    [[buffer(14)]],
    constant int&       stride_d    [[buffer(15)]],
    constant int&       stride_h    [[buffer(16)]],
    constant int&       stride_w    [[buffer(17)]],
    constant int&       d_out       [[buffer(18)]],
    constant int&       h_out       [[buffer(19)]],
    constant int&       w_out       [[buffer(20)]],
    uint3 gid [[thread_position_in_grid]])
{
    int oj = gid.x;
    int oi = gid.y;
    int bhd = gid.z;
    int b  = bhd / (heads * d_out);
    int rem = bhd % (heads * d_out);
    int h  = rem / d_out;
    int od = rem % d_out;
    if (b >= batch_size || oi >= h_out || oj >= w_out || od >= d_out) return;

    int dp = od * stride_d;
    int i  = oi * stride_h;
    int j  = oj * stride_w;
    int nh_size_d = kernel_size_d / 2;
    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int nd = get_window_start(dp, depth, kernel_size_d, nh_size_d, dilation_d);
    int ni = get_window_start(i, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = get_window_start(j, width, kernel_size_w, nh_size_w, dilation_w);

    int k_vol = kernel_size_d * kernel_size_h * kernel_size_w;

    for (int d = 0; d < dim; d++) {
        float sum = 0.0f;
        for (int kd = 0; kd < kernel_size_d; kd++) {
            int val_d = kd * dilation_d + nd;
            for (int ki = 0; ki < kernel_size_h; ki++) {
                int val_i = ki * dilation_h + ni;
                for (int kj = 0; kj < kernel_size_w; kj++) {
                    int val_j = kj * dilation_w + nj;
                    if (val_d >= 0 && val_d < depth &&
                        val_i >= 0 && val_i < height &&
                        val_j >= 0 && val_j < width) {
                        int attn_idx = ((((b*heads+h)*d_out+od)*h_out+oi)*w_out+oj)*k_vol
                                       + (kd*kernel_size_h+ki)*kernel_size_w+kj;
                        int val_idx  = ((((b*heads+h)*depth+val_d)*height+val_i)*width+val_j)*dim+d;
                        sum += float(attn[attn_idx]) * float(value[val_idx]);
                    }
                }
            }
        }
        int out_idx = ((((b*heads+h)*d_out+od)*h_out+oi)*w_out+oj)*dim+d;
        out[out_idx] = half(sum);
    }
}

// ===========================================================================
// Combined causal+strided kernel variants
// ===========================================================================

// ---------------------------------------------------------------------------
// 1D QK causal+strided forward (float32)  –  grid: (L_out, H, B)
// ---------------------------------------------------------------------------
kernel void natten1d_qk_causal_strided_forward(
    device const float* query    [[buffer(0)]],
    device const float* key      [[buffer(1)]],
    device float*       attn     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       length      [[buffer(5)]],
    constant int&       dim         [[buffer(6)]],
    constant int&       kernel_size [[buffer(7)]],
    constant int&       dilation    [[buffer(8)]],
    constant int&       stride_val  [[buffer(9)]],
    constant int&       l_out       [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]])
{
    int oi = gid.x;
    int h = gid.y;
    int b = gid.z;
    if (b >= batch_size || h >= heads || oi >= l_out) return;

    int i = oi * stride_val;
    int ni = get_causal_window_start(i, kernel_size, dilation);

    int q_base = ((b * heads + h) * length + i) * dim;
    for (int ki = 0; ki < kernel_size; ki++) {
        int key_i = ki * dilation + ni;
        float sum = 0.0f;
        bool valid = (key_i >= 0 && key_i < length && key_i <= i);
        if (valid) {
            int k_base = ((b * heads + h) * length + key_i) * dim;
            sum = dot_f32(query + q_base, key + k_base, dim);
        }
        int attn_idx = ((b * heads + h) * l_out + oi) * kernel_size + ki;
        attn[attn_idx] = valid ? sum : -INFINITY;
    }
}

// ---------------------------------------------------------------------------
// 1D AV causal+strided forward (float32)  –  grid: (L_out, H, B)
// ---------------------------------------------------------------------------
kernel void natten1d_av_causal_strided_forward(
    device const float* attn    [[buffer(0)]],
    device const float* value   [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       length      [[buffer(5)]],
    constant int&       dim         [[buffer(6)]],
    constant int&       kernel_size [[buffer(7)]],
    constant int&       dilation    [[buffer(8)]],
    constant int&       stride_val  [[buffer(9)]],
    constant int&       l_out       [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]])
{
    int oi = gid.x;
    int h = gid.y;
    int b = gid.z;
    if (b >= batch_size || h >= heads || oi >= l_out) return;

    int i = oi * stride_val;
    int ni = get_causal_window_start(i, kernel_size, dilation);

    for (int d = 0; d < dim; d++) {
        float sum = 0.0f;
        for (int ki = 0; ki < kernel_size; ki++) {
            int value_i = ki * dilation + ni;
            if (value_i >= 0 && value_i < length && value_i <= i) {
                int attn_idx = ((b * heads + h) * l_out + oi) * kernel_size + ki;
                int val_idx  = ((b * heads + h) * length + value_i) * dim + d;
                sum += attn[attn_idx] * value[val_idx];
            }
        }
        int out_idx = ((b * heads + h) * l_out + oi) * dim + d;
        out[out_idx] = sum;
    }
}

// ---------------------------------------------------------------------------
// 2D QK causal+strided forward (float32)  –  grid: (Wi_out, Hi_out, B*H)
// ---------------------------------------------------------------------------
kernel void natten2d_qk_causal_strided_forward(
    device const float* query    [[buffer(0)]],
    device const float* key      [[buffer(1)]],
    device float*       attn     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       height      [[buffer(5)]],
    constant int&       width       [[buffer(6)]],
    constant int&       dim         [[buffer(7)]],
    constant int&       kernel_size_h [[buffer(8)]],
    constant int&       kernel_size_w [[buffer(9)]],
    constant int&       dilation_h    [[buffer(10)]],
    constant int&       dilation_w    [[buffer(11)]],
    constant int&       causal_h    [[buffer(12)]],
    constant int&       causal_w    [[buffer(13)]],
    constant int&       stride_h    [[buffer(14)]],
    constant int&       stride_w    [[buffer(15)]],
    constant int&       h_out       [[buffer(16)]],
    constant int&       w_out       [[buffer(17)]],
    uint3 gid [[thread_position_in_grid]])
{
    int oj = gid.x;
    int oi = gid.y;
    int bh = gid.z;
    int b = bh / heads;
    int h = bh % heads;
    if (b >= batch_size || oi >= h_out || oj >= w_out) return;

    int i = oi * stride_h;
    int j = oj * stride_w;

    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int ni = causal_h ? get_causal_window_start(i, kernel_size_h, dilation_h)
                      : get_window_start(i, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = causal_w ? get_causal_window_start(j, kernel_size_w, dilation_w)
                      : get_window_start(j, width, kernel_size_w, nh_size_w, dilation_w);

    int q_base = (((b * heads + h) * height + i) * width + j) * dim;
    for (int ki = 0; ki < kernel_size_h; ki++) {
        int key_i = ki * dilation_h + ni;
        bool valid_i = (key_i >= 0 && key_i < height);
        if (causal_h) valid_i = valid_i && (key_i <= i);
        for (int kj = 0; kj < kernel_size_w; kj++) {
            int key_j = kj * dilation_w + nj;
            bool valid_j = (key_j >= 0 && key_j < width);
            if (causal_w) valid_j = valid_j && (key_j <= j);
            bool valid = valid_i && valid_j;
            float sum = 0.0f;
            if (valid) {
                int k_base = (((b * heads + h) * height + key_i) * width + key_j) * dim;
                sum = dot_f32(query + q_base, key + k_base, dim);
            }
            int attn_idx = (((b * heads + h) * h_out + oi) * w_out + oj) * (kernel_size_h * kernel_size_w) + ki * kernel_size_w + kj;
            attn[attn_idx] = valid ? sum : -INFINITY;
        }
    }
}

// ---------------------------------------------------------------------------
// 2D AV causal+strided forward (float32)  –  grid: (Wi_out, Hi_out, B*H)
// ---------------------------------------------------------------------------
kernel void natten2d_av_causal_strided_forward(
    device const float* attn    [[buffer(0)]],
    device const float* value   [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       height      [[buffer(5)]],
    constant int&       width       [[buffer(6)]],
    constant int&       dim         [[buffer(7)]],
    constant int&       kernel_size_h [[buffer(8)]],
    constant int&       kernel_size_w [[buffer(9)]],
    constant int&       dilation_h    [[buffer(10)]],
    constant int&       dilation_w    [[buffer(11)]],
    constant int&       causal_h    [[buffer(12)]],
    constant int&       causal_w    [[buffer(13)]],
    constant int&       stride_h    [[buffer(14)]],
    constant int&       stride_w    [[buffer(15)]],
    constant int&       h_out       [[buffer(16)]],
    constant int&       w_out       [[buffer(17)]],
    uint3 gid [[thread_position_in_grid]])
{
    int oj = gid.x;
    int oi = gid.y;
    int bh = gid.z;
    int b = bh / heads;
    int h = bh % heads;
    if (b >= batch_size || oi >= h_out || oj >= w_out) return;

    int i = oi * stride_h;
    int j = oj * stride_w;

    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int ni = causal_h ? get_causal_window_start(i, kernel_size_h, dilation_h)
                      : get_window_start(i, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = causal_w ? get_causal_window_start(j, kernel_size_w, dilation_w)
                      : get_window_start(j, width, kernel_size_w, nh_size_w, dilation_w);

    for (int d = 0; d < dim; d++) {
        float sum = 0.0f;
        for (int ki = 0; ki < kernel_size_h; ki++) {
            int val_i = ki * dilation_h + ni;
            bool valid_i = (val_i >= 0 && val_i < height);
            if (causal_h) valid_i = valid_i && (val_i <= i);
            for (int kj = 0; kj < kernel_size_w; kj++) {
                int val_j = kj * dilation_w + nj;
                bool valid_j = (val_j >= 0 && val_j < width);
                if (causal_w) valid_j = valid_j && (val_j <= j);
                if (valid_i && valid_j) {
                    int attn_idx = (((b * heads + h) * h_out + oi) * w_out + oj) * (kernel_size_h * kernel_size_w) + ki * kernel_size_w + kj;
                    int val_idx  = (((b * heads + h) * height + val_i) * width + val_j) * dim + d;
                    sum += attn[attn_idx] * value[val_idx];
                }
            }
        }
        int out_idx = (((b * heads + h) * h_out + oi) * w_out + oj) * dim + d;
        out[out_idx] = sum;
    }
}

// ---------------------------------------------------------------------------
// 3D QK causal+strided forward (float32)  –  grid: (Wi_out, Hi_out, B*H*Dp_out)
// ---------------------------------------------------------------------------
kernel void natten3d_qk_causal_strided_forward(
    device const float* query    [[buffer(0)]],
    device const float* key      [[buffer(1)]],
    device float*       attn     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       depth       [[buffer(5)]],
    constant int&       height      [[buffer(6)]],
    constant int&       width       [[buffer(7)]],
    constant int&       dim         [[buffer(8)]],
    constant int&       kernel_size_d [[buffer(9)]],
    constant int&       kernel_size_h [[buffer(10)]],
    constant int&       kernel_size_w [[buffer(11)]],
    constant int&       dilation_d    [[buffer(12)]],
    constant int&       dilation_h    [[buffer(13)]],
    constant int&       dilation_w    [[buffer(14)]],
    constant int&       causal_d    [[buffer(15)]],
    constant int&       causal_h    [[buffer(16)]],
    constant int&       causal_w    [[buffer(17)]],
    constant int&       stride_d    [[buffer(18)]],
    constant int&       stride_h    [[buffer(19)]],
    constant int&       stride_w    [[buffer(20)]],
    constant int&       d_out       [[buffer(21)]],
    constant int&       h_out       [[buffer(22)]],
    constant int&       w_out       [[buffer(23)]],
    uint3 gid [[thread_position_in_grid]])
{
    int oj = gid.x;
    int oi = gid.y;
    int bhd = gid.z;
    int b  = bhd / (heads * d_out);
    int rem = bhd % (heads * d_out);
    int h  = rem / d_out;
    int od = rem % d_out;
    if (b >= batch_size || oi >= h_out || oj >= w_out || od >= d_out) return;

    int dp = od * stride_d;
    int i  = oi * stride_h;
    int j  = oj * stride_w;

    int nh_size_d = kernel_size_d / 2;
    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int nd = causal_d ? get_causal_window_start(dp, kernel_size_d, dilation_d)
                      : get_window_start(dp, depth, kernel_size_d, nh_size_d, dilation_d);
    int ni = causal_h ? get_causal_window_start(i, kernel_size_h, dilation_h)
                      : get_window_start(i, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = causal_w ? get_causal_window_start(j, kernel_size_w, dilation_w)
                      : get_window_start(j, width, kernel_size_w, nh_size_w, dilation_w);

    int k_vol = kernel_size_d * kernel_size_h * kernel_size_w;
    int q_base = ((((b*heads+h)*depth+dp)*height+i)*width+j)*dim;
    for (int kd = 0; kd < kernel_size_d; kd++) {
        int key_d = kd * dilation_d + nd;
        bool valid_d = (key_d >= 0 && key_d < depth);
        if (causal_d) valid_d = valid_d && (key_d <= dp);
        for (int ki = 0; ki < kernel_size_h; ki++) {
            int key_i = ki * dilation_h + ni;
            bool valid_i = (key_i >= 0 && key_i < height);
            if (causal_h) valid_i = valid_i && (key_i <= i);
            for (int kj = 0; kj < kernel_size_w; kj++) {
                int key_j = kj * dilation_w + nj;
                bool valid_j = (key_j >= 0 && key_j < width);
                if (causal_w) valid_j = valid_j && (key_j <= j);
                bool valid = valid_d && valid_i && valid_j;
                float sum = 0.0f;
                if (valid) {
                    int k_base = ((((b*heads+h)*depth+key_d)*height+key_i)*width+key_j)*dim;
                    sum = dot_f32(query + q_base, key + k_base, dim);
                }
                int attn_idx = ((((b*heads+h)*d_out+od)*h_out+oi)*w_out+oj)*k_vol
                               + (kd*kernel_size_h+ki)*kernel_size_w+kj;
                attn[attn_idx] = valid ? sum : -INFINITY;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3D AV causal+strided forward (float32)  –  grid: (Wi_out, Hi_out, B*H*Dp_out)
// ---------------------------------------------------------------------------
kernel void natten3d_av_causal_strided_forward(
    device const float* attn    [[buffer(0)]],
    device const float* value   [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       depth       [[buffer(5)]],
    constant int&       height      [[buffer(6)]],
    constant int&       width       [[buffer(7)]],
    constant int&       dim         [[buffer(8)]],
    constant int&       kernel_size_d [[buffer(9)]],
    constant int&       kernel_size_h [[buffer(10)]],
    constant int&       kernel_size_w [[buffer(11)]],
    constant int&       dilation_d    [[buffer(12)]],
    constant int&       dilation_h    [[buffer(13)]],
    constant int&       dilation_w    [[buffer(14)]],
    constant int&       causal_d    [[buffer(15)]],
    constant int&       causal_h    [[buffer(16)]],
    constant int&       causal_w    [[buffer(17)]],
    constant int&       stride_d    [[buffer(18)]],
    constant int&       stride_h    [[buffer(19)]],
    constant int&       stride_w    [[buffer(20)]],
    constant int&       d_out       [[buffer(21)]],
    constant int&       h_out       [[buffer(22)]],
    constant int&       w_out       [[buffer(23)]],
    uint3 gid [[thread_position_in_grid]])
{
    int oj = gid.x;
    int oi = gid.y;
    int bhd = gid.z;
    int b  = bhd / (heads * d_out);
    int rem = bhd % (heads * d_out);
    int h  = rem / d_out;
    int od = rem % d_out;
    if (b >= batch_size || oi >= h_out || oj >= w_out || od >= d_out) return;

    int dp = od * stride_d;
    int i  = oi * stride_h;
    int j  = oj * stride_w;

    int nh_size_d = kernel_size_d / 2;
    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int nd = causal_d ? get_causal_window_start(dp, kernel_size_d, dilation_d)
                      : get_window_start(dp, depth, kernel_size_d, nh_size_d, dilation_d);
    int ni = causal_h ? get_causal_window_start(i, kernel_size_h, dilation_h)
                      : get_window_start(i, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = causal_w ? get_causal_window_start(j, kernel_size_w, dilation_w)
                      : get_window_start(j, width, kernel_size_w, nh_size_w, dilation_w);

    int k_vol = kernel_size_d * kernel_size_h * kernel_size_w;
    for (int d = 0; d < dim; d++) {
        float sum = 0.0f;
        for (int kd = 0; kd < kernel_size_d; kd++) {
            int val_d = kd * dilation_d + nd;
            bool valid_d = (val_d >= 0 && val_d < depth);
            if (causal_d) valid_d = valid_d && (val_d <= dp);
            for (int ki = 0; ki < kernel_size_h; ki++) {
                int val_i = ki * dilation_h + ni;
                bool valid_i = (val_i >= 0 && val_i < height);
                if (causal_h) valid_i = valid_i && (val_i <= i);
                for (int kj = 0; kj < kernel_size_w; kj++) {
                    int val_j = kj * dilation_w + nj;
                    bool valid_j = (val_j >= 0 && val_j < width);
                    if (causal_w) valid_j = valid_j && (val_j <= j);
                    if (valid_d && valid_i && valid_j) {
                        int attn_idx = ((((b*heads+h)*d_out+od)*h_out+oi)*w_out+oj)*k_vol
                                       + (kd*kernel_size_h+ki)*kernel_size_w+kj;
                        int val_idx  = ((((b*heads+h)*depth+val_d)*height+val_i)*width+val_j)*dim+d;
                        sum += attn[attn_idx] * value[val_idx];
                    }
                }
            }
        }
        int out_idx = ((((b*heads+h)*d_out+od)*h_out+oi)*w_out+oj)*dim+d;
        out[out_idx] = sum;
    }
}

// ---------------------------------------------------------------------------
// f16 causal+strided variants
// ---------------------------------------------------------------------------

kernel void natten1d_qk_causal_strided_forward_f16(
    device const half*  query    [[buffer(0)]],
    device const half*  key      [[buffer(1)]],
    device half*        attn     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       length      [[buffer(5)]],
    constant int&       dim         [[buffer(6)]],
    constant int&       kernel_size [[buffer(7)]],
    constant int&       dilation    [[buffer(8)]],
    constant int&       stride_val  [[buffer(9)]],
    constant int&       l_out       [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]])
{
    int oi = gid.x;
    int h = gid.y;
    int b = gid.z;
    if (b >= batch_size || h >= heads || oi >= l_out) return;

    int i = oi * stride_val;
    int ni = get_causal_window_start(i, kernel_size, dilation);

    int q_base = ((b * heads + h) * length + i) * dim;
    for (int ki = 0; ki < kernel_size; ki++) {
        int key_i = ki * dilation + ni;
        float sum = 0.0f;
        bool valid = (key_i >= 0 && key_i < length && key_i <= i);
        if (valid) {
            int k_base = ((b * heads + h) * length + key_i) * dim;
            sum = dot_f16(query + q_base, key + k_base, dim);
        }
        int attn_idx = ((b * heads + h) * l_out + oi) * kernel_size + ki;
        attn[attn_idx] = valid ? half(sum) : half(-INFINITY);
    }
}

kernel void natten1d_av_causal_strided_forward_f16(
    device const half*  attn    [[buffer(0)]],
    device const half*  value   [[buffer(1)]],
    device half*        out     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       length      [[buffer(5)]],
    constant int&       dim         [[buffer(6)]],
    constant int&       kernel_size [[buffer(7)]],
    constant int&       dilation    [[buffer(8)]],
    constant int&       stride_val  [[buffer(9)]],
    constant int&       l_out       [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]])
{
    int oi = gid.x;
    int h = gid.y;
    int b = gid.z;
    if (b >= batch_size || h >= heads || oi >= l_out) return;

    int i = oi * stride_val;
    int ni = get_causal_window_start(i, kernel_size, dilation);

    for (int d = 0; d < dim; d++) {
        float sum = 0.0f;
        for (int ki = 0; ki < kernel_size; ki++) {
            int value_i = ki * dilation + ni;
            if (value_i >= 0 && value_i < length && value_i <= i) {
                int attn_idx = ((b * heads + h) * l_out + oi) * kernel_size + ki;
                int val_idx  = ((b * heads + h) * length + value_i) * dim + d;
                sum += float(attn[attn_idx]) * float(value[val_idx]);
            }
        }
        int out_idx = ((b * heads + h) * l_out + oi) * dim + d;
        out[out_idx] = half(sum);
    }
}

kernel void natten2d_qk_causal_strided_forward_f16(
    device const half*  query    [[buffer(0)]],
    device const half*  key      [[buffer(1)]],
    device half*        attn     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       height      [[buffer(5)]],
    constant int&       width       [[buffer(6)]],
    constant int&       dim         [[buffer(7)]],
    constant int&       kernel_size_h [[buffer(8)]],
    constant int&       kernel_size_w [[buffer(9)]],
    constant int&       dilation_h    [[buffer(10)]],
    constant int&       dilation_w    [[buffer(11)]],
    constant int&       causal_h    [[buffer(12)]],
    constant int&       causal_w    [[buffer(13)]],
    constant int&       stride_h    [[buffer(14)]],
    constant int&       stride_w    [[buffer(15)]],
    constant int&       h_out       [[buffer(16)]],
    constant int&       w_out       [[buffer(17)]],
    uint3 gid [[thread_position_in_grid]])
{
    int oj = gid.x;
    int oi = gid.y;
    int bh = gid.z;
    int b = bh / heads;
    int h = bh % heads;
    if (b >= batch_size || oi >= h_out || oj >= w_out) return;

    int i = oi * stride_h;
    int j = oj * stride_w;

    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int ni = causal_h ? get_causal_window_start(i, kernel_size_h, dilation_h)
                      : get_window_start(i, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = causal_w ? get_causal_window_start(j, kernel_size_w, dilation_w)
                      : get_window_start(j, width, kernel_size_w, nh_size_w, dilation_w);

    int q_base = (((b * heads + h) * height + i) * width + j) * dim;
    for (int ki = 0; ki < kernel_size_h; ki++) {
        int key_i = ki * dilation_h + ni;
        bool valid_i = (key_i >= 0 && key_i < height);
        if (causal_h) valid_i = valid_i && (key_i <= i);
        for (int kj = 0; kj < kernel_size_w; kj++) {
            int key_j = kj * dilation_w + nj;
            bool valid_j = (key_j >= 0 && key_j < width);
            if (causal_w) valid_j = valid_j && (key_j <= j);
            bool valid = valid_i && valid_j;
            float sum = 0.0f;
            if (valid) {
                int k_base = (((b * heads + h) * height + key_i) * width + key_j) * dim;
                sum = dot_f16(query + q_base, key + k_base, dim);
            }
            int attn_idx = (((b * heads + h) * h_out + oi) * w_out + oj) * (kernel_size_h * kernel_size_w) + ki * kernel_size_w + kj;
            attn[attn_idx] = valid ? half(sum) : half(-INFINITY);
        }
    }
}

kernel void natten2d_av_causal_strided_forward_f16(
    device const half*  attn    [[buffer(0)]],
    device const half*  value   [[buffer(1)]],
    device half*        out     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       height      [[buffer(5)]],
    constant int&       width       [[buffer(6)]],
    constant int&       dim         [[buffer(7)]],
    constant int&       kernel_size_h [[buffer(8)]],
    constant int&       kernel_size_w [[buffer(9)]],
    constant int&       dilation_h    [[buffer(10)]],
    constant int&       dilation_w    [[buffer(11)]],
    constant int&       causal_h    [[buffer(12)]],
    constant int&       causal_w    [[buffer(13)]],
    constant int&       stride_h    [[buffer(14)]],
    constant int&       stride_w    [[buffer(15)]],
    constant int&       h_out       [[buffer(16)]],
    constant int&       w_out       [[buffer(17)]],
    uint3 gid [[thread_position_in_grid]])
{
    int oj = gid.x;
    int oi = gid.y;
    int bh = gid.z;
    int b = bh / heads;
    int h = bh % heads;
    if (b >= batch_size || oi >= h_out || oj >= w_out) return;

    int i = oi * stride_h;
    int j = oj * stride_w;

    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int ni = causal_h ? get_causal_window_start(i, kernel_size_h, dilation_h)
                      : get_window_start(i, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = causal_w ? get_causal_window_start(j, kernel_size_w, dilation_w)
                      : get_window_start(j, width, kernel_size_w, nh_size_w, dilation_w);

    for (int d = 0; d < dim; d++) {
        float sum = 0.0f;
        for (int ki = 0; ki < kernel_size_h; ki++) {
            int val_i = ki * dilation_h + ni;
            bool valid_i = (val_i >= 0 && val_i < height);
            if (causal_h) valid_i = valid_i && (val_i <= i);
            for (int kj = 0; kj < kernel_size_w; kj++) {
                int val_j = kj * dilation_w + nj;
                bool valid_j = (val_j >= 0 && val_j < width);
                if (causal_w) valid_j = valid_j && (val_j <= j);
                if (valid_i && valid_j) {
                    int attn_idx = (((b * heads + h) * h_out + oi) * w_out + oj) * (kernel_size_h * kernel_size_w) + ki * kernel_size_w + kj;
                    int val_idx  = (((b * heads + h) * height + val_i) * width + val_j) * dim + d;
                    sum += float(attn[attn_idx]) * float(value[val_idx]);
                }
            }
        }
        int out_idx = (((b * heads + h) * h_out + oi) * w_out + oj) * dim + d;
        out[out_idx] = half(sum);
    }
}

kernel void natten3d_qk_causal_strided_forward_f16(
    device const half*  query    [[buffer(0)]],
    device const half*  key      [[buffer(1)]],
    device half*        attn     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       depth       [[buffer(5)]],
    constant int&       height      [[buffer(6)]],
    constant int&       width       [[buffer(7)]],
    constant int&       dim         [[buffer(8)]],
    constant int&       kernel_size_d [[buffer(9)]],
    constant int&       kernel_size_h [[buffer(10)]],
    constant int&       kernel_size_w [[buffer(11)]],
    constant int&       dilation_d    [[buffer(12)]],
    constant int&       dilation_h    [[buffer(13)]],
    constant int&       dilation_w    [[buffer(14)]],
    constant int&       causal_d    [[buffer(15)]],
    constant int&       causal_h    [[buffer(16)]],
    constant int&       causal_w    [[buffer(17)]],
    constant int&       stride_d    [[buffer(18)]],
    constant int&       stride_h    [[buffer(19)]],
    constant int&       stride_w    [[buffer(20)]],
    constant int&       d_out       [[buffer(21)]],
    constant int&       h_out       [[buffer(22)]],
    constant int&       w_out       [[buffer(23)]],
    uint3 gid [[thread_position_in_grid]])
{
    int oj = gid.x;
    int oi = gid.y;
    int bhd = gid.z;
    int b  = bhd / (heads * d_out);
    int rem = bhd % (heads * d_out);
    int h  = rem / d_out;
    int od = rem % d_out;
    if (b >= batch_size || oi >= h_out || oj >= w_out || od >= d_out) return;

    int dp = od * stride_d;
    int i  = oi * stride_h;
    int j  = oj * stride_w;

    int nh_size_d = kernel_size_d / 2;
    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int nd = causal_d ? get_causal_window_start(dp, kernel_size_d, dilation_d)
                      : get_window_start(dp, depth, kernel_size_d, nh_size_d, dilation_d);
    int ni = causal_h ? get_causal_window_start(i, kernel_size_h, dilation_h)
                      : get_window_start(i, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = causal_w ? get_causal_window_start(j, kernel_size_w, dilation_w)
                      : get_window_start(j, width, kernel_size_w, nh_size_w, dilation_w);

    int k_vol = kernel_size_d * kernel_size_h * kernel_size_w;
    int q_base = ((((b*heads+h)*depth+dp)*height+i)*width+j)*dim;
    for (int kd = 0; kd < kernel_size_d; kd++) {
        int key_d = kd * dilation_d + nd;
        bool valid_d = (key_d >= 0 && key_d < depth);
        if (causal_d) valid_d = valid_d && (key_d <= dp);
        for (int ki = 0; ki < kernel_size_h; ki++) {
            int key_i = ki * dilation_h + ni;
            bool valid_i = (key_i >= 0 && key_i < height);
            if (causal_h) valid_i = valid_i && (key_i <= i);
            for (int kj = 0; kj < kernel_size_w; kj++) {
                int key_j = kj * dilation_w + nj;
                bool valid_j = (key_j >= 0 && key_j < width);
                if (causal_w) valid_j = valid_j && (key_j <= j);
                bool valid = valid_d && valid_i && valid_j;
                float sum = 0.0f;
                if (valid) {
                    int k_base = ((((b*heads+h)*depth+key_d)*height+key_i)*width+key_j)*dim;
                    sum = dot_f16(query + q_base, key + k_base, dim);
                }
                int attn_idx = ((((b*heads+h)*d_out+od)*h_out+oi)*w_out+oj)*k_vol
                               + (kd*kernel_size_h+ki)*kernel_size_w+kj;
                attn[attn_idx] = valid ? half(sum) : half(-INFINITY);
            }
        }
    }
}

kernel void natten3d_av_causal_strided_forward_f16(
    device const half*  attn    [[buffer(0)]],
    device const half*  value   [[buffer(1)]],
    device half*        out     [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       depth       [[buffer(5)]],
    constant int&       height      [[buffer(6)]],
    constant int&       width       [[buffer(7)]],
    constant int&       dim         [[buffer(8)]],
    constant int&       kernel_size_d [[buffer(9)]],
    constant int&       kernel_size_h [[buffer(10)]],
    constant int&       kernel_size_w [[buffer(11)]],
    constant int&       dilation_d    [[buffer(12)]],
    constant int&       dilation_h    [[buffer(13)]],
    constant int&       dilation_w    [[buffer(14)]],
    constant int&       causal_d    [[buffer(15)]],
    constant int&       causal_h    [[buffer(16)]],
    constant int&       causal_w    [[buffer(17)]],
    constant int&       stride_d    [[buffer(18)]],
    constant int&       stride_h    [[buffer(19)]],
    constant int&       stride_w    [[buffer(20)]],
    constant int&       d_out       [[buffer(21)]],
    constant int&       h_out       [[buffer(22)]],
    constant int&       w_out       [[buffer(23)]],
    uint3 gid [[thread_position_in_grid]])
{
    int oj = gid.x;
    int oi = gid.y;
    int bhd = gid.z;
    int b  = bhd / (heads * d_out);
    int rem = bhd % (heads * d_out);
    int h  = rem / d_out;
    int od = rem % d_out;
    if (b >= batch_size || oi >= h_out || oj >= w_out || od >= d_out) return;

    int dp = od * stride_d;
    int i  = oi * stride_h;
    int j  = oj * stride_w;

    int nh_size_d = kernel_size_d / 2;
    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int nd = causal_d ? get_causal_window_start(dp, kernel_size_d, dilation_d)
                      : get_window_start(dp, depth, kernel_size_d, nh_size_d, dilation_d);
    int ni = causal_h ? get_causal_window_start(i, kernel_size_h, dilation_h)
                      : get_window_start(i, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = causal_w ? get_causal_window_start(j, kernel_size_w, dilation_w)
                      : get_window_start(j, width, kernel_size_w, nh_size_w, dilation_w);

    int k_vol = kernel_size_d * kernel_size_h * kernel_size_w;
    for (int d = 0; d < dim; d++) {
        float sum = 0.0f;
        for (int kd = 0; kd < kernel_size_d; kd++) {
            int val_d = kd * dilation_d + nd;
            bool valid_d = (val_d >= 0 && val_d < depth);
            if (causal_d) valid_d = valid_d && (val_d <= dp);
            for (int ki = 0; ki < kernel_size_h; ki++) {
                int val_i = ki * dilation_h + ni;
                bool valid_i = (val_i >= 0 && val_i < height);
                if (causal_h) valid_i = valid_i && (val_i <= i);
                for (int kj = 0; kj < kernel_size_w; kj++) {
                    int val_j = kj * dilation_w + nj;
                    bool valid_j = (val_j >= 0 && val_j < width);
                    if (causal_w) valid_j = valid_j && (val_j <= j);
                    if (valid_d && valid_i && valid_j) {
                        int attn_idx = ((((b*heads+h)*d_out+od)*h_out+oi)*w_out+oj)*k_vol
                                       + (kd*kernel_size_h+ki)*kernel_size_w+kj;
                        int val_idx  = ((((b*heads+h)*depth+val_d)*height+val_i)*width+val_j)*dim+d;
                        sum += float(attn[attn_idx]) * float(value[val_idx]);
                    }
                }
            }
        }
        int out_idx = ((((b*heads+h)*d_out+od)*h_out+oi)*w_out+oj)*dim+d;
        out[out_idx] = half(sum);
    }
}

// ===========================================================================
// BACKWARD KERNELS
// ===========================================================================

// ---------------------------------------------------------------------------
// 1D QK backward: d_query  –  grid: (L, H, B)
// d_query[i,d] = sum_ki( d_attn[oi,ki] * key[ni + ki*dil, d] )
// Supports causal masking and strided output.
// ---------------------------------------------------------------------------
kernel void natten1d_q_backward(
    device const float* d_attn   [[buffer(0)]],
    device const float* key      [[buffer(1)]],
    device float*       d_query  [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       length      [[buffer(5)]],
    constant int&       dim         [[buffer(6)]],
    constant int&       kernel_size [[buffer(7)]],
    constant int&       dilation    [[buffer(8)]],
    constant int&       stride_val  [[buffer(9)]],
    constant int&       l_out       [[buffer(10)]],
    constant int&       is_causal   [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]])
{
    int i = gid.x;
    int h = gid.y;
    int b = gid.z;
    if (b >= batch_size || h >= heads || i >= length) return;

    int oi = i / stride_val;
    bool query_valid = (i % stride_val == 0) && (oi < l_out);
    if (!query_valid) {
        for (int d = 0; d < dim; d++) {
            d_query[((b * heads + h) * length + i) * dim + d] = 0.0f;
        }
        return;
    }

    int neighborhood_size = kernel_size / 2;
    int ni = is_causal ? get_causal_window_start(i, kernel_size, dilation)
                       : get_window_start(i, length, kernel_size, neighborhood_size, dilation);

    for (int d = 0; d < dim; d++) {
        float grad = 0.0f;
        for (int ki = 0; ki < kernel_size; ki++) {
            int key_i = ki * dilation + ni;
            bool valid = is_causal ? (key_i >= 0 && key_i < length && key_i <= i)
                                   : (key_i >= 0 && key_i < length);
            if (valid) {
                int da_idx = ((b * heads + h) * l_out + oi) * kernel_size + ki;
                int k_idx  = ((b * heads + h) * length + key_i) * dim + d;
                grad += d_attn[da_idx] * key[k_idx];
            }
        }
        int q_idx = ((b * heads + h) * length + i) * dim + d;
        d_query[q_idx] = grad;
    }
}

// ---------------------------------------------------------------------------
// 1D QK backward: d_key  –  grid: (L, H, B)
// Reverse iteration: find all output positions whose window contains this key
// Supports causal masking and strided output.
// ---------------------------------------------------------------------------
kernel void natten1d_k_backward(
    device const float* d_attn   [[buffer(0)]],
    device const float* query    [[buffer(1)]],
    device float*       d_key    [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       length      [[buffer(5)]],
    constant int&       dim         [[buffer(6)]],
    constant int&       kernel_size [[buffer(7)]],
    constant int&       dilation    [[buffer(8)]],
    constant int&       stride_val  [[buffer(9)]],
    constant int&       l_out       [[buffer(10)]],
    constant int&       is_causal   [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]])
{
    int i = gid.x;
    int h = gid.y;
    int b = gid.z;
    if (b >= batch_size || h >= heads || i >= length) return;

    int neighborhood_size = kernel_size / 2;

    for (int d = 0; d < dim; d++) {
        float grad = 0.0f;
        for (int oi = 0; oi < l_out; oi++) {
            int qi = oi * stride_val;
            int ni = is_causal ? get_causal_window_start(qi, kernel_size, dilation)
                               : get_window_start(qi, length, kernel_size, neighborhood_size, dilation);
            for (int ki = 0; ki < kernel_size; ki++) {
                int key_i = ki * dilation + ni;
                bool valid = is_causal ? (key_i >= 0 && key_i < length && key_i <= qi)
                                       : (key_i >= 0 && key_i < length);
                if (valid && key_i == i) {
                    int da_idx = ((b * heads + h) * l_out + oi) * kernel_size + ki;
                    int q_idx  = ((b * heads + h) * length + qi) * dim + d;
                    grad += d_attn[da_idx] * query[q_idx];
                    break;
                }
            }
        }
        int k_idx = ((b * heads + h) * length + i) * dim + d;
        d_key[k_idx] = grad;
    }
}

// ---------------------------------------------------------------------------
// 1D AV backward: d_attn  –  grid: (L_out, H, B)
// d_attn[oi,ki] = sum_d( d_out[oi,d] * value[ni + ki*dil, d] )
// Supports causal masking and strided output.
// ---------------------------------------------------------------------------
kernel void natten1d_a_backward(
    device const float* d_out    [[buffer(0)]],
    device const float* value    [[buffer(1)]],
    device float*       d_attn   [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       length      [[buffer(5)]],
    constant int&       dim         [[buffer(6)]],
    constant int&       kernel_size [[buffer(7)]],
    constant int&       dilation    [[buffer(8)]],
    constant int&       stride_val  [[buffer(9)]],
    constant int&       l_out       [[buffer(10)]],
    constant int&       is_causal   [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]])
{
    int oi = gid.x;
    int h = gid.y;
    int b = gid.z;
    if (b >= batch_size || h >= heads || oi >= l_out) return;

    int i = oi * stride_val;
    int neighborhood_size = kernel_size / 2;
    int ni = is_causal ? get_causal_window_start(i, kernel_size, dilation)
                       : get_window_start(i, length, kernel_size, neighborhood_size, dilation);

    int do_base = ((b * heads + h) * l_out + oi) * dim;
    for (int ki = 0; ki < kernel_size; ki++) {
        float grad = 0.0f;
        int value_i = ki * dilation + ni;
        bool valid = is_causal ? (value_i >= 0 && value_i < length && value_i <= i)
                               : (value_i >= 0 && value_i < length);
        if (valid) {
            int v_base = ((b * heads + h) * length + value_i) * dim;
            grad = dot_f32(d_out + do_base, value + v_base, dim);
        }
        int da_idx = ((b * heads + h) * l_out + oi) * kernel_size + ki;
        d_attn[da_idx] = grad;
    }
}

// ---------------------------------------------------------------------------
// 1D AV backward: d_value  –  grid: (L, H, B)
// Reverse iteration: find all output positions whose window contains this value
// Supports causal masking and strided output.
// ---------------------------------------------------------------------------
kernel void natten1d_v_backward(
    device const float* d_out    [[buffer(0)]],
    device const float* attn     [[buffer(1)]],
    device float*       d_value  [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       length      [[buffer(5)]],
    constant int&       dim         [[buffer(6)]],
    constant int&       kernel_size [[buffer(7)]],
    constant int&       dilation    [[buffer(8)]],
    constant int&       stride_val  [[buffer(9)]],
    constant int&       l_out       [[buffer(10)]],
    constant int&       is_causal   [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]])
{
    int i = gid.x;
    int h = gid.y;
    int b = gid.z;
    if (b >= batch_size || h >= heads || i >= length) return;

    int neighborhood_size = kernel_size / 2;

    for (int d = 0; d < dim; d++) {
        float grad = 0.0f;
        for (int oi = 0; oi < l_out; oi++) {
            int qi = oi * stride_val;
            int ni = is_causal ? get_causal_window_start(qi, kernel_size, dilation)
                               : get_window_start(qi, length, kernel_size, neighborhood_size, dilation);
            for (int ki = 0; ki < kernel_size; ki++) {
                int value_i = ki * dilation + ni;
                bool valid = is_causal ? (value_i >= 0 && value_i < length && value_i <= qi)
                                       : (value_i >= 0 && value_i < length);
                if (valid && value_i == i) {
                    int a_idx  = ((b * heads + h) * l_out + oi) * kernel_size + ki;
                    int do_idx = ((b * heads + h) * l_out + oi) * dim + d;
                    grad += attn[a_idx] * d_out[do_idx];
                    break;
                }
            }
        }
        int v_idx = ((b * heads + h) * length + i) * dim + d;
        d_value[v_idx] = grad;
    }
}

// ---------------------------------------------------------------------------
// 1D backward (f16 variants)
// ---------------------------------------------------------------------------
kernel void natten1d_q_backward_f16(
    device const half*  d_attn   [[buffer(0)]],
    device const half*  key      [[buffer(1)]],
    device half*        d_query  [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       length      [[buffer(5)]],
    constant int&       dim         [[buffer(6)]],
    constant int&       kernel_size [[buffer(7)]],
    constant int&       dilation    [[buffer(8)]],
    constant int&       stride_val  [[buffer(9)]],
    constant int&       l_out       [[buffer(10)]],
    constant int&       is_causal   [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]])
{
    int i = gid.x;
    int h = gid.y;
    int b = gid.z;
    if (b >= batch_size || h >= heads || i >= length) return;

    int oi = i / stride_val;
    bool query_valid = (i % stride_val == 0) && (oi < l_out);
    if (!query_valid) {
        for (int d = 0; d < dim; d++) {
            d_query[((b * heads + h) * length + i) * dim + d] = half(0.0f);
        }
        return;
    }

    int neighborhood_size = kernel_size / 2;
    int ni = is_causal ? get_causal_window_start(i, kernel_size, dilation)
                       : get_window_start(i, length, kernel_size, neighborhood_size, dilation);

    for (int d = 0; d < dim; d++) {
        float grad = 0.0f;
        for (int ki = 0; ki < kernel_size; ki++) {
            int key_i = ki * dilation + ni;
            bool valid = is_causal ? (key_i >= 0 && key_i < length && key_i <= i)
                                   : (key_i >= 0 && key_i < length);
            if (valid) {
                int da_idx = ((b * heads + h) * l_out + oi) * kernel_size + ki;
                int k_idx  = ((b * heads + h) * length + key_i) * dim + d;
                grad += float(d_attn[da_idx]) * float(key[k_idx]);
            }
        }
        int q_idx = ((b * heads + h) * length + i) * dim + d;
        d_query[q_idx] = half(grad);
    }
}

kernel void natten1d_k_backward_f16(
    device const half*  d_attn   [[buffer(0)]],
    device const half*  query    [[buffer(1)]],
    device half*        d_key    [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       length      [[buffer(5)]],
    constant int&       dim         [[buffer(6)]],
    constant int&       kernel_size [[buffer(7)]],
    constant int&       dilation    [[buffer(8)]],
    constant int&       stride_val  [[buffer(9)]],
    constant int&       l_out       [[buffer(10)]],
    constant int&       is_causal   [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]])
{
    int i = gid.x;
    int h = gid.y;
    int b = gid.z;
    if (b >= batch_size || h >= heads || i >= length) return;

    int neighborhood_size = kernel_size / 2;

    for (int d = 0; d < dim; d++) {
        float grad = 0.0f;
        for (int oi = 0; oi < l_out; oi++) {
            int qi = oi * stride_val;
            int ni = is_causal ? get_causal_window_start(qi, kernel_size, dilation)
                               : get_window_start(qi, length, kernel_size, neighborhood_size, dilation);
            for (int ki = 0; ki < kernel_size; ki++) {
                int key_i = ki * dilation + ni;
                bool valid = is_causal ? (key_i >= 0 && key_i < length && key_i <= qi)
                                       : (key_i >= 0 && key_i < length);
                if (valid && key_i == i) {
                    int da_idx = ((b * heads + h) * l_out + oi) * kernel_size + ki;
                    int q_idx  = ((b * heads + h) * length + qi) * dim + d;
                    grad += float(d_attn[da_idx]) * float(query[q_idx]);
                    break;
                }
            }
        }
        int k_idx = ((b * heads + h) * length + i) * dim + d;
        d_key[k_idx] = half(grad);
    }
}

kernel void natten1d_a_backward_f16(
    device const half*  d_out    [[buffer(0)]],
    device const half*  value    [[buffer(1)]],
    device half*        d_attn   [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       length      [[buffer(5)]],
    constant int&       dim         [[buffer(6)]],
    constant int&       kernel_size [[buffer(7)]],
    constant int&       dilation    [[buffer(8)]],
    constant int&       stride_val  [[buffer(9)]],
    constant int&       l_out       [[buffer(10)]],
    constant int&       is_causal   [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]])
{
    int oi = gid.x;
    int h = gid.y;
    int b = gid.z;
    if (b >= batch_size || h >= heads || oi >= l_out) return;

    int i = oi * stride_val;
    int neighborhood_size = kernel_size / 2;
    int ni = is_causal ? get_causal_window_start(i, kernel_size, dilation)
                       : get_window_start(i, length, kernel_size, neighborhood_size, dilation);

    int do_base = ((b * heads + h) * l_out + oi) * dim;
    for (int ki = 0; ki < kernel_size; ki++) {
        float grad = 0.0f;
        int value_i = ki * dilation + ni;
        bool valid = is_causal ? (value_i >= 0 && value_i < length && value_i <= i)
                               : (value_i >= 0 && value_i < length);
        if (valid) {
            int v_base = ((b * heads + h) * length + value_i) * dim;
            grad = dot_f16(d_out + do_base, value + v_base, dim);
        }
        int da_idx = ((b * heads + h) * l_out + oi) * kernel_size + ki;
        d_attn[da_idx] = half(grad);
    }
}

kernel void natten1d_v_backward_f16(
    device const half*  d_out    [[buffer(0)]],
    device const half*  attn     [[buffer(1)]],
    device half*        d_value  [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       length      [[buffer(5)]],
    constant int&       dim         [[buffer(6)]],
    constant int&       kernel_size [[buffer(7)]],
    constant int&       dilation    [[buffer(8)]],
    constant int&       stride_val  [[buffer(9)]],
    constant int&       l_out       [[buffer(10)]],
    constant int&       is_causal   [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]])
{
    int i = gid.x;
    int h = gid.y;
    int b = gid.z;
    if (b >= batch_size || h >= heads || i >= length) return;

    int neighborhood_size = kernel_size / 2;

    for (int d = 0; d < dim; d++) {
        float grad = 0.0f;
        for (int oi = 0; oi < l_out; oi++) {
            int qi = oi * stride_val;
            int ni = is_causal ? get_causal_window_start(qi, kernel_size, dilation)
                               : get_window_start(qi, length, kernel_size, neighborhood_size, dilation);
            for (int ki = 0; ki < kernel_size; ki++) {
                int value_i = ki * dilation + ni;
                bool valid = is_causal ? (value_i >= 0 && value_i < length && value_i <= qi)
                                       : (value_i >= 0 && value_i < length);
                if (valid && value_i == i) {
                    int a_idx  = ((b * heads + h) * l_out + oi) * kernel_size + ki;
                    int do_idx = ((b * heads + h) * l_out + oi) * dim + d;
                    grad += float(attn[a_idx]) * float(d_out[do_idx]);
                    break;
                }
            }
        }
        int v_idx = ((b * heads + h) * length + i) * dim + d;
        d_value[v_idx] = half(grad);
    }
}

// ---------------------------------------------------------------------------
// 2D QK backward: d_query  –  grid: (W, Hi, B*H)
// ---------------------------------------------------------------------------
kernel void natten2d_q_backward(
    device const float* d_attn   [[buffer(0)]],
    device const float* key      [[buffer(1)]],
    device float*       d_query  [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       height      [[buffer(5)]],
    constant int&       width       [[buffer(6)]],
    constant int&       dim         [[buffer(7)]],
    constant int&       kernel_size_h [[buffer(8)]],
    constant int&       kernel_size_w [[buffer(9)]],
    constant int&       dilation_h    [[buffer(10)]],
    constant int&       dilation_w    [[buffer(11)]],
    constant int&       stride_h      [[buffer(12)]],
    constant int&       stride_w      [[buffer(13)]],
    constant int&       h_out         [[buffer(14)]],
    constant int&       w_out         [[buffer(15)]],
    constant int&       is_causal_h   [[buffer(16)]],
    constant int&       is_causal_w   [[buffer(17)]],
    uint3 gid [[thread_position_in_grid]])
{
    int j = gid.x;
    int i = gid.y;
    int bh = gid.z;
    int b = bh / heads;
    int h = bh % heads;
    if (b >= batch_size || i >= height || j >= width) return;

    int oi = i / stride_h;
    int oj = j / stride_w;
    bool query_valid = (i % stride_h == 0) && (j % stride_w == 0) && (oi < h_out) && (oj < w_out);
    if (!query_valid) {
        for (int d = 0; d < dim; d++) {
            d_query[(((b * heads + h) * height + i) * width + j) * dim + d] = 0.0f;
        }
        return;
    }

    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int ni = is_causal_h ? get_causal_window_start(i, kernel_size_h, dilation_h)
                         : get_window_start(i, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = is_causal_w ? get_causal_window_start(j, kernel_size_w, dilation_w)
                         : get_window_start(j, width, kernel_size_w, nh_size_w, dilation_w);
    int kk = kernel_size_h * kernel_size_w;

    for (int d = 0; d < dim; d++) {
        float grad = 0.0f;
        for (int ki = 0; ki < kernel_size_h; ki++) {
            int key_i = ki * dilation_h + ni;
            bool valid_i = is_causal_h ? (key_i >= 0 && key_i < height && key_i <= i)
                                       : (key_i >= 0 && key_i < height);
            if (!valid_i) continue;
            for (int kj = 0; kj < kernel_size_w; kj++) {
                int key_j = kj * dilation_w + nj;
                bool valid_j = is_causal_w ? (key_j >= 0 && key_j < width && key_j <= j)
                                           : (key_j >= 0 && key_j < width);
                if (valid_j) {
                    int da_idx = (((b * heads + h) * h_out + oi) * w_out + oj) * kk + ki * kernel_size_w + kj;
                    int k_idx  = (((b * heads + h) * height + key_i) * width + key_j) * dim + d;
                    grad += d_attn[da_idx] * key[k_idx];
                }
            }
        }
        int q_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
        d_query[q_idx] = grad;
    }
}

// ---------------------------------------------------------------------------
// 2D QK backward: d_key  –  grid: (W, Hi, B*H)
// ---------------------------------------------------------------------------
kernel void natten2d_k_backward(
    device const float* d_attn   [[buffer(0)]],
    device const float* query    [[buffer(1)]],
    device float*       d_key    [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       height      [[buffer(5)]],
    constant int&       width       [[buffer(6)]],
    constant int&       dim         [[buffer(7)]],
    constant int&       kernel_size_h [[buffer(8)]],
    constant int&       kernel_size_w [[buffer(9)]],
    constant int&       dilation_h    [[buffer(10)]],
    constant int&       dilation_w    [[buffer(11)]],
    constant int&       stride_h      [[buffer(12)]],
    constant int&       stride_w      [[buffer(13)]],
    constant int&       h_out         [[buffer(14)]],
    constant int&       w_out         [[buffer(15)]],
    constant int&       is_causal_h   [[buffer(16)]],
    constant int&       is_causal_w   [[buffer(17)]],
    uint3 gid [[thread_position_in_grid]])
{
    int j = gid.x;
    int i = gid.y;
    int bh = gid.z;
    int b = bh / heads;
    int h = bh % heads;
    if (b >= batch_size || i >= height || j >= width) return;

    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int kk = kernel_size_h * kernel_size_w;

    for (int d = 0; d < dim; d++) {
        float grad = 0.0f;
        for (int oi = 0; oi < h_out; oi++) {
            int qi = oi * stride_h;
            int ni = is_causal_h ? get_causal_window_start(qi, kernel_size_h, dilation_h)
                                 : get_window_start(qi, height, kernel_size_h, nh_size_h, dilation_h);
            for (int oj = 0; oj < w_out; oj++) {
                int qj = oj * stride_w;
                int nj = is_causal_w ? get_causal_window_start(qj, kernel_size_w, dilation_w)
                                     : get_window_start(qj, width, kernel_size_w, nh_size_w, dilation_w);
                for (int ki = 0; ki < kernel_size_h; ki++) {
                    int key_i = ki * dilation_h + ni;
                    bool valid_i = is_causal_h ? (key_i >= 0 && key_i < height && key_i <= qi)
                                               : (key_i >= 0 && key_i < height);
                    if (!valid_i || key_i != i) continue;
                    for (int kj = 0; kj < kernel_size_w; kj++) {
                        int key_j = kj * dilation_w + nj;
                        bool valid_j = is_causal_w ? (key_j >= 0 && key_j < width && key_j <= qj)
                                                   : (key_j >= 0 && key_j < width);
                        if (valid_j && key_j == j) {
                            int da_idx = (((b * heads + h) * h_out + oi) * w_out + oj) * kk + ki * kernel_size_w + kj;
                            int q_idx  = (((b * heads + h) * height + qi) * width + qj) * dim + d;
                            grad += d_attn[da_idx] * query[q_idx];
                        }
                    }
                }
            }
        }
        int k_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
        d_key[k_idx] = grad;
    }
}

// ---------------------------------------------------------------------------
// 2D AV backward: d_attn  –  grid: (W, Hi, B*H)
// ---------------------------------------------------------------------------
kernel void natten2d_a_backward(
    device const float* d_out    [[buffer(0)]],
    device const float* value    [[buffer(1)]],
    device float*       d_attn   [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       height      [[buffer(5)]],
    constant int&       width       [[buffer(6)]],
    constant int&       dim         [[buffer(7)]],
    constant int&       kernel_size_h [[buffer(8)]],
    constant int&       kernel_size_w [[buffer(9)]],
    constant int&       dilation_h    [[buffer(10)]],
    constant int&       dilation_w    [[buffer(11)]],
    constant int&       stride_h      [[buffer(12)]],
    constant int&       stride_w      [[buffer(13)]],
    constant int&       h_out         [[buffer(14)]],
    constant int&       w_out         [[buffer(15)]],
    constant int&       is_causal_h   [[buffer(16)]],
    constant int&       is_causal_w   [[buffer(17)]],
    uint3 gid [[thread_position_in_grid]])
{
    int oj = gid.x;
    int oi = gid.y;
    int bh = gid.z;
    int b = bh / heads;
    int h = bh % heads;
    if (b >= batch_size || oi >= h_out || oj >= w_out) return;

    int i = oi * stride_h;
    int j = oj * stride_w;
    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int ni = is_causal_h ? get_causal_window_start(i, kernel_size_h, dilation_h)
                         : get_window_start(i, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = is_causal_w ? get_causal_window_start(j, kernel_size_w, dilation_w)
                         : get_window_start(j, width, kernel_size_w, nh_size_w, dilation_w);

    int do_base = (((b * heads + h) * h_out + oi) * w_out + oj) * dim;
    for (int ki = 0; ki < kernel_size_h; ki++) {
        int val_i = ki * dilation_h + ni;
        bool valid_i = is_causal_h ? (val_i >= 0 && val_i < height && val_i <= i)
                                   : (val_i >= 0 && val_i < height);
        for (int kj = 0; kj < kernel_size_w; kj++) {
            int val_j = kj * dilation_w + nj;
            bool valid_j = is_causal_w ? (val_j >= 0 && val_j < width && val_j <= j)
                                       : (val_j >= 0 && val_j < width);
            float grad = 0.0f;
            if (valid_i && valid_j) {
                int v_base = (((b * heads + h) * height + val_i) * width + val_j) * dim;
                grad = dot_f32(d_out + do_base, value + v_base, dim);
            }
            int da_idx = (((b * heads + h) * h_out + oi) * w_out + oj) * (kernel_size_h * kernel_size_w) + ki * kernel_size_w + kj;
            d_attn[da_idx] = grad;
        }
    }
}

// ---------------------------------------------------------------------------
// 2D AV backward: d_value  –  grid: (W, Hi, B*H)
// ---------------------------------------------------------------------------
kernel void natten2d_v_backward(
    device const float* d_out    [[buffer(0)]],
    device const float* attn     [[buffer(1)]],
    device float*       d_value  [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       height      [[buffer(5)]],
    constant int&       width       [[buffer(6)]],
    constant int&       dim         [[buffer(7)]],
    constant int&       kernel_size_h [[buffer(8)]],
    constant int&       kernel_size_w [[buffer(9)]],
    constant int&       dilation_h    [[buffer(10)]],
    constant int&       dilation_w    [[buffer(11)]],
    constant int&       stride_h      [[buffer(12)]],
    constant int&       stride_w      [[buffer(13)]],
    constant int&       h_out         [[buffer(14)]],
    constant int&       w_out         [[buffer(15)]],
    constant int&       is_causal_h   [[buffer(16)]],
    constant int&       is_causal_w   [[buffer(17)]],
    uint3 gid [[thread_position_in_grid]])
{
    int j = gid.x;
    int i = gid.y;
    int bh = gid.z;
    int b = bh / heads;
    int h = bh % heads;
    if (b >= batch_size || i >= height || j >= width) return;

    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int kk = kernel_size_h * kernel_size_w;

    for (int d = 0; d < dim; d++) {
        float grad = 0.0f;
        for (int oi = 0; oi < h_out; oi++) {
            int qi = oi * stride_h;
            int ni = is_causal_h ? get_causal_window_start(qi, kernel_size_h, dilation_h)
                                 : get_window_start(qi, height, kernel_size_h, nh_size_h, dilation_h);
            for (int oj = 0; oj < w_out; oj++) {
                int qj = oj * stride_w;
                int nj = is_causal_w ? get_causal_window_start(qj, kernel_size_w, dilation_w)
                                     : get_window_start(qj, width, kernel_size_w, nh_size_w, dilation_w);
                for (int ki = 0; ki < kernel_size_h; ki++) {
                    int val_i = ki * dilation_h + ni;
                    bool valid_i = is_causal_h ? (val_i >= 0 && val_i < height && val_i <= qi)
                                               : (val_i >= 0 && val_i < height);
                    if (!valid_i || val_i != i) continue;
                    for (int kj = 0; kj < kernel_size_w; kj++) {
                        int val_j = kj * dilation_w + nj;
                        bool valid_j = is_causal_w ? (val_j >= 0 && val_j < width && val_j <= qj)
                                                   : (val_j >= 0 && val_j < width);
                        if (valid_j && val_j == j) {
                            int a_idx  = (((b * heads + h) * h_out + oi) * w_out + oj) * kk + ki * kernel_size_w + kj;
                            int do_idx = (((b * heads + h) * h_out + oi) * w_out + oj) * dim + d;
                            grad += attn[a_idx] * d_out[do_idx];
                        }
                    }
                }
            }
        }
        int v_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
        d_value[v_idx] = grad;
    }
}

// ---------------------------------------------------------------------------
// 2D backward (f16 variants)
// ---------------------------------------------------------------------------
kernel void natten2d_q_backward_f16(
    device const half*  d_attn   [[buffer(0)]],
    device const half*  key      [[buffer(1)]],
    device half*        d_query  [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       height      [[buffer(5)]],
    constant int&       width       [[buffer(6)]],
    constant int&       dim         [[buffer(7)]],
    constant int&       kernel_size_h [[buffer(8)]],
    constant int&       kernel_size_w [[buffer(9)]],
    constant int&       dilation_h    [[buffer(10)]],
    constant int&       dilation_w    [[buffer(11)]],
    constant int&       stride_h      [[buffer(12)]],
    constant int&       stride_w      [[buffer(13)]],
    constant int&       h_out         [[buffer(14)]],
    constant int&       w_out         [[buffer(15)]],
    constant int&       is_causal_h   [[buffer(16)]],
    constant int&       is_causal_w   [[buffer(17)]],
    uint3 gid [[thread_position_in_grid]])
{
    int j = gid.x;
    int i = gid.y;
    int bh = gid.z;
    int b = bh / heads;
    int h = bh % heads;
    if (b >= batch_size || i >= height || j >= width) return;

    int oi = i / stride_h;
    int oj = j / stride_w;
    bool query_valid = (i % stride_h == 0) && (j % stride_w == 0) && (oi < h_out) && (oj < w_out);
    if (!query_valid) {
        for (int d = 0; d < dim; d++) {
            d_query[(((b * heads + h) * height + i) * width + j) * dim + d] = half(0.0f);
        }
        return;
    }

    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int ni = is_causal_h ? get_causal_window_start(i, kernel_size_h, dilation_h)
                         : get_window_start(i, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = is_causal_w ? get_causal_window_start(j, kernel_size_w, dilation_w)
                         : get_window_start(j, width, kernel_size_w, nh_size_w, dilation_w);
    int kk = kernel_size_h * kernel_size_w;

    for (int d = 0; d < dim; d++) {
        float grad = 0.0f;
        for (int ki = 0; ki < kernel_size_h; ki++) {
            int key_i = ki * dilation_h + ni;
            bool valid_i = is_causal_h ? (key_i >= 0 && key_i < height && key_i <= i)
                                       : (key_i >= 0 && key_i < height);
            if (!valid_i) continue;
            for (int kj = 0; kj < kernel_size_w; kj++) {
                int key_j = kj * dilation_w + nj;
                bool valid_j = is_causal_w ? (key_j >= 0 && key_j < width && key_j <= j)
                                           : (key_j >= 0 && key_j < width);
                if (valid_j) {
                    int da_idx = (((b * heads + h) * h_out + oi) * w_out + oj) * kk + ki * kernel_size_w + kj;
                    int k_idx  = (((b * heads + h) * height + key_i) * width + key_j) * dim + d;
                    grad += float(d_attn[da_idx]) * float(key[k_idx]);
                }
            }
        }
        int q_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
        d_query[q_idx] = half(grad);
    }
}

kernel void natten2d_k_backward_f16(
    device const half*  d_attn   [[buffer(0)]],
    device const half*  query    [[buffer(1)]],
    device half*        d_key    [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       height      [[buffer(5)]],
    constant int&       width       [[buffer(6)]],
    constant int&       dim         [[buffer(7)]],
    constant int&       kernel_size_h [[buffer(8)]],
    constant int&       kernel_size_w [[buffer(9)]],
    constant int&       dilation_h    [[buffer(10)]],
    constant int&       dilation_w    [[buffer(11)]],
    constant int&       stride_h      [[buffer(12)]],
    constant int&       stride_w      [[buffer(13)]],
    constant int&       h_out         [[buffer(14)]],
    constant int&       w_out         [[buffer(15)]],
    constant int&       is_causal_h   [[buffer(16)]],
    constant int&       is_causal_w   [[buffer(17)]],
    uint3 gid [[thread_position_in_grid]])
{
    int j = gid.x;
    int i = gid.y;
    int bh = gid.z;
    int b = bh / heads;
    int h = bh % heads;
    if (b >= batch_size || i >= height || j >= width) return;

    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int kk = kernel_size_h * kernel_size_w;

    for (int d = 0; d < dim; d++) {
        float grad = 0.0f;
        for (int oi = 0; oi < h_out; oi++) {
            int qi = oi * stride_h;
            int ni = is_causal_h ? get_causal_window_start(qi, kernel_size_h, dilation_h)
                                 : get_window_start(qi, height, kernel_size_h, nh_size_h, dilation_h);
            for (int oj = 0; oj < w_out; oj++) {
                int qj = oj * stride_w;
                int nj = is_causal_w ? get_causal_window_start(qj, kernel_size_w, dilation_w)
                                     : get_window_start(qj, width, kernel_size_w, nh_size_w, dilation_w);
                for (int ki = 0; ki < kernel_size_h; ki++) {
                    int key_i = ki * dilation_h + ni;
                    bool valid_i = is_causal_h ? (key_i >= 0 && key_i < height && key_i <= qi)
                                               : (key_i >= 0 && key_i < height);
                    if (!valid_i || key_i != i) continue;
                    for (int kj = 0; kj < kernel_size_w; kj++) {
                        int key_j = kj * dilation_w + nj;
                        bool valid_j = is_causal_w ? (key_j >= 0 && key_j < width && key_j <= qj)
                                                   : (key_j >= 0 && key_j < width);
                        if (valid_j && key_j == j) {
                            int da_idx = (((b * heads + h) * h_out + oi) * w_out + oj) * kk + ki * kernel_size_w + kj;
                            int q_idx  = (((b * heads + h) * height + qi) * width + qj) * dim + d;
                            grad += float(d_attn[da_idx]) * float(query[q_idx]);
                        }
                    }
                }
            }
        }
        int k_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
        d_key[k_idx] = half(grad);
    }
}

kernel void natten2d_a_backward_f16(
    device const half*  d_out    [[buffer(0)]],
    device const half*  value    [[buffer(1)]],
    device half*        d_attn   [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       height      [[buffer(5)]],
    constant int&       width       [[buffer(6)]],
    constant int&       dim         [[buffer(7)]],
    constant int&       kernel_size_h [[buffer(8)]],
    constant int&       kernel_size_w [[buffer(9)]],
    constant int&       dilation_h    [[buffer(10)]],
    constant int&       dilation_w    [[buffer(11)]],
    constant int&       stride_h      [[buffer(12)]],
    constant int&       stride_w      [[buffer(13)]],
    constant int&       h_out         [[buffer(14)]],
    constant int&       w_out         [[buffer(15)]],
    constant int&       is_causal_h   [[buffer(16)]],
    constant int&       is_causal_w   [[buffer(17)]],
    uint3 gid [[thread_position_in_grid]])
{
    int oj = gid.x;
    int oi = gid.y;
    int bh = gid.z;
    int b = bh / heads;
    int h = bh % heads;
    if (b >= batch_size || oi >= h_out || oj >= w_out) return;

    int i = oi * stride_h;
    int j = oj * stride_w;
    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int ni = is_causal_h ? get_causal_window_start(i, kernel_size_h, dilation_h)
                         : get_window_start(i, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = is_causal_w ? get_causal_window_start(j, kernel_size_w, dilation_w)
                         : get_window_start(j, width, kernel_size_w, nh_size_w, dilation_w);

    int do_base = (((b * heads + h) * h_out + oi) * w_out + oj) * dim;
    for (int ki = 0; ki < kernel_size_h; ki++) {
        int val_i = ki * dilation_h + ni;
        bool valid_i = is_causal_h ? (val_i >= 0 && val_i < height && val_i <= i)
                                   : (val_i >= 0 && val_i < height);
        for (int kj = 0; kj < kernel_size_w; kj++) {
            int val_j = kj * dilation_w + nj;
            bool valid_j = is_causal_w ? (val_j >= 0 && val_j < width && val_j <= j)
                                       : (val_j >= 0 && val_j < width);
            float grad = 0.0f;
            if (valid_i && valid_j) {
                int v_base = (((b * heads + h) * height + val_i) * width + val_j) * dim;
                grad = dot_f16(d_out + do_base, value + v_base, dim);
            }
            int da_idx = (((b * heads + h) * h_out + oi) * w_out + oj) * (kernel_size_h * kernel_size_w) + ki * kernel_size_w + kj;
            d_attn[da_idx] = half(grad);
        }
    }
}

kernel void natten2d_v_backward_f16(
    device const half*  d_out    [[buffer(0)]],
    device const half*  attn     [[buffer(1)]],
    device half*        d_value  [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       height      [[buffer(5)]],
    constant int&       width       [[buffer(6)]],
    constant int&       dim         [[buffer(7)]],
    constant int&       kernel_size_h [[buffer(8)]],
    constant int&       kernel_size_w [[buffer(9)]],
    constant int&       dilation_h    [[buffer(10)]],
    constant int&       dilation_w    [[buffer(11)]],
    constant int&       stride_h      [[buffer(12)]],
    constant int&       stride_w      [[buffer(13)]],
    constant int&       h_out         [[buffer(14)]],
    constant int&       w_out         [[buffer(15)]],
    constant int&       is_causal_h   [[buffer(16)]],
    constant int&       is_causal_w   [[buffer(17)]],
    uint3 gid [[thread_position_in_grid]])
{
    int j = gid.x;
    int i = gid.y;
    int bh = gid.z;
    int b = bh / heads;
    int h = bh % heads;
    if (b >= batch_size || i >= height || j >= width) return;

    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int kk = kernel_size_h * kernel_size_w;

    for (int d = 0; d < dim; d++) {
        float grad = 0.0f;
        for (int oi = 0; oi < h_out; oi++) {
            int qi = oi * stride_h;
            int ni = is_causal_h ? get_causal_window_start(qi, kernel_size_h, dilation_h)
                                 : get_window_start(qi, height, kernel_size_h, nh_size_h, dilation_h);
            for (int oj = 0; oj < w_out; oj++) {
                int qj = oj * stride_w;
                int nj = is_causal_w ? get_causal_window_start(qj, kernel_size_w, dilation_w)
                                     : get_window_start(qj, width, kernel_size_w, nh_size_w, dilation_w);
                for (int ki = 0; ki < kernel_size_h; ki++) {
                    int val_i = ki * dilation_h + ni;
                    bool valid_i = is_causal_h ? (val_i >= 0 && val_i < height && val_i <= qi)
                                               : (val_i >= 0 && val_i < height);
                    if (!valid_i || val_i != i) continue;
                    for (int kj = 0; kj < kernel_size_w; kj++) {
                        int val_j = kj * dilation_w + nj;
                        bool valid_j = is_causal_w ? (val_j >= 0 && val_j < width && val_j <= qj)
                                                   : (val_j >= 0 && val_j < width);
                        if (valid_j && val_j == j) {
                            int a_idx  = (((b * heads + h) * h_out + oi) * w_out + oj) * kk + ki * kernel_size_w + kj;
                            int do_idx = (((b * heads + h) * h_out + oi) * w_out + oj) * dim + d;
                            grad += float(attn[a_idx]) * float(d_out[do_idx]);
                        }
                    }
                }
            }
        }
        int v_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
        d_value[v_idx] = half(grad);
    }
}

// ---------------------------------------------------------------------------
// 3D QK backward: d_query  –  grid: (W, H, B*heads*Dp)
// ---------------------------------------------------------------------------
kernel void natten3d_q_backward(
    device const float* d_attn   [[buffer(0)]],
    device const float* key      [[buffer(1)]],
    device float*       d_query  [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       depth       [[buffer(5)]],
    constant int&       height      [[buffer(6)]],
    constant int&       width       [[buffer(7)]],
    constant int&       dim         [[buffer(8)]],
    constant int&       kernel_size_d [[buffer(9)]],
    constant int&       kernel_size_h [[buffer(10)]],
    constant int&       kernel_size_w [[buffer(11)]],
    constant int&       dilation_d    [[buffer(12)]],
    constant int&       dilation_h    [[buffer(13)]],
    constant int&       dilation_w    [[buffer(14)]],
    constant int&       stride_d      [[buffer(15)]],
    constant int&       stride_h      [[buffer(16)]],
    constant int&       stride_w      [[buffer(17)]],
    constant int&       d_out         [[buffer(18)]],
    constant int&       h_out         [[buffer(19)]],
    constant int&       w_out         [[buffer(20)]],
    constant int&       is_causal_d   [[buffer(21)]],
    constant int&       is_causal_h   [[buffer(22)]],
    constant int&       is_causal_w   [[buffer(23)]],
    uint3 gid [[thread_position_in_grid]])
{
    int j = gid.x;
    int i = gid.y;
    int bhd = gid.z;
    int b  = bhd / (heads * depth);
    int rem = bhd % (heads * depth);
    int h  = rem / depth;
    int dp = rem % depth;
    if (b >= batch_size || i >= height || j >= width || dp >= depth) return;

    int od = dp / stride_d;
    int oi = i / stride_h;
    int oj = j / stride_w;
    bool query_valid = (dp % stride_d == 0) && (i % stride_h == 0) && (j % stride_w == 0)
                       && (od < d_out) && (oi < h_out) && (oj < w_out);
    if (!query_valid) {
        for (int d = 0; d < dim; d++) {
            d_query[((((b*heads+h)*depth+dp)*height+i)*width+j)*dim+d] = 0.0f;
        }
        return;
    }

    int nh_size_d = kernel_size_d / 2;
    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int nd = is_causal_d ? get_causal_window_start(dp, kernel_size_d, dilation_d)
                         : get_window_start(dp, depth, kernel_size_d, nh_size_d, dilation_d);
    int ni = is_causal_h ? get_causal_window_start(i, kernel_size_h, dilation_h)
                         : get_window_start(i, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = is_causal_w ? get_causal_window_start(j, kernel_size_w, dilation_w)
                         : get_window_start(j, width, kernel_size_w, nh_size_w, dilation_w);
    int k_vol = kernel_size_d * kernel_size_h * kernel_size_w;

    for (int d = 0; d < dim; d++) {
        float grad = 0.0f;
        for (int kd = 0; kd < kernel_size_d; kd++) {
            int key_d = kd * dilation_d + nd;
            bool vd = is_causal_d ? (key_d >= 0 && key_d < depth && key_d <= dp)
                                  : (key_d >= 0 && key_d < depth);
            if (!vd) continue;
            for (int ki = 0; ki < kernel_size_h; ki++) {
                int key_i = ki * dilation_h + ni;
                bool vh = is_causal_h ? (key_i >= 0 && key_i < height && key_i <= i)
                                      : (key_i >= 0 && key_i < height);
                if (!vh) continue;
                for (int kj = 0; kj < kernel_size_w; kj++) {
                    int key_j = kj * dilation_w + nj;
                    bool vw = is_causal_w ? (key_j >= 0 && key_j < width && key_j <= j)
                                          : (key_j >= 0 && key_j < width);
                    if (vw) {
                        int da_idx = ((((b*heads+h)*d_out+od)*h_out+oi)*w_out+oj)*k_vol
                                     + (kd*kernel_size_h+ki)*kernel_size_w+kj;
                        int k_idx  = ((((b*heads+h)*depth+key_d)*height+key_i)*width+key_j)*dim+d;
                        grad += d_attn[da_idx] * key[k_idx];
                    }
                }
            }
        }
        int q_idx = ((((b*heads+h)*depth+dp)*height+i)*width+j)*dim+d;
        d_query[q_idx] = grad;
    }
}

// ---------------------------------------------------------------------------
// 3D QK backward: d_key  –  grid: (W, H, B*heads*Dp)
// ---------------------------------------------------------------------------
kernel void natten3d_k_backward(
    device const float* d_attn   [[buffer(0)]],
    device const float* query    [[buffer(1)]],
    device float*       d_key    [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       depth       [[buffer(5)]],
    constant int&       height      [[buffer(6)]],
    constant int&       width       [[buffer(7)]],
    constant int&       dim         [[buffer(8)]],
    constant int&       kernel_size_d [[buffer(9)]],
    constant int&       kernel_size_h [[buffer(10)]],
    constant int&       kernel_size_w [[buffer(11)]],
    constant int&       dilation_d    [[buffer(12)]],
    constant int&       dilation_h    [[buffer(13)]],
    constant int&       dilation_w    [[buffer(14)]],
    constant int&       stride_d      [[buffer(15)]],
    constant int&       stride_h      [[buffer(16)]],
    constant int&       stride_w      [[buffer(17)]],
    constant int&       d_out         [[buffer(18)]],
    constant int&       h_out         [[buffer(19)]],
    constant int&       w_out         [[buffer(20)]],
    constant int&       is_causal_d   [[buffer(21)]],
    constant int&       is_causal_h   [[buffer(22)]],
    constant int&       is_causal_w   [[buffer(23)]],
    uint3 gid [[thread_position_in_grid]])
{
    int j = gid.x;
    int i = gid.y;
    int bhd = gid.z;
    int b  = bhd / (heads * depth);
    int rem = bhd % (heads * depth);
    int h  = rem / depth;
    int dp = rem % depth;
    if (b >= batch_size || i >= height || j >= width || dp >= depth) return;

    int nh_size_d = kernel_size_d / 2;
    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int k_vol = kernel_size_d * kernel_size_h * kernel_size_w;

    for (int d = 0; d < dim; d++) {
        float grad = 0.0f;
        for (int od = 0; od < d_out; od++) {
            int qd = od * stride_d;
            int nd = is_causal_d ? get_causal_window_start(qd, kernel_size_d, dilation_d)
                                 : get_window_start(qd, depth, kernel_size_d, nh_size_d, dilation_d);
            for (int ohi = 0; ohi < h_out; ohi++) {
                int qi = ohi * stride_h;
                int ni = is_causal_h ? get_causal_window_start(qi, kernel_size_h, dilation_h)
                                     : get_window_start(qi, height, kernel_size_h, nh_size_h, dilation_h);
                for (int oj = 0; oj < w_out; oj++) {
                    int qj = oj * stride_w;
                    int nj = is_causal_w ? get_causal_window_start(qj, kernel_size_w, dilation_w)
                                         : get_window_start(qj, width, kernel_size_w, nh_size_w, dilation_w);
                    for (int kd = 0; kd < kernel_size_d; kd++) {
                        int key_d = kd * dilation_d + nd;
                        if (key_d != dp) continue;
                        bool vd = is_causal_d ? (key_d >= 0 && key_d < depth && key_d <= qd)
                                              : (key_d >= 0 && key_d < depth);
                        if (!vd) continue;
                        for (int ki = 0; ki < kernel_size_h; ki++) {
                            int key_i = ki * dilation_h + ni;
                            if (key_i != i) continue;
                            bool vh = is_causal_h ? (key_i >= 0 && key_i < height && key_i <= qi)
                                                  : (key_i >= 0 && key_i < height);
                            if (!vh) continue;
                            for (int kj = 0; kj < kernel_size_w; kj++) {
                                int key_j = kj * dilation_w + nj;
                                if (key_j != j) continue;
                                bool vw = is_causal_w ? (key_j >= 0 && key_j < width && key_j <= qj)
                                                      : (key_j >= 0 && key_j < width);
                                if (vw) {
                                    int da_idx = ((((b*heads+h)*d_out+od)*h_out+ohi)*w_out+oj)*k_vol
                                                 + (kd*kernel_size_h+ki)*kernel_size_w+kj;
                                    int q_idx  = ((((b*heads+h)*depth+qd)*height+qi)*width+qj)*dim+d;
                                    grad += d_attn[da_idx] * query[q_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
        int k_idx = ((((b*heads+h)*depth+dp)*height+i)*width+j)*dim+d;
        d_key[k_idx] = grad;
    }
}

// ---------------------------------------------------------------------------
// 3D AV backward: d_attn  –  grid: (W_out, H_out, B*heads*D_out)
// ---------------------------------------------------------------------------
kernel void natten3d_a_backward(
    device const float* d_out    [[buffer(0)]],
    device const float* value    [[buffer(1)]],
    device float*       d_attn   [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       depth       [[buffer(5)]],
    constant int&       height      [[buffer(6)]],
    constant int&       width       [[buffer(7)]],
    constant int&       dim         [[buffer(8)]],
    constant int&       kernel_size_d [[buffer(9)]],
    constant int&       kernel_size_h [[buffer(10)]],
    constant int&       kernel_size_w [[buffer(11)]],
    constant int&       dilation_d    [[buffer(12)]],
    constant int&       dilation_h    [[buffer(13)]],
    constant int&       dilation_w    [[buffer(14)]],
    constant int&       stride_d      [[buffer(15)]],
    constant int&       stride_h      [[buffer(16)]],
    constant int&       stride_w      [[buffer(17)]],
    constant int&       d_out_sz      [[buffer(18)]],
    constant int&       h_out         [[buffer(19)]],
    constant int&       w_out         [[buffer(20)]],
    constant int&       is_causal_d   [[buffer(21)]],
    constant int&       is_causal_h   [[buffer(22)]],
    constant int&       is_causal_w   [[buffer(23)]],
    uint3 gid [[thread_position_in_grid]])
{
    int oj = gid.x;
    int oi = gid.y;
    int bho = gid.z;
    int b  = bho / (heads * d_out_sz);
    int rem = bho % (heads * d_out_sz);
    int h  = rem / d_out_sz;
    int od = rem % d_out_sz;
    if (b >= batch_size || oi >= h_out || oj >= w_out || od >= d_out_sz) return;

    int dp = od * stride_d;
    int ii = oi * stride_h;
    int jj = oj * stride_w;

    int nh_size_d = kernel_size_d / 2;
    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int nd = is_causal_d ? get_causal_window_start(dp, kernel_size_d, dilation_d)
                         : get_window_start(dp, depth, kernel_size_d, nh_size_d, dilation_d);
    int ni = is_causal_h ? get_causal_window_start(ii, kernel_size_h, dilation_h)
                         : get_window_start(ii, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = is_causal_w ? get_causal_window_start(jj, kernel_size_w, dilation_w)
                         : get_window_start(jj, width, kernel_size_w, nh_size_w, dilation_w);
    int k_vol = kernel_size_d * kernel_size_h * kernel_size_w;

    int do_base = ((((b*heads+h)*d_out_sz+od)*h_out+oi)*w_out+oj)*dim;
    for (int kd = 0; kd < kernel_size_d; kd++) {
        int val_d = kd * dilation_d + nd;
        bool vd = is_causal_d ? (val_d >= 0 && val_d < depth && val_d <= dp)
                              : (val_d >= 0 && val_d < depth);
        for (int ki = 0; ki < kernel_size_h; ki++) {
            int val_i = ki * dilation_h + ni;
            bool vh = is_causal_h ? (val_i >= 0 && val_i < height && val_i <= ii)
                                  : (val_i >= 0 && val_i < height);
            for (int kj = 0; kj < kernel_size_w; kj++) {
                int val_j = kj * dilation_w + nj;
                bool vw = is_causal_w ? (val_j >= 0 && val_j < width && val_j <= jj)
                                      : (val_j >= 0 && val_j < width);
                float grad = 0.0f;
                if (vd && vh && vw) {
                    int v_base = ((((b*heads+h)*depth+val_d)*height+val_i)*width+val_j)*dim;
                    grad = dot_f32(d_out + do_base, value + v_base, dim);
                }
                int da_idx = ((((b*heads+h)*d_out_sz+od)*h_out+oi)*w_out+oj)*k_vol
                             + (kd*kernel_size_h+ki)*kernel_size_w+kj;
                d_attn[da_idx] = grad;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3D AV backward: d_value  –  grid: (W, H, B*heads*Dp)
// ---------------------------------------------------------------------------
kernel void natten3d_v_backward(
    device const float* d_out    [[buffer(0)]],
    device const float* attn     [[buffer(1)]],
    device float*       d_value  [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       depth       [[buffer(5)]],
    constant int&       height      [[buffer(6)]],
    constant int&       width       [[buffer(7)]],
    constant int&       dim         [[buffer(8)]],
    constant int&       kernel_size_d [[buffer(9)]],
    constant int&       kernel_size_h [[buffer(10)]],
    constant int&       kernel_size_w [[buffer(11)]],
    constant int&       dilation_d    [[buffer(12)]],
    constant int&       dilation_h    [[buffer(13)]],
    constant int&       dilation_w    [[buffer(14)]],
    constant int&       stride_d      [[buffer(15)]],
    constant int&       stride_h      [[buffer(16)]],
    constant int&       stride_w      [[buffer(17)]],
    constant int&       d_out_sz      [[buffer(18)]],
    constant int&       h_out         [[buffer(19)]],
    constant int&       w_out         [[buffer(20)]],
    constant int&       is_causal_d   [[buffer(21)]],
    constant int&       is_causal_h   [[buffer(22)]],
    constant int&       is_causal_w   [[buffer(23)]],
    uint3 gid [[thread_position_in_grid]])
{
    int j = gid.x;
    int i = gid.y;
    int bhd = gid.z;
    int b  = bhd / (heads * depth);
    int rem = bhd % (heads * depth);
    int h  = rem / depth;
    int dp = rem % depth;
    if (b >= batch_size || i >= height || j >= width || dp >= depth) return;

    int nh_size_d = kernel_size_d / 2;
    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int k_vol = kernel_size_d * kernel_size_h * kernel_size_w;

    for (int d = 0; d < dim; d++) {
        float grad = 0.0f;
        for (int od = 0; od < d_out_sz; od++) {
            int qd = od * stride_d;
            int nd = is_causal_d ? get_causal_window_start(qd, kernel_size_d, dilation_d)
                                 : get_window_start(qd, depth, kernel_size_d, nh_size_d, dilation_d);
            for (int ohi = 0; ohi < h_out; ohi++) {
                int qi = ohi * stride_h;
                int ni = is_causal_h ? get_causal_window_start(qi, kernel_size_h, dilation_h)
                                     : get_window_start(qi, height, kernel_size_h, nh_size_h, dilation_h);
                for (int oj = 0; oj < w_out; oj++) {
                    int qj = oj * stride_w;
                    int nj = is_causal_w ? get_causal_window_start(qj, kernel_size_w, dilation_w)
                                         : get_window_start(qj, width, kernel_size_w, nh_size_w, dilation_w);
                    for (int kd = 0; kd < kernel_size_d; kd++) {
                        int val_d = kd * dilation_d + nd;
                        if (val_d != dp) continue;
                        bool vd = is_causal_d ? (val_d >= 0 && val_d < depth && val_d <= qd)
                                              : (val_d >= 0 && val_d < depth);
                        if (!vd) continue;
                        for (int ki = 0; ki < kernel_size_h; ki++) {
                            int val_i = ki * dilation_h + ni;
                            if (val_i != i) continue;
                            bool vh = is_causal_h ? (val_i >= 0 && val_i < height && val_i <= qi)
                                                  : (val_i >= 0 && val_i < height);
                            if (!vh) continue;
                            for (int kj = 0; kj < kernel_size_w; kj++) {
                                int val_j = kj * dilation_w + nj;
                                if (val_j != j) continue;
                                bool vw = is_causal_w ? (val_j >= 0 && val_j < width && val_j <= qj)
                                                      : (val_j >= 0 && val_j < width);
                                if (vw) {
                                    int a_idx  = ((((b*heads+h)*d_out_sz+od)*h_out+ohi)*w_out+oj)*k_vol
                                                 + (kd*kernel_size_h+ki)*kernel_size_w+kj;
                                    int do_idx = ((((b*heads+h)*d_out_sz+od)*h_out+ohi)*w_out+oj)*dim+d;
                                    grad += attn[a_idx] * d_out[do_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
        int v_idx = ((((b*heads+h)*depth+dp)*height+i)*width+j)*dim+d;
        d_value[v_idx] = grad;
    }
}

// ---------------------------------------------------------------------------
// 3D backward (f16 variants)
// ---------------------------------------------------------------------------
kernel void natten3d_q_backward_f16(
    device const half*  d_attn   [[buffer(0)]],
    device const half*  key      [[buffer(1)]],
    device half*        d_query  [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       depth       [[buffer(5)]],
    constant int&       height      [[buffer(6)]],
    constant int&       width       [[buffer(7)]],
    constant int&       dim         [[buffer(8)]],
    constant int&       kernel_size_d [[buffer(9)]],
    constant int&       kernel_size_h [[buffer(10)]],
    constant int&       kernel_size_w [[buffer(11)]],
    constant int&       dilation_d    [[buffer(12)]],
    constant int&       dilation_h    [[buffer(13)]],
    constant int&       dilation_w    [[buffer(14)]],
    constant int&       stride_d      [[buffer(15)]],
    constant int&       stride_h      [[buffer(16)]],
    constant int&       stride_w      [[buffer(17)]],
    constant int&       d_out         [[buffer(18)]],
    constant int&       h_out         [[buffer(19)]],
    constant int&       w_out         [[buffer(20)]],
    constant int&       is_causal_d   [[buffer(21)]],
    constant int&       is_causal_h   [[buffer(22)]],
    constant int&       is_causal_w   [[buffer(23)]],
    uint3 gid [[thread_position_in_grid]])
{
    int j = gid.x;
    int i = gid.y;
    int bhd = gid.z;
    int b  = bhd / (heads * depth);
    int rem = bhd % (heads * depth);
    int h  = rem / depth;
    int dp = rem % depth;
    if (b >= batch_size || i >= height || j >= width || dp >= depth) return;

    int od = dp / stride_d;
    int oi = i / stride_h;
    int oj = j / stride_w;
    bool query_valid = (dp % stride_d == 0) && (i % stride_h == 0) && (j % stride_w == 0)
                       && (od < d_out) && (oi < h_out) && (oj < w_out);
    if (!query_valid) {
        for (int d = 0; d < dim; d++) {
            d_query[((((b*heads+h)*depth+dp)*height+i)*width+j)*dim+d] = half(0.0f);
        }
        return;
    }

    int nh_size_d = kernel_size_d / 2;
    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int nd = is_causal_d ? get_causal_window_start(dp, kernel_size_d, dilation_d)
                         : get_window_start(dp, depth, kernel_size_d, nh_size_d, dilation_d);
    int ni = is_causal_h ? get_causal_window_start(i, kernel_size_h, dilation_h)
                         : get_window_start(i, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = is_causal_w ? get_causal_window_start(j, kernel_size_w, dilation_w)
                         : get_window_start(j, width, kernel_size_w, nh_size_w, dilation_w);
    int k_vol = kernel_size_d * kernel_size_h * kernel_size_w;

    for (int d = 0; d < dim; d++) {
        float grad = 0.0f;
        for (int kd = 0; kd < kernel_size_d; kd++) {
            int key_d = kd * dilation_d + nd;
            bool vd = is_causal_d ? (key_d >= 0 && key_d < depth && key_d <= dp)
                                  : (key_d >= 0 && key_d < depth);
            if (!vd) continue;
            for (int ki = 0; ki < kernel_size_h; ki++) {
                int key_i = ki * dilation_h + ni;
                bool vh = is_causal_h ? (key_i >= 0 && key_i < height && key_i <= i)
                                      : (key_i >= 0 && key_i < height);
                if (!vh) continue;
                for (int kj = 0; kj < kernel_size_w; kj++) {
                    int key_j = kj * dilation_w + nj;
                    bool vw = is_causal_w ? (key_j >= 0 && key_j < width && key_j <= j)
                                          : (key_j >= 0 && key_j < width);
                    if (vw) {
                        int da_idx = ((((b*heads+h)*d_out+od)*h_out+oi)*w_out+oj)*k_vol
                                     + (kd*kernel_size_h+ki)*kernel_size_w+kj;
                        int k_idx  = ((((b*heads+h)*depth+key_d)*height+key_i)*width+key_j)*dim+d;
                        grad += float(d_attn[da_idx]) * float(key[k_idx]);
                    }
                }
            }
        }
        int q_idx = ((((b*heads+h)*depth+dp)*height+i)*width+j)*dim+d;
        d_query[q_idx] = half(grad);
    }
}

kernel void natten3d_k_backward_f16(
    device const half*  d_attn   [[buffer(0)]],
    device const half*  query    [[buffer(1)]],
    device half*        d_key    [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       depth       [[buffer(5)]],
    constant int&       height      [[buffer(6)]],
    constant int&       width       [[buffer(7)]],
    constant int&       dim         [[buffer(8)]],
    constant int&       kernel_size_d [[buffer(9)]],
    constant int&       kernel_size_h [[buffer(10)]],
    constant int&       kernel_size_w [[buffer(11)]],
    constant int&       dilation_d    [[buffer(12)]],
    constant int&       dilation_h    [[buffer(13)]],
    constant int&       dilation_w    [[buffer(14)]],
    constant int&       stride_d      [[buffer(15)]],
    constant int&       stride_h      [[buffer(16)]],
    constant int&       stride_w      [[buffer(17)]],
    constant int&       d_out         [[buffer(18)]],
    constant int&       h_out         [[buffer(19)]],
    constant int&       w_out         [[buffer(20)]],
    constant int&       is_causal_d   [[buffer(21)]],
    constant int&       is_causal_h   [[buffer(22)]],
    constant int&       is_causal_w   [[buffer(23)]],
    uint3 gid [[thread_position_in_grid]])
{
    int j = gid.x;
    int i = gid.y;
    int bhd = gid.z;
    int b  = bhd / (heads * depth);
    int rem = bhd % (heads * depth);
    int h  = rem / depth;
    int dp = rem % depth;
    if (b >= batch_size || i >= height || j >= width || dp >= depth) return;

    int nh_size_d = kernel_size_d / 2;
    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int k_vol = kernel_size_d * kernel_size_h * kernel_size_w;

    for (int d = 0; d < dim; d++) {
        float grad = 0.0f;
        for (int od = 0; od < d_out; od++) {
            int qd = od * stride_d;
            int nd = is_causal_d ? get_causal_window_start(qd, kernel_size_d, dilation_d)
                                 : get_window_start(qd, depth, kernel_size_d, nh_size_d, dilation_d);
            for (int ohi = 0; ohi < h_out; ohi++) {
                int qi = ohi * stride_h;
                int ni = is_causal_h ? get_causal_window_start(qi, kernel_size_h, dilation_h)
                                     : get_window_start(qi, height, kernel_size_h, nh_size_h, dilation_h);
                for (int oj = 0; oj < w_out; oj++) {
                    int qj = oj * stride_w;
                    int nj = is_causal_w ? get_causal_window_start(qj, kernel_size_w, dilation_w)
                                         : get_window_start(qj, width, kernel_size_w, nh_size_w, dilation_w);
                    for (int kd = 0; kd < kernel_size_d; kd++) {
                        int key_d = kd * dilation_d + nd;
                        if (key_d != dp) continue;
                        bool vd = is_causal_d ? (key_d >= 0 && key_d < depth && key_d <= qd)
                                              : (key_d >= 0 && key_d < depth);
                        if (!vd) continue;
                        for (int ki = 0; ki < kernel_size_h; ki++) {
                            int key_i = ki * dilation_h + ni;
                            if (key_i != i) continue;
                            bool vh = is_causal_h ? (key_i >= 0 && key_i < height && key_i <= qi)
                                                  : (key_i >= 0 && key_i < height);
                            if (!vh) continue;
                            for (int kj = 0; kj < kernel_size_w; kj++) {
                                int key_j = kj * dilation_w + nj;
                                if (key_j != j) continue;
                                bool vw = is_causal_w ? (key_j >= 0 && key_j < width && key_j <= qj)
                                                      : (key_j >= 0 && key_j < width);
                                if (vw) {
                                    int da_idx = ((((b*heads+h)*d_out+od)*h_out+ohi)*w_out+oj)*k_vol
                                                 + (kd*kernel_size_h+ki)*kernel_size_w+kj;
                                    int q_idx  = ((((b*heads+h)*depth+qd)*height+qi)*width+qj)*dim+d;
                                    grad += float(d_attn[da_idx]) * float(query[q_idx]);
                                }
                            }
                        }
                    }
                }
            }
        }
        int k_idx = ((((b*heads+h)*depth+dp)*height+i)*width+j)*dim+d;
        d_key[k_idx] = half(grad);
    }
}

kernel void natten3d_a_backward_f16(
    device const half*  d_out    [[buffer(0)]],
    device const half*  value    [[buffer(1)]],
    device half*        d_attn   [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       depth       [[buffer(5)]],
    constant int&       height      [[buffer(6)]],
    constant int&       width       [[buffer(7)]],
    constant int&       dim         [[buffer(8)]],
    constant int&       kernel_size_d [[buffer(9)]],
    constant int&       kernel_size_h [[buffer(10)]],
    constant int&       kernel_size_w [[buffer(11)]],
    constant int&       dilation_d    [[buffer(12)]],
    constant int&       dilation_h    [[buffer(13)]],
    constant int&       dilation_w    [[buffer(14)]],
    constant int&       stride_d      [[buffer(15)]],
    constant int&       stride_h      [[buffer(16)]],
    constant int&       stride_w      [[buffer(17)]],
    constant int&       d_out_sz      [[buffer(18)]],
    constant int&       h_out         [[buffer(19)]],
    constant int&       w_out         [[buffer(20)]],
    constant int&       is_causal_d   [[buffer(21)]],
    constant int&       is_causal_h   [[buffer(22)]],
    constant int&       is_causal_w   [[buffer(23)]],
    uint3 gid [[thread_position_in_grid]])
{
    int oj = gid.x;
    int oi = gid.y;
    int bho = gid.z;
    int b  = bho / (heads * d_out_sz);
    int rem = bho % (heads * d_out_sz);
    int h  = rem / d_out_sz;
    int od = rem % d_out_sz;
    if (b >= batch_size || oi >= h_out || oj >= w_out || od >= d_out_sz) return;

    int dp = od * stride_d;
    int ii = oi * stride_h;
    int jj = oj * stride_w;

    int nh_size_d = kernel_size_d / 2;
    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int nd = is_causal_d ? get_causal_window_start(dp, kernel_size_d, dilation_d)
                         : get_window_start(dp, depth, kernel_size_d, nh_size_d, dilation_d);
    int ni = is_causal_h ? get_causal_window_start(ii, kernel_size_h, dilation_h)
                         : get_window_start(ii, height, kernel_size_h, nh_size_h, dilation_h);
    int nj = is_causal_w ? get_causal_window_start(jj, kernel_size_w, dilation_w)
                         : get_window_start(jj, width, kernel_size_w, nh_size_w, dilation_w);
    int k_vol = kernel_size_d * kernel_size_h * kernel_size_w;

    int do_base = ((((b*heads+h)*d_out_sz+od)*h_out+oi)*w_out+oj)*dim;
    for (int kd = 0; kd < kernel_size_d; kd++) {
        int val_d = kd * dilation_d + nd;
        bool vd = is_causal_d ? (val_d >= 0 && val_d < depth && val_d <= dp)
                              : (val_d >= 0 && val_d < depth);
        for (int ki = 0; ki < kernel_size_h; ki++) {
            int val_i = ki * dilation_h + ni;
            bool vh = is_causal_h ? (val_i >= 0 && val_i < height && val_i <= ii)
                                  : (val_i >= 0 && val_i < height);
            for (int kj = 0; kj < kernel_size_w; kj++) {
                int val_j = kj * dilation_w + nj;
                bool vw = is_causal_w ? (val_j >= 0 && val_j < width && val_j <= jj)
                                      : (val_j >= 0 && val_j < width);
                float grad = 0.0f;
                if (vd && vh && vw) {
                    int v_base = ((((b*heads+h)*depth+val_d)*height+val_i)*width+val_j)*dim;
                    grad = dot_f16(d_out + do_base, value + v_base, dim);
                }
                int da_idx = ((((b*heads+h)*d_out_sz+od)*h_out+oi)*w_out+oj)*k_vol
                             + (kd*kernel_size_h+ki)*kernel_size_w+kj;
                d_attn[da_idx] = half(grad);
            }
        }
    }
}

kernel void natten3d_v_backward_f16(
    device const half*  d_out    [[buffer(0)]],
    device const half*  attn     [[buffer(1)]],
    device half*        d_value  [[buffer(2)]],
    constant int&       batch_size  [[buffer(3)]],
    constant int&       heads       [[buffer(4)]],
    constant int&       depth       [[buffer(5)]],
    constant int&       height      [[buffer(6)]],
    constant int&       width       [[buffer(7)]],
    constant int&       dim         [[buffer(8)]],
    constant int&       kernel_size_d [[buffer(9)]],
    constant int&       kernel_size_h [[buffer(10)]],
    constant int&       kernel_size_w [[buffer(11)]],
    constant int&       dilation_d    [[buffer(12)]],
    constant int&       dilation_h    [[buffer(13)]],
    constant int&       dilation_w    [[buffer(14)]],
    constant int&       stride_d      [[buffer(15)]],
    constant int&       stride_h      [[buffer(16)]],
    constant int&       stride_w      [[buffer(17)]],
    constant int&       d_out_sz      [[buffer(18)]],
    constant int&       h_out         [[buffer(19)]],
    constant int&       w_out         [[buffer(20)]],
    constant int&       is_causal_d   [[buffer(21)]],
    constant int&       is_causal_h   [[buffer(22)]],
    constant int&       is_causal_w   [[buffer(23)]],
    uint3 gid [[thread_position_in_grid]])
{
    int j = gid.x;
    int i = gid.y;
    int bhd = gid.z;
    int b  = bhd / (heads * depth);
    int rem = bhd % (heads * depth);
    int h  = rem / depth;
    int dp = rem % depth;
    if (b >= batch_size || i >= height || j >= width || dp >= depth) return;

    int nh_size_d = kernel_size_d / 2;
    int nh_size_h = kernel_size_h / 2;
    int nh_size_w = kernel_size_w / 2;
    int k_vol = kernel_size_d * kernel_size_h * kernel_size_w;

    for (int d = 0; d < dim; d++) {
        float grad = 0.0f;
        for (int od = 0; od < d_out_sz; od++) {
            int qd = od * stride_d;
            int nd = is_causal_d ? get_causal_window_start(qd, kernel_size_d, dilation_d)
                                 : get_window_start(qd, depth, kernel_size_d, nh_size_d, dilation_d);
            for (int ohi = 0; ohi < h_out; ohi++) {
                int qi = ohi * stride_h;
                int ni = is_causal_h ? get_causal_window_start(qi, kernel_size_h, dilation_h)
                                     : get_window_start(qi, height, kernel_size_h, nh_size_h, dilation_h);
                for (int oj = 0; oj < w_out; oj++) {
                    int qj = oj * stride_w;
                    int nj = is_causal_w ? get_causal_window_start(qj, kernel_size_w, dilation_w)
                                         : get_window_start(qj, width, kernel_size_w, nh_size_w, dilation_w);
                    for (int kd = 0; kd < kernel_size_d; kd++) {
                        int val_d = kd * dilation_d + nd;
                        if (val_d != dp) continue;
                        bool vd = is_causal_d ? (val_d >= 0 && val_d < depth && val_d <= qd)
                                              : (val_d >= 0 && val_d < depth);
                        if (!vd) continue;
                        for (int ki = 0; ki < kernel_size_h; ki++) {
                            int val_i = ki * dilation_h + ni;
                            if (val_i != i) continue;
                            bool vh = is_causal_h ? (val_i >= 0 && val_i < height && val_i <= qi)
                                                  : (val_i >= 0 && val_i < height);
                            if (!vh) continue;
                            for (int kj = 0; kj < kernel_size_w; kj++) {
                                int val_j = kj * dilation_w + nj;
                                if (val_j != j) continue;
                                bool vw = is_causal_w ? (val_j >= 0 && val_j < width && val_j <= qj)
                                                      : (val_j >= 0 && val_j < width);
                                if (vw) {
                                    int a_idx  = ((((b*heads+h)*d_out_sz+od)*h_out+ohi)*w_out+oj)*k_vol
                                                 + (kd*kernel_size_h+ki)*kernel_size_w+kj;
                                    int do_idx = ((((b*heads+h)*d_out_sz+od)*h_out+ohi)*w_out+oj)*dim+d;
                                    grad += float(attn[a_idx]) * float(d_out[do_idx]);
                                }
                            }
                        }
                    }
                }
            }
        }
        int v_idx = ((((b*heads+h)*depth+dp)*height+i)*width+j)*dim+d;
        d_value[v_idx] = half(grad);
    }
}

// ===========================================================================
// Inverse-map backward kernels (CSR-based, replaces brute-force k/v backward)
// ===========================================================================
//
// Each key/value position has a precomputed list of (output_pos, neighbor_idx)
// pairs stored in CSR format:
//   inv_offsets[i] .. inv_offsets[i+1]  = range of edges for position i
//   inv_attn_base[edge]                 = linear index into d_attn (within bh)
//   inv_data_base[edge]                 = linear index into query/grad_out (within bh)
//
// Grid: (dim, num_positions, B*H)  for 1D
//        (dim, H*W, B*H)           for 2D
//        (dim, D*H*W, B*heads)     for 3D

// ---------------------------------------------------------------------------
// 1D QK backward: d_key (inverse)  –  grid: (dim, length, B*H)
// ---------------------------------------------------------------------------
kernel void natten1d_k_backward_inv(
    device const float* d_attn        [[buffer(0)]],
    device const float* query         [[buffer(1)]],
    device float*       d_key         [[buffer(2)]],
    device const int*   inv_offsets   [[buffer(3)]],
    device const int*   inv_attn_base [[buffer(4)]],
    device const int*   inv_query_base[[buffer(5)]],
    constant int&       batch_size    [[buffer(6)]],
    constant int&       heads         [[buffer(7)]],
    constant int&       length        [[buffer(8)]],
    constant int&       dim           [[buffer(9)]],
    constant int&       l_out         [[buffer(10)]],
    constant int&       kernel_size   [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]])
{
    int d = gid.x;
    int key_i = gid.y;
    int bh = gid.z;
    int b = bh / heads;
    int h = bh % heads;
    if (b >= batch_size || key_i >= length || d >= dim) return;

    int attn_bh = (b * heads + h) * l_out * kernel_size;
    int q_bh = (b * heads + h) * length * dim;

    int start = inv_offsets[key_i];
    int end = inv_offsets[key_i + 1];

    float acc = 0.0f;
    for (int edge = start; edge < end; edge++) {
        acc += d_attn[attn_bh + inv_attn_base[edge]] * query[q_bh + inv_query_base[edge] + d];
    }
    d_key[(b * heads + h) * length * dim + key_i * dim + d] = acc;
}

kernel void natten1d_k_backward_inv_f16(
    device const half*  d_attn        [[buffer(0)]],
    device const half*  query         [[buffer(1)]],
    device half*        d_key         [[buffer(2)]],
    device const int*   inv_offsets   [[buffer(3)]],
    device const int*   inv_attn_base [[buffer(4)]],
    device const int*   inv_query_base[[buffer(5)]],
    constant int&       batch_size    [[buffer(6)]],
    constant int&       heads         [[buffer(7)]],
    constant int&       length        [[buffer(8)]],
    constant int&       dim           [[buffer(9)]],
    constant int&       l_out         [[buffer(10)]],
    constant int&       kernel_size   [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]])
{
    int d = gid.x;
    int key_i = gid.y;
    int bh = gid.z;
    int b = bh / heads;
    int h = bh % heads;
    if (b >= batch_size || key_i >= length || d >= dim) return;

    int attn_bh = (b * heads + h) * l_out * kernel_size;
    int q_bh = (b * heads + h) * length * dim;

    int start = inv_offsets[key_i];
    int end = inv_offsets[key_i + 1];

    float acc = 0.0f;
    for (int edge = start; edge < end; edge++) {
        acc += float(d_attn[attn_bh + inv_attn_base[edge]]) * float(query[q_bh + inv_query_base[edge] + d]);
    }
    d_key[(b * heads + h) * length * dim + key_i * dim + d] = half(acc);
}

// ---------------------------------------------------------------------------
// 1D AV backward: d_value (inverse)  –  grid: (dim, length, B*H)
// ---------------------------------------------------------------------------
kernel void natten1d_v_backward_inv(
    device const float* d_out         [[buffer(0)]],
    device const float* attn          [[buffer(1)]],
    device float*       d_value       [[buffer(2)]],
    device const int*   inv_offsets   [[buffer(3)]],
    device const int*   inv_attn_base [[buffer(4)]],
    device const int*   inv_grad_base [[buffer(5)]],
    constant int&       batch_size    [[buffer(6)]],
    constant int&       heads         [[buffer(7)]],
    constant int&       length        [[buffer(8)]],
    constant int&       dim           [[buffer(9)]],
    constant int&       l_out         [[buffer(10)]],
    constant int&       kernel_size   [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]])
{
    int d = gid.x;
    int val_i = gid.y;
    int bh = gid.z;
    int b = bh / heads;
    int h = bh % heads;
    if (b >= batch_size || val_i >= length || d >= dim) return;

    int attn_bh = (b * heads + h) * l_out * kernel_size;
    int grad_bh = (b * heads + h) * l_out * dim;

    int start = inv_offsets[val_i];
    int end = inv_offsets[val_i + 1];

    float acc = 0.0f;
    for (int edge = start; edge < end; edge++) {
        acc += attn[attn_bh + inv_attn_base[edge]] * d_out[grad_bh + inv_grad_base[edge] + d];
    }
    d_value[(b * heads + h) * length * dim + val_i * dim + d] = acc;
}

kernel void natten1d_v_backward_inv_f16(
    device const half*  d_out         [[buffer(0)]],
    device const half*  attn          [[buffer(1)]],
    device half*        d_value       [[buffer(2)]],
    device const int*   inv_offsets   [[buffer(3)]],
    device const int*   inv_attn_base [[buffer(4)]],
    device const int*   inv_grad_base [[buffer(5)]],
    constant int&       batch_size    [[buffer(6)]],
    constant int&       heads         [[buffer(7)]],
    constant int&       length        [[buffer(8)]],
    constant int&       dim           [[buffer(9)]],
    constant int&       l_out         [[buffer(10)]],
    constant int&       kernel_size   [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]])
{
    int d = gid.x;
    int val_i = gid.y;
    int bh = gid.z;
    int b = bh / heads;
    int h = bh % heads;
    if (b >= batch_size || val_i >= length || d >= dim) return;

    int attn_bh = (b * heads + h) * l_out * kernel_size;
    int grad_bh = (b * heads + h) * l_out * dim;

    int start = inv_offsets[val_i];
    int end = inv_offsets[val_i + 1];

    float acc = 0.0f;
    for (int edge = start; edge < end; edge++) {
        acc += float(attn[attn_bh + inv_attn_base[edge]]) * float(d_out[grad_bh + inv_grad_base[edge] + d]);
    }
    d_value[(b * heads + h) * length * dim + val_i * dim + d] = half(acc);
}

// ---------------------------------------------------------------------------
// 2D QK backward: d_key (inverse)  –  grid: (dim, H*W, B*heads)
// ---------------------------------------------------------------------------
kernel void natten2d_k_backward_inv(
    device const float* d_attn        [[buffer(0)]],
    device const float* query         [[buffer(1)]],
    device float*       d_key         [[buffer(2)]],
    device const int*   inv_offsets   [[buffer(3)]],
    device const int*   inv_attn_base [[buffer(4)]],
    device const int*   inv_query_base[[buffer(5)]],
    constant int&       batch_size    [[buffer(6)]],
    constant int&       heads         [[buffer(7)]],
    constant int&       height        [[buffer(8)]],
    constant int&       width         [[buffer(9)]],
    constant int&       dim           [[buffer(10)]],
    constant int&       out_count     [[buffer(11)]],
    constant int&       k_area        [[buffer(12)]],
    uint3 gid [[thread_position_in_grid]])
{
    int d = gid.x;
    int key_linear = gid.y;
    int bh = gid.z;
    int b = bh / heads;
    int h = bh % heads;
    int hw = height * width;
    if (b >= batch_size || key_linear >= hw || d >= dim) return;

    int attn_bh = (b * heads + h) * out_count * k_area;
    int q_bh = (b * heads + h) * hw * dim;

    int start = inv_offsets[key_linear];
    int end = inv_offsets[key_linear + 1];

    float acc = 0.0f;
    for (int edge = start; edge < end; edge++) {
        acc += d_attn[attn_bh + inv_attn_base[edge]] * query[q_bh + inv_query_base[edge] + d];
    }
    d_key[(b * heads + h) * hw * dim + key_linear * dim + d] = acc;
}

kernel void natten2d_k_backward_inv_f16(
    device const half*  d_attn        [[buffer(0)]],
    device const half*  query         [[buffer(1)]],
    device half*        d_key         [[buffer(2)]],
    device const int*   inv_offsets   [[buffer(3)]],
    device const int*   inv_attn_base [[buffer(4)]],
    device const int*   inv_query_base[[buffer(5)]],
    constant int&       batch_size    [[buffer(6)]],
    constant int&       heads         [[buffer(7)]],
    constant int&       height        [[buffer(8)]],
    constant int&       width         [[buffer(9)]],
    constant int&       dim           [[buffer(10)]],
    constant int&       out_count     [[buffer(11)]],
    constant int&       k_area        [[buffer(12)]],
    uint3 gid [[thread_position_in_grid]])
{
    int d = gid.x;
    int key_linear = gid.y;
    int bh = gid.z;
    int b = bh / heads;
    int h = bh % heads;
    int hw = height * width;
    if (b >= batch_size || key_linear >= hw || d >= dim) return;

    int attn_bh = (b * heads + h) * out_count * k_area;
    int q_bh = (b * heads + h) * hw * dim;

    int start = inv_offsets[key_linear];
    int end = inv_offsets[key_linear + 1];

    float acc = 0.0f;
    for (int edge = start; edge < end; edge++) {
        acc += float(d_attn[attn_bh + inv_attn_base[edge]]) * float(query[q_bh + inv_query_base[edge] + d]);
    }
    d_key[(b * heads + h) * hw * dim + key_linear * dim + d] = half(acc);
}

// ---------------------------------------------------------------------------
// 2D AV backward: d_value (inverse)  –  grid: (dim, H*W, B*heads)
// ---------------------------------------------------------------------------
kernel void natten2d_v_backward_inv(
    device const float* d_out         [[buffer(0)]],
    device const float* attn          [[buffer(1)]],
    device float*       d_value       [[buffer(2)]],
    device const int*   inv_offsets   [[buffer(3)]],
    device const int*   inv_attn_base [[buffer(4)]],
    device const int*   inv_grad_base [[buffer(5)]],
    constant int&       batch_size    [[buffer(6)]],
    constant int&       heads         [[buffer(7)]],
    constant int&       height        [[buffer(8)]],
    constant int&       width         [[buffer(9)]],
    constant int&       dim           [[buffer(10)]],
    constant int&       out_count     [[buffer(11)]],
    constant int&       k_area        [[buffer(12)]],
    uint3 gid [[thread_position_in_grid]])
{
    int d = gid.x;
    int val_linear = gid.y;
    int bh = gid.z;
    int b = bh / heads;
    int h = bh % heads;
    int hw = height * width;
    if (b >= batch_size || val_linear >= hw || d >= dim) return;

    int attn_bh = (b * heads + h) * out_count * k_area;
    int grad_bh = (b * heads + h) * out_count * dim;

    int start = inv_offsets[val_linear];
    int end = inv_offsets[val_linear + 1];

    float acc = 0.0f;
    for (int edge = start; edge < end; edge++) {
        acc += attn[attn_bh + inv_attn_base[edge]] * d_out[grad_bh + inv_grad_base[edge] + d];
    }
    d_value[(b * heads + h) * hw * dim + val_linear * dim + d] = acc;
}

kernel void natten2d_v_backward_inv_f16(
    device const half*  d_out         [[buffer(0)]],
    device const half*  attn          [[buffer(1)]],
    device half*        d_value       [[buffer(2)]],
    device const int*   inv_offsets   [[buffer(3)]],
    device const int*   inv_attn_base [[buffer(4)]],
    device const int*   inv_grad_base [[buffer(5)]],
    constant int&       batch_size    [[buffer(6)]],
    constant int&       heads         [[buffer(7)]],
    constant int&       height        [[buffer(8)]],
    constant int&       width         [[buffer(9)]],
    constant int&       dim           [[buffer(10)]],
    constant int&       out_count     [[buffer(11)]],
    constant int&       k_area        [[buffer(12)]],
    uint3 gid [[thread_position_in_grid]])
{
    int d = gid.x;
    int val_linear = gid.y;
    int bh = gid.z;
    int b = bh / heads;
    int h = bh % heads;
    int hw = height * width;
    if (b >= batch_size || val_linear >= hw || d >= dim) return;

    int attn_bh = (b * heads + h) * out_count * k_area;
    int grad_bh = (b * heads + h) * out_count * dim;

    int start = inv_offsets[val_linear];
    int end = inv_offsets[val_linear + 1];

    float acc = 0.0f;
    for (int edge = start; edge < end; edge++) {
        acc += float(attn[attn_bh + inv_attn_base[edge]]) * float(d_out[grad_bh + inv_grad_base[edge] + d]);
    }
    d_value[(b * heads + h) * hw * dim + val_linear * dim + d] = half(acc);
}

// ---------------------------------------------------------------------------
// 3D QK backward: d_key (inverse)  –  grid: (dim, D*H*W, B*heads)
// ---------------------------------------------------------------------------
kernel void natten3d_k_backward_inv(
    device const float* d_attn        [[buffer(0)]],
    device const float* query         [[buffer(1)]],
    device float*       d_key         [[buffer(2)]],
    device const int*   inv_offsets   [[buffer(3)]],
    device const int*   inv_attn_base [[buffer(4)]],
    device const int*   inv_query_base[[buffer(5)]],
    constant int&       batch_size    [[buffer(6)]],
    constant int&       heads         [[buffer(7)]],
    constant int&       vol           [[buffer(8)]],
    constant int&       dim           [[buffer(9)]],
    constant int&       out_count     [[buffer(10)]],
    constant int&       k_vol         [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]])
{
    int d = gid.x;
    int key_linear = gid.y;
    int bh = gid.z;
    int b = bh / heads;
    int h = bh % heads;
    if (b >= batch_size || key_linear >= vol || d >= dim) return;

    int attn_bh = (b * heads + h) * out_count * k_vol;
    int q_bh = (b * heads + h) * vol * dim;

    int start = inv_offsets[key_linear];
    int end = inv_offsets[key_linear + 1];

    float acc = 0.0f;
    for (int edge = start; edge < end; edge++) {
        acc += d_attn[attn_bh + inv_attn_base[edge]] * query[q_bh + inv_query_base[edge] + d];
    }
    d_key[(b * heads + h) * vol * dim + key_linear * dim + d] = acc;
}

kernel void natten3d_k_backward_inv_f16(
    device const half*  d_attn        [[buffer(0)]],
    device const half*  query         [[buffer(1)]],
    device half*        d_key         [[buffer(2)]],
    device const int*   inv_offsets   [[buffer(3)]],
    device const int*   inv_attn_base [[buffer(4)]],
    device const int*   inv_query_base[[buffer(5)]],
    constant int&       batch_size    [[buffer(6)]],
    constant int&       heads         [[buffer(7)]],
    constant int&       vol           [[buffer(8)]],
    constant int&       dim           [[buffer(9)]],
    constant int&       out_count     [[buffer(10)]],
    constant int&       k_vol         [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]])
{
    int d = gid.x;
    int key_linear = gid.y;
    int bh = gid.z;
    int b = bh / heads;
    int h = bh % heads;
    if (b >= batch_size || key_linear >= vol || d >= dim) return;

    int attn_bh = (b * heads + h) * out_count * k_vol;
    int q_bh = (b * heads + h) * vol * dim;

    int start = inv_offsets[key_linear];
    int end = inv_offsets[key_linear + 1];

    float acc = 0.0f;
    for (int edge = start; edge < end; edge++) {
        acc += float(d_attn[attn_bh + inv_attn_base[edge]]) * float(query[q_bh + inv_query_base[edge] + d]);
    }
    d_key[(b * heads + h) * vol * dim + key_linear * dim + d] = half(acc);
}

// ---------------------------------------------------------------------------
// 3D AV backward: d_value (inverse)  –  grid: (dim, D*H*W, B*heads)
// ---------------------------------------------------------------------------
kernel void natten3d_v_backward_inv(
    device const float* d_out         [[buffer(0)]],
    device const float* attn          [[buffer(1)]],
    device float*       d_value       [[buffer(2)]],
    device const int*   inv_offsets   [[buffer(3)]],
    device const int*   inv_attn_base [[buffer(4)]],
    device const int*   inv_grad_base [[buffer(5)]],
    constant int&       batch_size    [[buffer(6)]],
    constant int&       heads         [[buffer(7)]],
    constant int&       vol           [[buffer(8)]],
    constant int&       dim           [[buffer(9)]],
    constant int&       out_count     [[buffer(10)]],
    constant int&       k_vol         [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]])
{
    int d = gid.x;
    int val_linear = gid.y;
    int bh = gid.z;
    int b = bh / heads;
    int h = bh % heads;
    if (b >= batch_size || val_linear >= vol || d >= dim) return;

    int attn_bh = (b * heads + h) * out_count * k_vol;
    int grad_bh = (b * heads + h) * out_count * dim;

    int start = inv_offsets[val_linear];
    int end = inv_offsets[val_linear + 1];

    float acc = 0.0f;
    for (int edge = start; edge < end; edge++) {
        acc += attn[attn_bh + inv_attn_base[edge]] * d_out[grad_bh + inv_grad_base[edge] + d];
    }
    d_value[(b * heads + h) * vol * dim + val_linear * dim + d] = acc;
}

kernel void natten3d_v_backward_inv_f16(
    device const half*  d_out         [[buffer(0)]],
    device const half*  attn          [[buffer(1)]],
    device half*        d_value       [[buffer(2)]],
    device const int*   inv_offsets   [[buffer(3)]],
    device const int*   inv_attn_base [[buffer(4)]],
    device const int*   inv_grad_base [[buffer(5)]],
    constant int&       batch_size    [[buffer(6)]],
    constant int&       heads         [[buffer(7)]],
    constant int&       vol           [[buffer(8)]],
    constant int&       dim           [[buffer(9)]],
    constant int&       out_count     [[buffer(10)]],
    constant int&       k_vol         [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]])
{
    int d = gid.x;
    int val_linear = gid.y;
    int bh = gid.z;
    int b = bh / heads;
    int h = bh % heads;
    if (b >= batch_size || val_linear >= vol || d >= dim) return;

    int attn_bh = (b * heads + h) * out_count * k_vol;
    int grad_bh = (b * heads + h) * out_count * dim;

    int start = inv_offsets[val_linear];
    int end = inv_offsets[val_linear + 1];

    float acc = 0.0f;
    for (int edge = start; edge < end; edge++) {
        acc += float(attn[attn_bh + inv_attn_base[edge]]) * float(d_out[grad_bh + inv_grad_base[edge] + d]);
    }
    d_value[(b * heads + h) * vol * dim + val_linear * dim + d] = half(acc);
}
"""
