"""Metal shader source for extras/allin1 QK+RPB and AV kernels.

Ported from natten-mlx's extras/allin1/kernels.py, adapted for
torch.mps.compile_shader() which requires full Metal kernel functions
with explicit buffer bindings (vs MLX's body-only format).

Covers K={3,5,7} × {1D,2D} × {QK+RPB, AV}:
  - 12 forward kernels (fp32)
  - 30 backward kernels (dQ, dK, dRPB, dAttn, dVal × 1D/2D × K=3,5,7)
"""

ALLIN1_METAL_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Helper functions (matching natten-mlx shift semantics)
// ---------------------------------------------------------------------------

inline int get_window_start(int idx, int len, int K, int nh, int dil) {
    int dilation_idx = idx % dil;
    int index_pdp = idx / dil;
    int length_pdp = (len + dil - 1) / dil;
    int num_padded = (length_pdp * dil) - len;
    if (dilation_idx >= (dil - num_padded)) {
        length_pdp -= 1;
    }
    int start_idx = index_pdp - nh;
    if (start_idx < 0) start_idx = 0;
    if (index_pdp + nh >= length_pdp) {
        start_idx += (length_pdp - index_pdp - nh - 1);
    }
    return start_idx * dil + dilation_idx;
}

inline int get_window_end(int start, int len, int K, int dil) {
    int end = start + K * dil;
    if (end > len) end = len;
    return end;
}

inline int get_pb_start(int idx, int len, int K, int nh, int dil) {
    int pb;
    if (dil <= 1) {
        pb = nh;
        if (idx < nh) pb += (nh - idx);
        if (idx + nh >= len) pb += (len - idx - 1 - nh);
    } else {
        if (idx - nh * dil < 0) {
            pb = (K - 1) - (idx / dil);
        } else if (idx + nh * dil >= len) {
            pb = (len - idx - 1) / dil;
        } else {
            pb = nh;
        }
    }
    return pb;
}

// ---------------------------------------------------------------------------
// 1D QK+RPB kernels
// ---------------------------------------------------------------------------

#define DEFINE_1D_QKRPB_KERNEL(KNAME, KSIZE, NH_VAL)                         \
kernel void KNAME(                                                            \
    device const float* query    [[buffer(0)]],                               \
    device const float* key      [[buffer(1)]],                               \
    device const float* rpb      [[buffer(2)]],                               \
    device float*       out      [[buffer(3)]],                               \
    constant int&       batch_size [[buffer(4)]],                             \
    constant int&       heads    [[buffer(5)]],                               \
    constant int&       length   [[buffer(6)]],                               \
    constant int&       dim      [[buffer(7)]],                               \
    constant int&       dilation [[buffer(8)]],                               \
    uint3 gid [[thread_position_in_grid]])                                    \
{                                                                             \
    const int K = KSIZE;                                                      \
    const int NH = NH_VAL;                                                    \
    int b = gid.z / heads;                                                    \
    int h = gid.z % heads;                                                    \
    int i = gid.x;                                                            \
    if (b >= batch_size || h >= heads || i >= length) return;                  \
                                                                              \
    int ni = get_window_start(i, length, K, NH, dilation);                    \
    int ei = get_window_end(ni, length, K, dilation);                         \
    int pi = get_pb_start(i, length, K, NH, dilation);                        \
                                                                              \
    for (int ki = 0; ki < K; ki++) {                                          \
        int key_i = ni + ki * dilation;                                       \
        float score;                                                          \
        if (key_i >= 0 && key_i < ei) {                                       \
            float sum = 0.0f;                                                 \
            int q_base = ((b * heads + h) * length + i) * dim;                \
            int k_base = ((b * heads + h) * length + key_i) * dim;            \
            for (int d = 0; d < dim; d++) {                                   \
                sum += query[q_base + d] * key[k_base + d];                   \
            }                                                                 \
            int rpb_idx = h * (2 * K - 1) + (pi + ki);                       \
            score = sum + rpb[rpb_idx];                                       \
        } else {                                                              \
            score = -INFINITY;                                                \
        }                                                                     \
        int out_idx = ((b * heads + h) * length + i) * K + ki;               \
        out[out_idx] = score;                                                 \
    }                                                                         \
}

DEFINE_1D_QKRPB_KERNEL(natten1d_qkrpb_k3, 3, 1)
DEFINE_1D_QKRPB_KERNEL(natten1d_qkrpb_k5, 5, 2)
DEFINE_1D_QKRPB_KERNEL(natten1d_qkrpb_k7, 7, 3)

// ---------------------------------------------------------------------------
// 1D AV kernels
// ---------------------------------------------------------------------------

#define DEFINE_1D_AV_KERNEL(KNAME, KSIZE, NH_VAL)                             \
kernel void KNAME(                                                            \
    device const float* attn     [[buffer(0)]],                               \
    device const float* value    [[buffer(1)]],                               \
    device float*       out      [[buffer(2)]],                               \
    constant int&       batch_size [[buffer(3)]],                             \
    constant int&       heads    [[buffer(4)]],                               \
    constant int&       length   [[buffer(5)]],                               \
    constant int&       dim      [[buffer(6)]],                               \
    constant int&       dilation [[buffer(7)]],                               \
    uint3 gid [[thread_position_in_grid]])                                    \
{                                                                             \
    const int K = KSIZE;                                                      \
    const int NH = NH_VAL;                                                    \
    int b = gid.z / heads;                                                    \
    int h = gid.z % heads;                                                    \
    int i = gid.x;                                                            \
    if (b >= batch_size || h >= heads || i >= length) return;                  \
                                                                              \
    int ni = get_window_start(i, length, K, NH, dilation);                    \
    int ei = get_window_end(ni, length, K, dilation);                         \
                                                                              \
    int out_base = ((b * heads + h) * length + i) * dim;                      \
    for (int d = 0; d < dim; d++) {                                           \
        float sum = 0.0f;                                                     \
        for (int ki = 0; ki < K; ki++) {                                      \
            int val_i = ni + ki * dilation;                                   \
            if (val_i >= 0 && val_i < ei) {                                   \
                int attn_idx = ((b * heads + h) * length + i) * K + ki;       \
                int val_idx = ((b * heads + h) * length + val_i) * dim + d;   \
                sum += attn[attn_idx] * value[val_idx];                       \
            }                                                                 \
        }                                                                     \
        out[out_base + d] = sum;                                              \
    }                                                                         \
}

DEFINE_1D_AV_KERNEL(natten1d_av_k3, 3, 1)
DEFINE_1D_AV_KERNEL(natten1d_av_k5, 5, 2)
DEFINE_1D_AV_KERNEL(natten1d_av_k7, 7, 3)

// ---------------------------------------------------------------------------
// 2D QK+RPB kernels
// ---------------------------------------------------------------------------

#define DEFINE_2D_QKRPB_KERNEL(KNAME, KSIZE, NH_VAL)                         \
kernel void KNAME(                                                            \
    device const float* query    [[buffer(0)]],                               \
    device const float* key      [[buffer(1)]],                               \
    device const float* rpb      [[buffer(2)]],                               \
    device float*       out      [[buffer(3)]],                               \
    constant int&       batch_size [[buffer(4)]],                             \
    constant int&       heads    [[buffer(5)]],                               \
    constant int&       height   [[buffer(6)]],                               \
    constant int&       width    [[buffer(7)]],                               \
    constant int&       dim      [[buffer(8)]],                               \
    constant int&       dilation [[buffer(9)]],                               \
    uint3 gid [[thread_position_in_grid]])                                    \
{                                                                             \
    const int K = KSIZE;                                                      \
    const int NH = NH_VAL;                                                    \
    const int L = K * K;                                                      \
    int b = gid.z / heads;                                                    \
    int h = gid.z % heads;                                                    \
    int i = gid.y;                                                            \
    int j = gid.x;                                                            \
    if (b >= batch_size || h >= heads || i >= height || j >= width) return;    \
                                                                              \
    int ni = get_window_start(i, height, K, NH, dilation);                    \
    int nj = get_window_start(j, width,  K, NH, dilation);                    \
    int ei = get_window_end(ni, height, K, dilation);                         \
    int ej = get_window_end(nj, width,  K, dilation);                         \
    int pi = get_pb_start(i, height, K, NH, dilation);                        \
    int pj = get_pb_start(j, width,  K, NH, dilation);                        \
                                                                              \
    int neighbor_idx = 0;                                                     \
    for (int ki = 0; ki < K; ki++) {                                          \
        for (int kj = 0; kj < K; kj++) {                                     \
            int key_i = ni + ki * dilation;                                   \
            int key_j = nj + kj * dilation;                                   \
            float score;                                                      \
            if (key_i >= 0 && key_i < ei && key_j >= 0 && key_j < ej) {      \
                float sum = 0.0f;                                             \
                int q_base = (((b*heads+h)*height+i)*width+j)*dim;            \
                int k_base = (((b*heads+h)*height+key_i)*width+key_j)*dim;    \
                for (int d = 0; d < dim; d++) {                               \
                    sum += query[q_base + d] * key[k_base + d];               \
                }                                                             \
                int rpb_row = pi + ki;                                        \
                int rpb_col = pj + kj;                                        \
                int rpb_idx = h*(2*K-1)*(2*K-1) + rpb_row*(2*K-1) + rpb_col; \
                score = sum + rpb[rpb_idx];                                   \
            } else {                                                          \
                score = -INFINITY;                                            \
            }                                                                 \
            int out_idx = (((b*heads+h)*height+i)*width+j)*L + neighbor_idx;  \
            out[out_idx] = score;                                             \
            neighbor_idx++;                                                   \
        }                                                                     \
    }                                                                         \
}

DEFINE_2D_QKRPB_KERNEL(natten2d_qkrpb_k3, 3, 1)
DEFINE_2D_QKRPB_KERNEL(natten2d_qkrpb_k5, 5, 2)
DEFINE_2D_QKRPB_KERNEL(natten2d_qkrpb_k7, 7, 3)

// ---------------------------------------------------------------------------
// 2D AV kernels
// ---------------------------------------------------------------------------

#define DEFINE_2D_AV_KERNEL(KNAME, KSIZE, NH_VAL)                             \
kernel void KNAME(                                                            \
    device const float* attn     [[buffer(0)]],                               \
    device const float* value    [[buffer(1)]],                               \
    device float*       out      [[buffer(2)]],                               \
    constant int&       batch_size [[buffer(3)]],                             \
    constant int&       heads    [[buffer(4)]],                               \
    constant int&       height   [[buffer(5)]],                               \
    constant int&       width    [[buffer(6)]],                               \
    constant int&       dim      [[buffer(7)]],                               \
    constant int&       dilation [[buffer(8)]],                               \
    uint3 gid [[thread_position_in_grid]])                                    \
{                                                                             \
    const int K = KSIZE;                                                      \
    const int NH = NH_VAL;                                                    \
    const int L = K * K;                                                      \
    int b = gid.z / heads;                                                    \
    int h = gid.z % heads;                                                    \
    int i = gid.y;                                                            \
    int j = gid.x;                                                            \
    if (b >= batch_size || h >= heads || i >= height || j >= width) return;    \
                                                                              \
    int ni = get_window_start(i, height, K, NH, dilation);                    \
    int nj = get_window_start(j, width,  K, NH, dilation);                    \
    int ei = get_window_end(ni, height, K, dilation);                         \
    int ej = get_window_end(nj, width,  K, dilation);                         \
                                                                              \
    int out_base = (((b*heads+h)*height+i)*width+j)*dim;                      \
    for (int d = 0; d < dim; d++) {                                           \
        float sum = 0.0f;                                                     \
        int neighbor_idx = 0;                                                 \
        for (int ki = 0; ki < K; ki++) {                                      \
            for (int kj = 0; kj < K; kj++) {                                  \
                int val_i = ni + ki * dilation;                               \
                int val_j = nj + kj * dilation;                               \
                if (val_i >= 0 && val_i < ei && val_j >= 0 && val_j < ej) {   \
                    int attn_idx = (((b*heads+h)*height+i)*width+j)*L         \
                                   + neighbor_idx;                            \
                    int val_idx = (((b*heads+h)*height+val_i)*width+val_j)    \
                                  *dim + d;                                   \
                    sum += attn[attn_idx] * value[val_idx];                   \
                }                                                             \
                neighbor_idx++;                                               \
            }                                                                 \
        }                                                                     \
        out[out_base + d] = sum;                                              \
    }                                                                         \
}

DEFINE_2D_AV_KERNEL(natten2d_av_k3, 3, 1)
DEFINE_2D_AV_KERNEL(natten2d_av_k5, 5, 2)
DEFINE_2D_AV_KERNEL(natten2d_av_k7, 7, 3)

// ===========================================================================
// BACKWARD KERNELS
// ===========================================================================

// ---------------------------------------------------------------------------
// 1D QK backward: d_query
// Grid: (length, 1, B*H)
// ---------------------------------------------------------------------------

#define DEFINE_1D_QK_BWD_DQ_KERNEL(KNAME, KSIZE, NH_VAL)                      \
kernel void KNAME(                                                            \
    device const float* query    [[buffer(0)]],                               \
    device const float* key      [[buffer(1)]],                               \
    device const float* d_attn   [[buffer(2)]],                               \
    device float*       out      [[buffer(3)]],                               \
    constant int&       batch_size [[buffer(4)]],                             \
    constant int&       heads    [[buffer(5)]],                               \
    constant int&       length   [[buffer(6)]],                               \
    constant int&       dim      [[buffer(7)]],                               \
    constant int&       dilation [[buffer(8)]],                               \
    uint3 gid [[thread_position_in_grid]])                                    \
{                                                                             \
    const int K = KSIZE;                                                      \
    const int NH = NH_VAL;                                                    \
    int b = gid.z / heads;                                                    \
    int h = gid.z % heads;                                                    \
    int i = gid.x;                                                            \
    if (b >= batch_size || h >= heads || i >= length) return;                  \
    int ni = get_window_start(i, length, K, NH, dilation);                    \
    int ei = get_window_end(ni, length, K, dilation);                         \
    for (int d = 0; d < dim; d++) {                                           \
        float sum = 0.0f;                                                     \
        int neighbor_idx = 0;                                                 \
        for (int ki = 0; ki < K; ki++) {                                      \
            int key_i = ni + ki * dilation;                                   \
            if (key_i >= 0 && key_i < ei) {                                   \
                int k_idx = ((b * heads + h) * length + key_i) * dim + d;     \
                int da_idx = ((b * heads + h) * length + i) * K               \
                             + neighbor_idx;                                  \
                sum += d_attn[da_idx] * key[k_idx];                           \
            }                                                                 \
            neighbor_idx++;                                                   \
        }                                                                     \
        int out_idx = ((b * heads + h) * length + i) * dim + d;              \
        out[out_idx] = sum;                                                   \
    }                                                                         \
}

DEFINE_1D_QK_BWD_DQ_KERNEL(natten1d_dq_k3, 3, 1)
DEFINE_1D_QK_BWD_DQ_KERNEL(natten1d_dq_k5, 5, 2)
DEFINE_1D_QK_BWD_DQ_KERNEL(natten1d_dq_k7, 7, 3)

// ---------------------------------------------------------------------------
// 1D QK backward: d_key (reduction over all query positions)
// Grid: (length, 1, B*H)
// ---------------------------------------------------------------------------

#define DEFINE_1D_QK_BWD_DK_KERNEL(KNAME, KSIZE, NH_VAL)                      \
kernel void KNAME(                                                            \
    device const float* query    [[buffer(0)]],                               \
    device const float* key      [[buffer(1)]],                               \
    device const float* d_attn   [[buffer(2)]],                               \
    device float*       out      [[buffer(3)]],                               \
    constant int&       batch_size [[buffer(4)]],                             \
    constant int&       heads    [[buffer(5)]],                               \
    constant int&       length   [[buffer(6)]],                               \
    constant int&       dim      [[buffer(7)]],                               \
    constant int&       dilation [[buffer(8)]],                               \
    uint3 gid [[thread_position_in_grid]])                                    \
{                                                                             \
    const int K = KSIZE;                                                      \
    const int NH = NH_VAL;                                                    \
    int b = gid.z / heads;                                                    \
    int h = gid.z % heads;                                                    \
    int vi = gid.x;                                                           \
    if (b >= batch_size || h >= heads || vi >= length) return;                 \
    for (int d = 0; d < dim; d++) {                                           \
        float sum = 0.0f;                                                     \
        for (int i = 0; i < length; i++) {                                    \
            int ni = get_window_start(i, length, K, NH, dilation);            \
            int ei = get_window_end(ni, length, K, dilation);                 \
            if (vi < ni || vi >= ei) continue;                                \
            int di = vi - ni;                                                 \
            if (di % dilation != 0) continue;                                 \
            int ki = di / dilation;                                           \
            if (ki < 0 || ki >= K) continue;                                  \
            int da_idx = ((b * heads + h) * length + i) * K + ki;            \
            int q_idx = ((b * heads + h) * length + i) * dim + d;            \
            sum += d_attn[da_idx] * query[q_idx];                            \
        }                                                                     \
        int out_idx = ((b * heads + h) * length + vi) * dim + d;             \
        out[out_idx] = sum;                                                   \
    }                                                                         \
}

DEFINE_1D_QK_BWD_DK_KERNEL(natten1d_dk_k3, 3, 1)
DEFINE_1D_QK_BWD_DK_KERNEL(natten1d_dk_k5, 5, 2)
DEFINE_1D_QK_BWD_DK_KERNEL(natten1d_dk_k7, 7, 3)

// ---------------------------------------------------------------------------
// 1D QK backward: d_rpb (reduction over batch and spatial)
// Grid: (2*K-1, 1, H)
// ---------------------------------------------------------------------------

#define DEFINE_1D_QK_BWD_DRPB_KERNEL(KNAME, KSIZE, NH_VAL)                    \
kernel void KNAME(                                                            \
    device const float* d_attn   [[buffer(0)]],                               \
    device float*       out      [[buffer(1)]],                               \
    constant int&       batch_size [[buffer(2)]],                             \
    constant int&       heads    [[buffer(3)]],                               \
    constant int&       length   [[buffer(4)]],                               \
    constant int&       dilation [[buffer(5)]],                               \
    uint3 gid [[thread_position_in_grid]])                                    \
{                                                                             \
    const int K = KSIZE;                                                      \
    const int NH = NH_VAL;                                                    \
    const int rpb_len = 2 * K - 1;                                            \
    int h = gid.z;                                                            \
    int ri = gid.x;                                                           \
    if (h >= heads || ri >= rpb_len) return;                                   \
    float sum = 0.0f;                                                         \
    for (int b = 0; b < batch_size; b++) {                                    \
        for (int i = 0; i < length; i++) {                                    \
            int ni = get_window_start(i, length, K, NH, dilation);            \
            int ei = get_window_end(ni, length, K, dilation);                 \
            int pi = get_pb_start(i, length, K, NH, dilation);                \
            int ki = ri - pi;                                                 \
            if (ki < 0 || ki >= K) continue;                                  \
            int key_i = ni + ki * dilation;                                   \
            if (key_i < 0 || key_i >= ei) continue;                           \
            int da_idx = ((b * heads + h) * length + i) * K + ki;            \
            sum += d_attn[da_idx];                                            \
        }                                                                     \
    }                                                                         \
    int out_idx = h * rpb_len + ri;                                           \
    out[out_idx] = sum;                                                       \
}

DEFINE_1D_QK_BWD_DRPB_KERNEL(natten1d_drpb_k3, 3, 1)
DEFINE_1D_QK_BWD_DRPB_KERNEL(natten1d_drpb_k5, 5, 2)
DEFINE_1D_QK_BWD_DRPB_KERNEL(natten1d_drpb_k7, 7, 3)

// ---------------------------------------------------------------------------
// 1D AV backward: d_attn
// Grid: (length, 1, B*H)
// ---------------------------------------------------------------------------

#define DEFINE_1D_AV_BWD_DATTN_KERNEL(KNAME, KSIZE, NH_VAL)                   \
kernel void KNAME(                                                            \
    device const float* d_out    [[buffer(0)]],                               \
    device const float* value    [[buffer(1)]],                               \
    device float*       out      [[buffer(2)]],                               \
    constant int&       batch_size [[buffer(3)]],                             \
    constant int&       heads    [[buffer(4)]],                               \
    constant int&       length   [[buffer(5)]],                               \
    constant int&       dim      [[buffer(6)]],                               \
    constant int&       dilation [[buffer(7)]],                               \
    uint3 gid [[thread_position_in_grid]])                                    \
{                                                                             \
    const int K = KSIZE;                                                      \
    const int NH = NH_VAL;                                                    \
    int b = gid.z / heads;                                                    \
    int h = gid.z % heads;                                                    \
    int i = gid.x;                                                            \
    if (b >= batch_size || h >= heads || i >= length) return;                  \
    int ni = get_window_start(i, length, K, NH, dilation);                    \
    int ei = get_window_end(ni, length, K, dilation);                         \
    for (int ki = 0; ki < K; ki++) {                                          \
        int val_i = ni + ki * dilation;                                       \
        float sum = 0.0f;                                                     \
        if (val_i >= 0 && val_i < ei) {                                       \
            for (int d = 0; d < dim; d++) {                                   \
                int do_idx = ((b * heads + h) * length + i) * dim + d;        \
                int v_idx = ((b * heads + h) * length + val_i) * dim + d;     \
                sum += d_out[do_idx] * value[v_idx];                          \
            }                                                                 \
        }                                                                     \
        int out_idx = ((b * heads + h) * length + i) * K + ki;               \
        out[out_idx] = sum;                                                   \
    }                                                                         \
}

DEFINE_1D_AV_BWD_DATTN_KERNEL(natten1d_dattn_k3, 3, 1)
DEFINE_1D_AV_BWD_DATTN_KERNEL(natten1d_dattn_k5, 5, 2)
DEFINE_1D_AV_BWD_DATTN_KERNEL(natten1d_dattn_k7, 7, 3)

// ---------------------------------------------------------------------------
// 1D AV backward: d_value (reduction over all query positions)
// Grid: (length, 1, B*H)
// ---------------------------------------------------------------------------

#define DEFINE_1D_AV_BWD_DVAL_KERNEL(KNAME, KSIZE, NH_VAL)                    \
kernel void KNAME(                                                            \
    device const float* d_out    [[buffer(0)]],                               \
    device const float* attn     [[buffer(1)]],                               \
    device float*       out      [[buffer(2)]],                               \
    constant int&       batch_size [[buffer(3)]],                             \
    constant int&       heads    [[buffer(4)]],                               \
    constant int&       length   [[buffer(5)]],                               \
    constant int&       dim      [[buffer(6)]],                               \
    constant int&       dilation [[buffer(7)]],                               \
    uint3 gid [[thread_position_in_grid]])                                    \
{                                                                             \
    const int K = KSIZE;                                                      \
    const int NH = NH_VAL;                                                    \
    int b = gid.z / heads;                                                    \
    int h = gid.z % heads;                                                    \
    int vi = gid.x;                                                           \
    if (b >= batch_size || h >= heads || vi >= length) return;                 \
    for (int d = 0; d < dim; d++) {                                           \
        float sum = 0.0f;                                                     \
        for (int i = 0; i < length; i++) {                                    \
            int ni = get_window_start(i, length, K, NH, dilation);            \
            int ei = get_window_end(ni, length, K, dilation);                 \
            if (vi < ni || vi >= ei) continue;                                \
            int di = vi - ni;                                                 \
            if (di % dilation != 0) continue;                                 \
            int ki = di / dilation;                                           \
            if (ki < 0 || ki >= K) continue;                                  \
            int do_idx = ((b * heads + h) * length + i) * dim + d;           \
            int attn_idx = ((b * heads + h) * length + i) * K + ki;          \
            sum += d_out[do_idx] * attn[attn_idx];                           \
        }                                                                     \
        int out_idx = ((b * heads + h) * length + vi) * dim + d;             \
        out[out_idx] = sum;                                                   \
    }                                                                         \
}

DEFINE_1D_AV_BWD_DVAL_KERNEL(natten1d_dval_k3, 3, 1)
DEFINE_1D_AV_BWD_DVAL_KERNEL(natten1d_dval_k5, 5, 2)
DEFINE_1D_AV_BWD_DVAL_KERNEL(natten1d_dval_k7, 7, 3)

// ---------------------------------------------------------------------------
// 2D QK backward: d_query
// Grid: (width, height, B*H)
// ---------------------------------------------------------------------------

#define DEFINE_2D_QK_BWD_DQ_KERNEL(KNAME, KSIZE, NH_VAL)                      \
kernel void KNAME(                                                            \
    device const float* query    [[buffer(0)]],                               \
    device const float* key      [[buffer(1)]],                               \
    device const float* d_attn   [[buffer(2)]],                               \
    device float*       out      [[buffer(3)]],                               \
    constant int&       batch_size [[buffer(4)]],                             \
    constant int&       heads    [[buffer(5)]],                               \
    constant int&       height   [[buffer(6)]],                               \
    constant int&       width    [[buffer(7)]],                               \
    constant int&       dim      [[buffer(8)]],                               \
    constant int&       dilation [[buffer(9)]],                               \
    uint3 gid [[thread_position_in_grid]])                                    \
{                                                                             \
    const int K = KSIZE;                                                      \
    const int NH = NH_VAL;                                                    \
    const int L = K * K;                                                      \
    int b = gid.z / heads;                                                    \
    int h = gid.z % heads;                                                    \
    int i = gid.y;                                                            \
    int j = gid.x;                                                            \
    if (b >= batch_size || h >= heads || i >= height || j >= width) return;    \
    int ni = get_window_start(i, height, K, NH, dilation);                    \
    int nj = get_window_start(j, width,  K, NH, dilation);                    \
    int ei = get_window_end(ni, height, K, dilation);                         \
    int ej = get_window_end(nj, width,  K, dilation);                         \
    for (int d = 0; d < dim; d++) {                                           \
        float sum = 0.0f;                                                     \
        int neighbor_idx = 0;                                                 \
        for (int ki = 0; ki < K; ki++) {                                      \
            for (int kj = 0; kj < K; kj++) {                                  \
                int key_i = ni + ki * dilation;                               \
                int key_j = nj + kj * dilation;                               \
                if (key_i >= 0 && key_i < ei &&                               \
                    key_j >= 0 && key_j < ej) {                               \
                    int k_idx = (((b*heads+h)*height+key_i)*width+key_j)      \
                                *dim + d;                                     \
                    int da_idx = (((b*heads+h)*height+i)*width+j)*L           \
                                 + neighbor_idx;                              \
                    sum += d_attn[da_idx] * key[k_idx];                       \
                }                                                             \
                neighbor_idx++;                                               \
            }                                                                 \
        }                                                                     \
        int out_idx = (((b*heads+h)*height+i)*width+j)*dim + d;              \
        out[out_idx] = sum;                                                   \
    }                                                                         \
}

DEFINE_2D_QK_BWD_DQ_KERNEL(natten2d_dq_k3, 3, 1)
DEFINE_2D_QK_BWD_DQ_KERNEL(natten2d_dq_k5, 5, 2)
DEFINE_2D_QK_BWD_DQ_KERNEL(natten2d_dq_k7, 7, 3)

// ---------------------------------------------------------------------------
// 2D QK backward: d_key (reduction over all query positions)
// Grid: (width, height, B*H)
// ---------------------------------------------------------------------------

#define DEFINE_2D_QK_BWD_DK_KERNEL(KNAME, KSIZE, NH_VAL)                      \
kernel void KNAME(                                                            \
    device const float* query    [[buffer(0)]],                               \
    device const float* key      [[buffer(1)]],                               \
    device const float* d_attn   [[buffer(2)]],                               \
    device float*       out      [[buffer(3)]],                               \
    constant int&       batch_size [[buffer(4)]],                             \
    constant int&       heads    [[buffer(5)]],                               \
    constant int&       height   [[buffer(6)]],                               \
    constant int&       width    [[buffer(7)]],                               \
    constant int&       dim      [[buffer(8)]],                               \
    constant int&       dilation [[buffer(9)]],                               \
    uint3 gid [[thread_position_in_grid]])                                    \
{                                                                             \
    const int K = KSIZE;                                                      \
    const int NH = NH_VAL;                                                    \
    const int L = K * K;                                                      \
    int b = gid.z / heads;                                                    \
    int h = gid.z % heads;                                                    \
    int vi = gid.y;                                                           \
    int vj = gid.x;                                                           \
    if (b >= batch_size || h >= heads || vi >= height || vj >= width) return;  \
    for (int d = 0; d < dim; d++) {                                           \
        float sum = 0.0f;                                                     \
        for (int i = 0; i < height; i++) {                                    \
            int ni = get_window_start(i, height, K, NH, dilation);            \
            int ei = get_window_end(ni, height, K, dilation);                 \
            if (vi < ni || vi >= ei) continue;                                \
            int di = vi - ni;                                                 \
            if (di % dilation != 0) continue;                                 \
            int ki = di / dilation;                                           \
            if (ki < 0 || ki >= K) continue;                                  \
            for (int j = 0; j < width; j++) {                                 \
                int nj = get_window_start(j, width, K, NH, dilation);         \
                int ej = get_window_end(nj, width, K, dilation);              \
                if (vj < nj || vj >= ej) continue;                            \
                int dj = vj - nj;                                             \
                if (dj % dilation != 0) continue;                             \
                int kj = dj / dilation;                                       \
                if (kj < 0 || kj >= K) continue;                              \
                int neighbor_idx = ki * K + kj;                               \
                int da_idx = (((b*heads+h)*height+i)*width+j)*L               \
                             + neighbor_idx;                                  \
                int q_idx = (((b*heads+h)*height+i)*width+j)*dim + d;         \
                sum += d_attn[da_idx] * query[q_idx];                         \
            }                                                                 \
        }                                                                     \
        int out_idx = (((b*heads+h)*height+vi)*width+vj)*dim + d;            \
        out[out_idx] = sum;                                                   \
    }                                                                         \
}

DEFINE_2D_QK_BWD_DK_KERNEL(natten2d_dk_k3, 3, 1)
DEFINE_2D_QK_BWD_DK_KERNEL(natten2d_dk_k5, 5, 2)
DEFINE_2D_QK_BWD_DK_KERNEL(natten2d_dk_k7, 7, 3)

// ---------------------------------------------------------------------------
// 2D QK backward: d_rpb (reduction over batch and spatial)
// Grid: (2*K-1, 2*K-1, H)
// ---------------------------------------------------------------------------

#define DEFINE_2D_QK_BWD_DRPB_KERNEL(KNAME, KSIZE, NH_VAL)                    \
kernel void KNAME(                                                            \
    device const float* d_attn   [[buffer(0)]],                               \
    device float*       out      [[buffer(1)]],                               \
    constant int&       batch_size [[buffer(2)]],                             \
    constant int&       heads    [[buffer(3)]],                               \
    constant int&       height   [[buffer(4)]],                               \
    constant int&       width    [[buffer(5)]],                               \
    constant int&       dilation [[buffer(6)]],                               \
    uint3 gid [[thread_position_in_grid]])                                    \
{                                                                             \
    const int K = KSIZE;                                                      \
    const int NH = NH_VAL;                                                    \
    const int rpb_h = 2 * K - 1;                                              \
    const int rpb_w = 2 * K - 1;                                              \
    int h = gid.z;                                                            \
    int ri = gid.y;                                                           \
    int rj = gid.x;                                                           \
    if (h >= heads || ri >= rpb_h || rj >= rpb_w) return;                     \
    float sum = 0.0f;                                                         \
    for (int b = 0; b < batch_size; b++) {                                    \
        for (int i = 0; i < height; i++) {                                    \
            int ni = get_window_start(i, height, K, NH, dilation);            \
            int ei = get_window_end(ni, height, K, dilation);                 \
            int pi = get_pb_start(i, height, K, NH, dilation);                \
            int ki = ri - pi;                                                 \
            if (ki < 0 || ki >= K) continue;                                  \
            int key_i = ni + ki * dilation;                                   \
            if (key_i < 0 || key_i >= ei) continue;                           \
            for (int j = 0; j < width; j++) {                                 \
                int nj = get_window_start(j, width, K, NH, dilation);         \
                int ej = get_window_end(nj, width, K, dilation);              \
                int pj = get_pb_start(j, width, K, NH, dilation);             \
                int kj = rj - pj;                                             \
                if (kj < 0 || kj >= K) continue;                              \
                int key_j = nj + kj * dilation;                               \
                if (key_j < 0 || key_j >= ej) continue;                       \
                int neighbor_idx = ki * K + kj;                               \
                int da_idx = (((b*heads+h)*height+i)*width+j)*(K*K)           \
                             + neighbor_idx;                                  \
                sum += d_attn[da_idx];                                        \
            }                                                                 \
        }                                                                     \
    }                                                                         \
    int out_idx = (h * rpb_h + ri) * rpb_w + rj;                             \
    out[out_idx] = sum;                                                       \
}

DEFINE_2D_QK_BWD_DRPB_KERNEL(natten2d_drpb_k3, 3, 1)
DEFINE_2D_QK_BWD_DRPB_KERNEL(natten2d_drpb_k5, 5, 2)
DEFINE_2D_QK_BWD_DRPB_KERNEL(natten2d_drpb_k7, 7, 3)

// ---------------------------------------------------------------------------
// 2D AV backward: d_attn
// Grid: (width, height, B*H)
// ---------------------------------------------------------------------------

#define DEFINE_2D_AV_BWD_DATTN_KERNEL(KNAME, KSIZE, NH_VAL)                   \
kernel void KNAME(                                                            \
    device const float* d_out    [[buffer(0)]],                               \
    device const float* value    [[buffer(1)]],                               \
    device float*       out      [[buffer(2)]],                               \
    constant int&       batch_size [[buffer(3)]],                             \
    constant int&       heads    [[buffer(4)]],                               \
    constant int&       height   [[buffer(5)]],                               \
    constant int&       width    [[buffer(6)]],                               \
    constant int&       dim      [[buffer(7)]],                               \
    constant int&       dilation [[buffer(8)]],                               \
    uint3 gid [[thread_position_in_grid]])                                    \
{                                                                             \
    const int K = KSIZE;                                                      \
    const int NH = NH_VAL;                                                    \
    const int L = K * K;                                                      \
    int b = gid.z / heads;                                                    \
    int h = gid.z % heads;                                                    \
    int i = gid.y;                                                            \
    int j = gid.x;                                                            \
    if (b >= batch_size || h >= heads || i >= height || j >= width) return;    \
    int ni = get_window_start(i, height, K, NH, dilation);                    \
    int nj = get_window_start(j, width,  K, NH, dilation);                    \
    int ei = get_window_end(ni, height, K, dilation);                         \
    int ej = get_window_end(nj, width,  K, dilation);                         \
    int neighbor_idx = 0;                                                     \
    for (int ki = 0; ki < K; ki++) {                                          \
        for (int kj = 0; kj < K; kj++) {                                     \
            int val_i = ni + ki * dilation;                                   \
            int val_j = nj + kj * dilation;                                   \
            float sum = 0.0f;                                                 \
            if (val_i >= 0 && val_i < ei &&                                   \
                val_j >= 0 && val_j < ej) {                                   \
                for (int d = 0; d < dim; d++) {                               \
                    int do_idx = (((b*heads+h)*height+i)*width+j)*dim + d;    \
                    int v_idx = (((b*heads+h)*height+val_i)*width+val_j)      \
                                *dim + d;                                     \
                    sum += d_out[do_idx] * value[v_idx];                      \
                }                                                             \
            }                                                                 \
            int out_idx = (((b*heads+h)*height+i)*width+j)*L                  \
                          + neighbor_idx;                                     \
            out[out_idx] = sum;                                               \
            neighbor_idx++;                                                   \
        }                                                                     \
    }                                                                         \
}

DEFINE_2D_AV_BWD_DATTN_KERNEL(natten2d_dattn_k3, 3, 1)
DEFINE_2D_AV_BWD_DATTN_KERNEL(natten2d_dattn_k5, 5, 2)
DEFINE_2D_AV_BWD_DATTN_KERNEL(natten2d_dattn_k7, 7, 3)

// ---------------------------------------------------------------------------
// 2D AV backward: d_value (reduction over all query positions)
// Grid: (width, height, B*H)
// ---------------------------------------------------------------------------

#define DEFINE_2D_AV_BWD_DVAL_KERNEL(KNAME, KSIZE, NH_VAL)                    \
kernel void KNAME(                                                            \
    device const float* d_out    [[buffer(0)]],                               \
    device const float* attn     [[buffer(1)]],                               \
    device float*       out      [[buffer(2)]],                               \
    constant int&       batch_size [[buffer(3)]],                             \
    constant int&       heads    [[buffer(4)]],                               \
    constant int&       height   [[buffer(5)]],                               \
    constant int&       width    [[buffer(6)]],                               \
    constant int&       dim      [[buffer(7)]],                               \
    constant int&       dilation [[buffer(8)]],                               \
    uint3 gid [[thread_position_in_grid]])                                    \
{                                                                             \
    const int K = KSIZE;                                                      \
    const int NH = NH_VAL;                                                    \
    const int L = K * K;                                                      \
    int b = gid.z / heads;                                                    \
    int h = gid.z % heads;                                                    \
    int vi = gid.y;                                                           \
    int vj = gid.x;                                                           \
    if (b >= batch_size || h >= heads || vi >= height || vj >= width) return;  \
    for (int d = 0; d < dim; d++) {                                           \
        float sum = 0.0f;                                                     \
        for (int i = 0; i < height; i++) {                                    \
            int ni = get_window_start(i, height, K, NH, dilation);            \
            int ei = get_window_end(ni, height, K, dilation);                 \
            if (vi < ni || vi >= ei) continue;                                \
            int di = vi - ni;                                                 \
            if (di % dilation != 0) continue;                                 \
            int ki = di / dilation;                                           \
            if (ki < 0 || ki >= K) continue;                                  \
            for (int j = 0; j < width; j++) {                                 \
                int nj = get_window_start(j, width, K, NH, dilation);         \
                int ej = get_window_end(nj, width, K, dilation);              \
                if (vj < nj || vj >= ej) continue;                            \
                int dj = vj - nj;                                             \
                if (dj % dilation != 0) continue;                             \
                int kj = dj / dilation;                                       \
                if (kj < 0 || kj >= K) continue;                              \
                int neighbor_idx = ki * K + kj;                               \
                int do_idx = (((b*heads+h)*height+i)*width+j)*dim + d;        \
                int attn_idx = (((b*heads+h)*height+i)*width+j)*L             \
                               + neighbor_idx;                                \
                sum += d_out[do_idx] * attn[attn_idx];                        \
            }                                                                 \
        }                                                                     \
        int out_idx = (((b*heads+h)*height+vi)*width+vj)*dim + d;            \
        out[out_idx] = sum;                                                   \
    }                                                                         \
}

DEFINE_2D_AV_BWD_DVAL_KERNEL(natten2d_dval_k3, 3, 1)
DEFINE_2D_AV_BWD_DVAL_KERNEL(natten2d_dval_k5, 5, 2)
DEFINE_2D_AV_BWD_DVAL_KERNEL(natten2d_dval_k7, 7, 3)
"""
