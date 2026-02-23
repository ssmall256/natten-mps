"""
NATTEN Unfused Reference Implementation - Author-Compatible Shift Semantics

Based on NATTEN v0.17.5 shifted/clipped window semantics.
Uses get_window_start/get_window_end/get_pb_start helpers for boundary handling.
"""


def get_window_start(index, length, kernel_size, neighborhood_size, dilation):
    """Compute window start with phase alignment."""
    dilation_idx = index % dilation
    index_pdp = index // dilation
    length_pdp = (length + dilation - 1) // dilation
    num_padded = (length_pdp * dilation) - length
    length_pdp -= 1 if (dilation_idx >= dilation - num_padded) else 0

    start_idx = max(index_pdp - neighborhood_size, 0) + \
        ((index_pdp + neighborhood_size >= length_pdp) *
         (length_pdp - index_pdp - neighborhood_size - 1))

    return start_idx * dilation + dilation_idx


def get_window_end(start_index, length, kernel_size, dilation):
    """Compute window end (half-open range)."""
    return min(length, start_index + kernel_size * dilation)


def get_pb_start(index, length, kernel_size, neighborhood_size, dilation):
    """Compute position bias start index."""
    if dilation <= 1:
        return neighborhood_size + \
            ((index < neighborhood_size) * (neighborhood_size - index)) + \
            ((index + neighborhood_size >= length) *
             (length - index - 1 - neighborhood_size))

    if index - neighborhood_size * dilation < 0:
        return kernel_size - 1 - (index // dilation)
    if index + neighborhood_size * dilation >= length:
        return (length - index - 1) // dilation
    return neighborhood_size
