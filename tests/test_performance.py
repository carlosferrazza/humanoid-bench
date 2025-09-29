import torch
import time

import sys
sys.path.append("../fast_td3")

from fast_td3.robots.H1 import H1

def generate_index(batch_size: int, device="cuda"):
        src, _ = zip(*H1().joint_connections_with_object)
        src = torch.tensor(src, dtype=torch.long, device=device)

        offsets = torch.arange(batch_size, device=device) * 20
        src_batch = (src.unsqueeze(0) + offsets.unsqueeze(1)).flatten().to(device)

        return src_batch

def unsorted_segment_sum_old(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean_old(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)

# ------------------------
# New fused implementation
# ------------------------
def unsorted_segment_mean_new(data, segment_ids, num_segments):
    """Compute segment mean faster than the old two-pass scatter version.

    Strategy:
    - Use index_add_ with original 1D segment_ids (no (N,D) expansion) to accumulate sums.
    - Use torch.bincount to get counts per segment (much cheaper than a second scatter_add_).
    - Divide sums by counts (clamped to 1 to avoid div-by-zero for empty segments).
    This avoids:
        1) Expanding segment_ids to (N, D) which creates large temporary tensors.
        2) A second scatter pass for the counts.
    """
    
    num_feats = data.size(1)
    sums = data.new_zeros((num_segments, num_feats))  # [S, D]
    # One pass accumulation of sums
    sums.index_add_(0, segment_ids, data)

    # Counts per segment (no expansion). bincount is efficient on GPU for large N.
    counts = torch.bincount(segment_ids, minlength=num_segments).clamp_min(1).unsqueeze(-1)
    return sums / counts




device = "cuda"
N, D, num_segments = 933888, 64, 163840

torch.manual_seed(0)
data = torch.randn(N, D, device=device, dtype=torch.float32)
segment_ids = generate_index(8192)
print(segment_ids.shape)
assert segment_ids.shape[0] == N

# Warm-up (GPU kernels lazy init etc.)
for fn in [unsorted_segment_mean_old, unsorted_segment_mean_new]:
    _ = fn(data, segment_ids, num_segments)

# Correctness check
out_old = unsorted_segment_mean_old(data, segment_ids, num_segments)
out_new = unsorted_segment_mean_new(data, segment_ids, num_segments)
print("Correctness check:", torch.allclose(out_old, out_new, atol=1e-6))

# Timing utility
def benchmark(fn, iters=1000):
    if device == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(iters):
            fn(data, segment_ids, num_segments)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters  # ms per call
    else:
        t0 = time.time()
        for _ in range(iters):
            fn(data, segment_ids, num_segments)
        return (time.time() - t0) * 1000 / iters  # ms per call

# Run benchmarks
t_old = benchmark(unsorted_segment_mean_old)
t_new = benchmark(unsorted_segment_mean_new)

print(f"Old two-pass version: {t_old:.6f} ms per call")
print(f"New one-pass version: {t_new:.6f} ms per call")
