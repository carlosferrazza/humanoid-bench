import pytest
import torch
import time
import sys
from typing import Callable

sys.path.append(".")

from fast_td3.actors.gnn.egnn import (
    unsorted_segment_mean as unsorted_segment_mean_new,
    unsorted_segment_sum as unsorted_segment_sum_new,
    EGNN
)

from torch import nn


def unsorted_segment_sum_old(data, segment_ids, num_segments):
    """Reference implementation using scatter_add for comparison."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean_old(data, segment_ids, num_segments):
    """Reference implementation using scatter_add for comparison."""
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)



@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


@pytest.fixture
def test_data(device):
    """Generate test data for segment aggregation functions."""
    torch.manual_seed(42)  # Fixed seed for reproducible tests
    N, D, num_segments = 933888, 64, 163840
    
    data = torch.randn(N, D, device=device, dtype=torch.float32)
    
    # Generate segment_ids using EGNN
    egnn = EGNN(
        in_node_nf=D,
        hidden_nf=128,
        out_node_nf=D,
        in_edge_nf=64,
        device=device,
        batch_size=8192,
        n_layers=4,
        env_name="h1-push-v0", 
        robot="h1",
        act_fn=nn.ReLU()
    )
    segment_ids = egnn.generate_index(8192, device)[0][0]
    
    return data, segment_ids, num_segments


@pytest.fixture
def small_test_data(device):
    """Generate smaller test data for edge case testing."""
    torch.manual_seed(42)
    N, D, num_segments = 100, 8, 20
    
    data = torch.randn(N, D, device=device, dtype=torch.float32)
    segment_ids = torch.randint(0, num_segments, (N,), device=device)
    
    return data, segment_ids, num_segments


def benchmark_function(fn: Callable, data: torch.Tensor, segment_ids: torch.Tensor, 
                      num_segments: int, device: str, iters: int = 100) -> float:
    """Benchmark a function and return average time per call in milliseconds."""
    # Warm-up
    for _ in range(5):
        _ = fn(data, segment_ids, num_segments)
    
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
    elif device == "mps":
        # MPS synchronization and timing
        torch.mps.synchronize()
        t0 = time.time()
        for _ in range(iters):
            fn(data, segment_ids, num_segments)
        torch.mps.synchronize()
        return (time.time() - t0) * 1000 / iters  # ms per call
    else:
        t0 = time.time()
        for _ in range(iters):
            fn(data, segment_ids, num_segments)
        return (time.time() - t0) * 1000 / iters  # ms per call


class TestSegmentAggregation:
    """Test suite for segment aggregation functions."""
    
    def test_unsorted_segment_sum_correctness(self, small_test_data):
        """Test that new sum implementation matches reference implementation."""
        data, segment_ids, num_segments = small_test_data
        
        result_old = unsorted_segment_sum_old(data, segment_ids, num_segments)
        result_new = unsorted_segment_sum_new(data, segment_ids, num_segments)
        
        assert torch.allclose(result_old, result_new, atol=1e-6), \
            "New sum implementation doesn't match reference"
    
    def test_unsorted_segment_mean_correctness(self, small_test_data):
        """Test that new mean implementation matches reference implementation."""
        data, segment_ids, num_segments = small_test_data
        
        result_old = unsorted_segment_mean_old(data, segment_ids, num_segments)
        result_new = unsorted_segment_mean_new(data, segment_ids, num_segments)
        
        assert torch.allclose(result_old, result_new, atol=1e-6), \
            "New mean implementation doesn't match reference"
    
    def test_unsorted_segment_sum_large_data_correctness(self, test_data):
        """Test correctness on large realistic data."""
        data, segment_ids, num_segments = test_data
        
        result_old = unsorted_segment_sum_old(data, segment_ids, num_segments)
        result_new = unsorted_segment_sum_new(data, segment_ids, num_segments)
        
        assert torch.allclose(result_old, result_new, rtol=1e-5, atol=1e-5), \
            "New sum implementation doesn't match reference on large data"
    
    def test_unsorted_segment_mean_large_data_correctness(self, test_data):
        """Test correctness on large realistic data."""
        data, segment_ids, num_segments = test_data
        
        result_old = unsorted_segment_mean_old(data, segment_ids, num_segments)
        result_new = unsorted_segment_mean_new(data, segment_ids, num_segments)
        
        assert torch.allclose(result_old, result_new, atol=1e-6), \
            "New mean implementation doesn't match reference on large data"
    
    def test_empty_segments(self, device):
        """Test behavior with empty segments."""
        data = torch.randn(10, 4, device=device)
        segment_ids = torch.tensor([0, 0, 2, 2, 2, 4, 4, 4, 4, 4], device=device)
        num_segments = 6  # segments 1, 3, 5 will be empty
        
        result_sum_old = unsorted_segment_sum_old(data, segment_ids, num_segments)
        result_sum_new = unsorted_segment_sum_new(data, segment_ids, num_segments)
        
        result_mean_old = unsorted_segment_mean_old(data, segment_ids, num_segments)
        result_mean_new = unsorted_segment_mean_new(data, segment_ids, num_segments)
        
        assert torch.allclose(result_sum_old, result_sum_new, atol=1e-6)
        assert torch.allclose(result_mean_old, result_mean_new, atol=1e-6)
        
        # Check that empty segments have zero sum
        assert torch.allclose(result_sum_new[1], torch.zeros_like(result_sum_new[1]))
        assert torch.allclose(result_sum_new[3], torch.zeros_like(result_sum_new[3]))
        assert torch.allclose(result_sum_new[5], torch.zeros_like(result_sum_new[5]))
    
    def test_single_element_segments(self, device):
        """Test behavior with single-element segments."""
        data = torch.randn(5, 3, device=device)
        segment_ids = torch.tensor([0, 1, 2, 3, 4], device=device)  # Each element in its own segment
        num_segments = 5
        
        result_sum_old = unsorted_segment_sum_old(data, segment_ids, num_segments)
        result_sum_new = unsorted_segment_sum_new(data, segment_ids, num_segments)
        
        result_mean_old = unsorted_segment_mean_old(data, segment_ids, num_segments)
        result_mean_new = unsorted_segment_mean_new(data, segment_ids, num_segments)
        
        assert torch.allclose(result_sum_old, result_sum_new, atol=1e-6)
        assert torch.allclose(result_mean_old, result_mean_new, atol=1e-6)
        
        # For single elements, sum and mean should be the same as original data
        assert torch.allclose(result_sum_new, data, atol=1e-6)
        assert torch.allclose(result_mean_new, data, atol=1e-6)
    
    def test_all_same_segment(self, device):
        """Test behavior when all elements belong to the same segment."""
        data = torch.randn(10, 4, device=device)
        segment_ids = torch.zeros(10, dtype=torch.long, device=device)
        num_segments = 1
        
        result_sum_old = unsorted_segment_sum_old(data, segment_ids, num_segments)
        result_sum_new = unsorted_segment_sum_new(data, segment_ids, num_segments)
        
        result_mean_old = unsorted_segment_mean_old(data, segment_ids, num_segments)
        result_mean_new = unsorted_segment_mean_new(data, segment_ids, num_segments)
        
        assert torch.allclose(result_sum_old, result_sum_new, atol=1e-6)
        assert torch.allclose(result_mean_old, result_mean_new, atol=1e-6)
        
        # Sum should equal sum of all data
        expected_sum = torch.sum(data, dim=0, keepdim=True)
        assert torch.allclose(result_sum_new, expected_sum, atol=1e-6)
        
        # Mean should equal mean of all data
        expected_mean = torch.mean(data, dim=0, keepdim=True)
        assert torch.allclose(result_mean_new, expected_mean, atol=1e-6)

class TestPerformance:
    """Performance regression tests."""
    
    def test_performance_improvement_sum(self, test_data, device):
        """Test that new sum implementation is faster than old one."""
        data, segment_ids, num_segments = test_data
        
        # Skip performance tests on CPU for faster CI
        if device == "cpu":
            pytest.skip("Performance tests skipped on CPU")

        print(segment_ids[:100])
        
        time_old = benchmark_function(
            unsorted_segment_sum_old, data, segment_ids, num_segments, device, iters=50
        )
        time_new = benchmark_function(
            unsorted_segment_sum_new, data, segment_ids, num_segments, device, iters=50
        )
        
        print("\nSum performance comparison:")
        print(f"Old implementation: {time_old:.6f} ms per call")
        print(f"New implementation: {time_new:.6f} ms per call")
        print(f"Speedup: {time_old / time_new:.2f}x")
        
        # The new implementation should be at least as fast as the old one
        # (allowing for some variance in measurement)
        assert time_new < time_old, \
            f"New implementation is slower: {time_new:.6f} ms vs {time_old:.6f} ms"
    
    def test_performance_improvement_mean(self, test_data, device):
        """Test that new mean implementation is faster than old one."""
        data, segment_ids, num_segments = test_data
        
        # Skip performance tests on CPU for faster CI
        if device == "cpu":
            pytest.skip("Performance tests skipped on CPU")
        
        time_new = benchmark_function(
            unsorted_segment_mean_new, data, segment_ids, num_segments, device, iters=50
        )
        time_old = benchmark_function(
            unsorted_segment_mean_old, data, segment_ids, num_segments, device, iters=50
        )
        
        print("\nMean performance comparison:")
        print(f"Old implementation: {time_old:.6f} ms per call")
        print(f"New implementation: {time_new:.6f} ms per call")
        print(f"Speedup: {time_old / time_new:.2f}x")
        
        # The new implementation should be at least as fast as the old one
        # (allowing for some variance in measurement)
        assert time_new < time_old,\
            f"New implementation is slower: {time_new:.6f} ms vs {time_old:.6f} ms"


if __name__ == "__main__":
    # Allow running as a script for quick testing
    pytest.main([__file__, "-v", "--tb=short"])
