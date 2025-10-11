import pytest
import torch
import sys

sys.path.append(".")

from fast_td3.actors.gnn.egnn import EGNN

from torch import nn

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

    # Generate segment_ids using EGNN
    egnn = EGNN(
        hidden_nf=128,
        out_node_nf=1,
        in_edge_nf=64,
        device=device,
        batch_size=8192,
        n_layers=4,
        env_name="h1-push-v0", 
        robot="h1",
        act_fn=nn.ReLU()
    )

    return egnn.get_cached_edges(8192)


class TestIndex:
    def test_generated_index(self, test_data):
        """Test that the generated segment_ids are valid."""
        indexes = test_data

        src, _ = indexes

        assert len(src) % 8192 == 0
        assert src.max() == 8192 * 19 - 1

        print(src[:len(src)//8192])

        # assert edge_attr.sum() == 8192 * 38 # there is in total 38 edges from a join to object and other way around
        # assert node_attr.sum() == 8192