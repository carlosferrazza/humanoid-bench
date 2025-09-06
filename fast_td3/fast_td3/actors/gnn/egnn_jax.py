import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.nn.initializers import glorot_uniform, uniform
from jax.tree_util import tree_map


def unsorted_segment_sum(data, segment_ids, num_segments):
    return jax.ops.segment_sum(data, segment_ids, num_segments)


def unsorted_segment_mean(data, segment_ids, num_segments):
    seg_sum = jax.ops.segment_sum(data, segment_ids, num_segments)
    seg_count = jax.ops.segment_sum(jnp.ones_like(data), segment_ids, num_segments)
    seg_count = jnp.maximum(seg_count, 1)  # Avoid 0 division
    return seg_sum / seg_count


def xavier_init(gain):
    def init(key, shape, dtype):
        bound = gain * jnp.sqrt(6.0 / (shape[0] + shape[1]))
        return jax.random.uniform(key, shape, dtype, -bound, bound)

    return init


class E_GCL(nn.Module):
    """
    E(n) Equivariant Message Passing Layer
    """

    hidden_nf: int
    act_fn: callable
    velocity: bool = False

    def edge_model(self, edge_index, h, coord, edge_attr):
        row, col = edge_index
        source, target = h[row], h[col]
        radial = self.coord2radial(edge_index, coord)

        edge_mlp = nn.Sequential(
            [
                nn.Dense(self.hidden_nf),
                self.act_fn,
                nn.Dense(self.hidden_nf),
                self.act_fn,
            ]
        )

        out = jnp.concatenate([source, target, radial, edge_attr], axis=1)

        return edge_mlp(out)

    def node_model(self, edge_index, edge_attr, x):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.shape[0])

        node_mlp = nn.Sequential(
            [nn.Dense(self.hidden_nf), self.act_fn, nn.Dense(self.hidden_nf)]
        )

        agg = jnp.concatenate([x, agg], axis=1)
        out = node_mlp(agg)

        # TODO do we need to add x to out? to update it
        return out, agg

    def coord_model(self, edge_index, edge_feat, coord):
        row, col = edge_index
        coord_mlp = nn.Sequential(
            [
                nn.Dense(self.hidden_nf),
                self.act_fn,
                nn.Dense(1, kernel_init=xavier_init(gain=0.001)),
            ]
        )

        coord_out = coord_mlp(edge_feat)
        trans = (coord[row] - coord[col]) * coord_out

        agg = unsorted_segment_mean(trans, row, num_segments=coord.shape[0])

        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        senders, receivers = edge_index
        coord_i, coord_j = coord[senders], coord[receivers]
        distance = jnp.sum((coord_i - coord_j) ** 2, axis=1, keepdims=True)
        return distance

    def coord_vel_model(self, coord, h, vel):
        coord_mlp_vel = nn.Sequential(
            [nn.Dense(self.hidden_nf), self.act_fn, nn.Dense(1)]
        )

        coord += coord_mlp_vel(h) * vel
        return coord

    @nn.compact
    def __call__(self, h, edge_index, coord, vel=None, edge_attr=None):
        m_ij = self.edge_model(edge_index, h, coord, edge_attr)
        h, agg = self.node_model(edge_index, m_ij, h)
        coord = self.coord_model(edge_index, m_ij, coord)
        if self.velocity:
            coord = self.coord_vel_model(coord, h, vel)
        return h, coord, m_ij


class E_GCL_OG(nn.Module):
    """
    E(n) Equivariant Message Passing Layer - Reproduction of initial
    """

    input_nf: int
    output_nf: int
    hidden_nf: int
    edges_in_d: int = 0
    nodes_attr_dim: int = 0
    act_fn: callable = nn.relu
    recurrent: bool = True
    coords_weight: float = 1.0
    attention: bool = False

    def setup(self):
        input_edge = self.input_nf * 2
        self.edge_mlp = nn.Sequential(
            [
                nn.Dense(
                    self.hidden_nf, kernel_init=glorot_uniform(), bias_init=uniform()
                ),
                self.act_fn,
                nn.Dense(
                    self.hidden_nf, kernel_init=glorot_uniform(), bias_init=uniform()
                ),
                self.act_fn,
            ]
        )

        self.node_mlp = nn.Sequential(
            [
                nn.Dense(
                    self.hidden_nf, kernel_init=glorot_uniform(), bias_init=uniform()
                ),
                self.act_fn,
                nn.Dense(
                    self.output_nf, kernel_init=glorot_uniform(), bias_init=uniform()
                ),
            ]
        )

        self.coord_mlp = nn.Sequential(
            [
                nn.Dense(
                    self.hidden_nf, kernel_init=glorot_uniform(), bias_init=uniform()
                ),
                self.act_fn,
                nn.Dense(1, use_bias=False),
            ]
        )

        if self.attention:
            self.att_mlp = nn.Sequential(
                [
                    nn.Dense(1, kernel_init=glorot_uniform(), bias_init=uniform()),
                    nn.sigmoid,
                ]
            )

    def edge_model(self, source, target, radial, edge_attr):
        # print(edge_attr)
        if edge_attr is None:
            out = jnp.concatenate([source, target, radial], axis=1)
        else:
            edge_attr = edge_attr.reshape(-1, edge_attr.shape[-1])
            out = jnp.concatenate(
                [source, target, radial.reshape(-1, 1), edge_attr], axis=1
            )
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = jax.ops.segment_sum(edge_attr, row, num_segments=x.shape[0])
        if node_attr is not None:
            agg = jnp.concatenate([x, agg, node_attr], axis=1)
        else:
            agg = jnp.concatenate([x, agg], axis=1)
        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, edge_mask):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat) * edge_mask
        trans = jnp.clip(trans, -100, 100)
        agg = jax.ops.segment_sum(trans, row, coord.shape[0])
        coord += agg * self.coords_weight
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = jnp.sum(coord_diff**2, axis=1, keepdims=True)
        # if self.norm_diff:
        #     norm = jnp.sqrt(radial) + 1
        #     coord_diff = coord_diff / norm
        return radial, coord_diff

    def coord_vel_model(self, coord, h, vel):
        coord_mlp_vel = nn.Sequential(
            [nn.Dense(self.hidden_nf), self.act_fn, nn.Dense(1)]
        )
        coord += coord_mlp_vel(h) * vel
        return coord

    def __call__(
        self,
        h,
        edge_index,
        coord,
        node_mask,
        edge_mask,
        edge_attr=None,
        node_attr=None,
        n_nodes=None,
    ):
        row, col = edge_index
        # print(coord)
        # print(f'coord shape {coord.shape}')
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)

        edge_feat = edge_feat * edge_mask

        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr


class EGNN_equiv(nn.Module):
    hidden_nf: int
    out_node_nf: int
    act_fn: callable = nn.relu
    n_layers: int = 4
    velocity: bool = False

    @nn.compact
    def __call__(self, h, x, edges, vel=None, edge_attr=None):
        h = nn.Dense(self.hidden_nf)(h)
        for i in range(self.n_layers):
            h, x, _ = E_GCL(
                self.hidden_nf,
                self.hidden_nf,
                self.hidden_nf,
                act_fn=self.act_fn,
                attention=True,
            )(
                h,
                edges,
                x,
                jnp.ones_like(x),
                jnp.ones((edges[0].shape[0], 1)),
                edge_attr=edge_attr,
            )
        h = nn.Dense(self.out_node_nf)(h)
        return h, x


class EGNN_QM9(nn.Module):
    hidden_nf: int
    out_node_nf: int
    act_fn: callable = nn.relu  # default activation function
    n_layers: int = 4
    attention: bool = False

    @nn.compact
    def __call__(self, h, x, edges, edge_attr, node_mask, edge_mask, n_nodes):
        h = nn.Dense(self.hidden_nf)(h)
        for i in range(self.n_layers):
            h, x, _ = E_GCL_OG(
                self.hidden_nf,
                self.hidden_nf,
                self.hidden_nf,
                act_fn=self.act_fn,
                attention=self.attention,
            )(h, edges, x, node_mask, edge_mask, edge_attr=None)
        h = h * node_mask  # Ensure node_mask is broadcasted correctly
        h = h.reshape(-1, n_nodes, self.hidden_nf)
        h = jnp.sum(h, axis=1)
        h = nn.Dense(self.out_node_nf)(h)
        h = jnp.squeeze(h, axis=-1)  # Squeeze the last dimension like in original repo
        return h, x


def preprocess_input(one_hot, charges, charge_power, charge_scale):
    charge_tensor = (charges[..., None] / charge_scale) ** jnp.arange(charge_power + 1)
    charge_tensor = charge_tensor.reshape(*charges.shape, -1)
    atom_scalars = (one_hot[..., None] * charge_tensor[..., None, :]).reshape(
        charges.shape[0], -1
    )
    return atom_scalars


def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)
    edges = [rows, cols]
    return edges


def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = jnp.ones(len(edges[0]) * batch_size, dtype=jnp.float32).reshape(-1, 1)
    edges = [
        jnp.array(edges[0]).astype(jnp.int32),
        jnp.array(edges[1]).astype(jnp.int32),
    ]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [jnp.concatenate(rows), jnp.concatenate(cols)]
    return edges, edge_attr


if __name__ == "__main__":
    # Dummy parameters
    batch_size = 8
    n_nodes = 4
    n_feat = 1
    x_dim = 3
    charge_power = 2
    charge_scale = 9

    # Dummy variables h, x and fully connected edges
    charges = jnp.array([0, 1, 2, 0, 1, 2, 0, 1])
    one_hot = jax.nn.one_hot(charges, num_classes=charge_power + 1)
    h = preprocess_input(one_hot, charges, charge_power, charge_scale)
    h = h.reshape(batch_size * n_nodes, -1)
    x = jnp.ones((batch_size * n_nodes, x_dim))
    edges, edge_attr = get_edges_batch(n_nodes, batch_size)

    rng = jax.random.PRNGKey(42)

    # Initialize EGNN with attention
    egnn = EGNN_QM9(hidden_nf=32, out_node_nf=1, attention=True)

    params = egnn.init(
        rng, h, x, edges, edge_attr, jnp.ones((batch_size * n_nodes, 1)), n_nodes
    )["params"]

    # Now you can use the model's `apply` method with these parameters
    output = egnn.apply(
        {"params": params},
        h,
        x,
        edges,
        edge_attr,
        jnp.ones((batch_size * n_nodes, 1)),
        n_nodes,
    )
    print(output)
