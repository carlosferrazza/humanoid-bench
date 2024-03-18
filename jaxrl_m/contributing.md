# Contributing

We discuss two key abstractions used heavily in this codebase: the use of `TrainState` and the expression of agents as `PytreeNodes` 

## Agents

In this codebase, we represent agents as PytreeNodes  (first-class Jax citizens), making them really easy to handle. The simplest working example we have in the codebase is probably `jaxrl_m/example_agents/discrete_bc.py`, so check that out for a concrete implementation.

The general structure of an Agent is as follows: it contains some number of neural networks, some set of configuration values, and has an update function that takes in a batch and returns a agent with updated parameters after performing some gradient update. Usually there's a `sample_actions` to sample from the resulting policy too.

```python
class Agent(flax.struct.PyTreeNode):
    value_function: TrainState
    policy: TrainState
    config: dict = nonpytree_field() # tells Jax to not look at this (usually contains discount factor / target update speed / other hyperparams)
    
    @jax.jit
    def update(self, batch: Batch):
        ...
        new_value_function = ...
        new_policy = ...
        info = {'loss': 100}
        new_agent = self.replace(value_function=value_function, policy=new_policy)
        return new_agent, info
    
    @jax.jit
    def sample_actions(self, observations, *, seed):
        actions = ...
        return actions
```

### Multiple Devices

Operating on multiple GPUs / TPUs is really easy! Check out the section at the bottom of the page as to how to accumulate gradients across all the GPUs. 


- `flax.jax_utils.replicate()`: replicates an object on all GPUs
- `jaxrl_m.common.common.shard_batch`: splits an batch evenly across all the GPUs
- `flax.jax_utils.unreplicate()` brings back to single GPU

```python
agent = ...
batch = ...

replicated_agent = replicate(agent)
replicated_agent, info = replicated_agent.update(shard_batch(batch))
info = unreplicate(info) # bring info back to single device


```
## TrainState 


The TrainState class (located at `jaxrl_mcommon.TrainState`) is a fork of Flax's TrainState class with some additional syntactic features for ease of use. 

The TrainState class combines a neural network module (`flax.linen.Module`) with a set of parameters for this network (alongside with potentially an optimizer)

### Creating a TrainState

```python
model_def = nn.Dense(10) # nn.Module
params = model_def.init(rng, x)['params'] # parameters for nn.Module
tx = optax.adam(1e-3)
model = TrainState.create(model_def, params, tx=tx)
```

### Running the Model

```python
model = TrainState.create(...)
y_pred = model(x)
```

In some cases, the neural network module may have several functions; for example, a VAE might have an `.encode(x)` function and a `.decode(z)` function. By default, the `__call__()` method is used, but this can be specified via an argument:

```python
z = model(x, method='encode')
x_pred = model(z, method='decode')
```

You can also run the model with a different set of parameters than that bound to the TrainState. This is most commonly done when taking the gradient with respect to model parameters. 

```python
y_pred = model(x, params=other_params)
```

```python
def loss(params):
    y_pred = model(x, params=params)
    return jnp.mean((y - y_pred) ** 2)

grads = jax.grad(loss)(model.params)
```

### Optimizing a TrainState

To update a model (that has a `tx`), we provide two convenience functions: `.apply_gradients` and `.apply_loss_fn`

`model.apply_gradients` takes in a set of gradients (same shape as parameters) and computes the new set of parameters using optax.

```python
def loss(params):
    y_pred = model(x, params=params)
    return jnp.mean((y - y_pred) ** 2)

grads = jax.grad(loss)(model.params)
new_model = model.apply_gradients(grads=grads)
```

`model.apply_loss_fn()` is a convenience method that both computes the gradients and runs `.apply_gradients()`.

```python
def loss(params):
    y_pred = model(x, params=params)
    return jnp.mean((y - y_pred) ** 2)

new_model = model.apply_loss_fn(loss_fn=loss)
```

If the model is being run across multiple GPUs / TPUs and we wish to aggregate gradients, this can be specified with the `pmap_axis` argument (you can always use jax.lax.pmean as an alternative):

```python
@functools.partial(jax.pmap, axis_name='pmap')
def update(model, x, y):
    def loss(params):
        y_pred = model(x, params=params)
        return jnp.mean((y - y_pred) ** 2)

    new_model = model.apply_loss_fn(loss_fn=loss, pmap_axis='pmap')
    return new_model
```



