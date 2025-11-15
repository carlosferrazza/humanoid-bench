"""
Training utility functions for FastTD3.

This module contains helper functions for training, including:
- Gradient statistics monitoring
- Actor creation factory function
"""

import torch


def print_gradient_stats(model, model_name, step, detailed=False):
    """
    Print gradient statistics for monitoring training.
    
    Args:
        model: The neural network model
        model_name: Name of the model (e.g., "Actor", "Critic")
        step: Current training step
        detailed: If True, print per-parameter stats; if False, print layer group summary
    """
    print(f"\n{'='*80}")
    print(f"{model_name} Gradients at Step {step}")
    print(f"{'='*80}")
    
    # Count parameters with/without gradients
    total_params = sum(1 for _ in model.parameters())
    params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    params_without_grad = total_params - params_with_grad
    
    if params_without_grad > 0:
        print(f"⚠️  WARNING: {params_without_grad}/{total_params} parameters have no gradients!")
    
    if detailed:
        # Print detailed per-parameter gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()
                grad_max = param.grad.abs().max().item()
                grad_min = param.grad.abs().min().item()
                print(f"  {name:60s}")
                print(f"    norm: {grad_norm:10.6f} | mean: {grad_mean:10.6f} | std: {grad_std:10.6f}")
                print(f"    max:  {grad_max:10.6f} | min:  {grad_min:10.6f}")
    else:
        # Group gradients by layer type for EGNN models
        layer_groups = {}
        zero_grad_params = []
        
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            
            # Check for zero/near-zero gradients
            if param.grad.abs().max().item() < 1e-8:
                zero_grad_params.append(name)
                
            # Categorize parameters with improved logic
            if "embedding_in" in name or "embedding_out" in name:
                group = "Embeddings"
            elif "joint_embedding" in name or "object_embedding" in name:
                group = "Input Embeddings"
            elif "gnn.layers" in name or "model.layers" in name:
                # Extract layer number for EGNN layers
                parts = name.split(".")
                layer_idx = next((i for i, p in enumerate(parts) if p == "layers"), None)
                layer_num = parts[layer_idx + 1] if layer_idx and layer_idx + 1 < len(parts) else "?"
                
                if "edge_mlp" in name:
                    group = f"L{layer_num}_EdgeMLP"
                elif "coord_mlp" in name:
                    group = f"L{layer_num}_CoordMLP"
                elif "node_mlp" in name:
                    group = f"L{layer_num}_NodeMLP"
                elif "att_mlp" in name:
                    group = f"L{layer_num}_AttentionMLP"
                else:
                    group = f"L{layer_num}_Other"
            elif "qf" in name:  # For critic Q-functions
                if "qf1" in name:
                    group = "Critic_QF1"
                elif "qf2" in name:
                    group = "Critic_QF2"
                else:
                    group = "Critic_Q"
            elif "fc" in name or ("linear" in name and "qf" not in name):
                group = "Linear Layers"
            else:
                group = f"Other ({name.split('.')[0]})"
            
            if group not in layer_groups:
                layer_groups[group] = []
            
            layer_groups[group].append(param.grad)
        
        # Print summary statistics for each group
        print(f"{'Layer Group':<30s} | {'Norm':>10s} | {'Mean':>10s} | {'Std':>10s} | {'Max':>10s}")
        print("-" * 80)
        
        vanishing_groups = []
        for group in sorted(layer_groups.keys()):
            grads = layer_groups[group]
            # Concatenate all gradients in the group
            all_grads = torch.cat([g.flatten() for g in grads])
            
            norm = all_grads.norm().item()
            mean = all_grads.mean().item()
            std = all_grads.std().item()
            max_val = all_grads.abs().max().item()
            
            # Flag vanishing gradients
            if max_val < 1e-4:
                vanishing_groups.append(group)
                print(f"{group:<30s} | {norm:10.4f} | {mean:10.6f} | {std:10.6f} | {max_val:10.6f} ⚠️")
            else:
                print(f"{group:<30s} | {norm:10.4f} | {mean:10.6f} | {std:10.6f} | {max_val:10.6f}")
        
        if vanishing_groups:
            print("-" * 80)
            print(f"⚠️  VANISHING GRADIENTS detected in: {', '.join(vanishing_groups)}")
        
        if zero_grad_params:
            print("-" * 80)
            print(f"🔴 ZERO GRADIENTS in {len(zero_grad_params)} parameters: {zero_grad_params[:3]}...")
    
    print(f"{'='*80}\n")


def collect_gradient_stats(model, model_name="model"):
    """
    Collect gradient norm statistics for wandb logging.
    
    Args:
        model: The neural network model
        model_name: Name of the model (e.g., "actor", "critic")
    
    Returns:
        dict: Dictionary of gradient norms for wandb logging
    """
    stats = {}
    
    # Group gradients by layer type
    layer_groups = {}
    
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        
        # Categorize parameters
        if "embedding_in" in name or "embedding_out" in name:
            group = "embeddings"
        elif "joint_embedding" in name or "object_embedding" in name or "object_mlp" in name:
            group = "input_embeddings"
        elif "layers" in name:
            # Extract layer number
            parts = name.split(".")
            layer_idx = next((i for i, p in enumerate(parts) if p == "layers"), None)
            layer_num = parts[layer_idx + 1] if layer_idx and layer_idx + 1 < len(parts) else "?"
            
            if "edge_mlp" in name:
                group = f"layer{layer_num}_edge"
            elif "coord_mlp" in name:
                group = f"layer{layer_num}_coord"
            elif "node_mlp" in name:
                group = f"layer{layer_num}_node"
            else:
                group = f"layer{layer_num}_other"
        elif "global_aggregation" in name:
            group = "global_aggregation"
        elif "skip_proj" in name:
            group = "skip_projection"
        else:
            group = "other"
        
        if group not in layer_groups:
            layer_groups[group] = []
        
        layer_groups[group].append(param.grad)
    
    # Compute gradient norm for each group
    for group, grads in layer_groups.items():
        all_grads = torch.cat([g.flatten() for g in grads])
        stats[f"{model_name}/grad_{group}_norm"] = all_grads.norm().item()
    
    # Overall gradient norm
    all_params_grads = [p.grad.flatten() for p in model.parameters() if p.grad is not None]
    if all_params_grads:
        all_grads = torch.cat(all_params_grads)
        stats[f"{model_name}/grad_overall_norm"] = all_grads.norm().item()
    
    return stats


def create_actor(
    actor_type,
    n_obs,
    n_act,
    num_envs,
    batch_size,
    device,
    init_scale,
    env_name,
    model_kwargs,
    actor_hidden_dim=None,
):
    """
    Helper function to create an actor based on the specified type.

    Args:
        actor_type (str): Type of actor ("egnn", "mlp", "mpnn", "hepi")
        n_obs (int): Number of observations
        n_act (int): Number of actions
        num_envs (int): Number of environments
        batch_size (int): Batch size
        device (torch.device): Device to place the actor on
        init_scale (float): Initialization scale
        env_name (str): Name of the environment
        model_kwargs (dict): Additional model parameters
        actor_hidden_dim (int, optional): Hidden dimension for MLP actor

    Returns:
        Actor: The created actor instance

    Raises:
        ValueError: If actor_type is not supported
    """
    from fast_td3.actors import (
        ActorEGNN,
        Actor,
        ActorMPNN,
        ActorHEPI,
        ActorAEGNN,
        ActorPONITA,
        ActorHEGNN,
    )
    
    if actor_type == "egnn":
        return ActorEGNN(
            n_obs=n_obs,
            n_act=n_act,
            num_envs=num_envs,
            batch_size=batch_size,
            device=device,
            init_scale=init_scale,
            env_name=env_name,
            **model_kwargs,
        )
    elif actor_type == "mlp":
        return Actor(
            n_obs=n_obs,
            n_act=n_act,
            num_envs=num_envs,
            device=device,
            init_scale=init_scale,
            hidden_dim=actor_hidden_dim,
        )
    elif actor_type == "mpnn":
        return ActorMPNN(
            n_obs=n_obs,
            n_act=n_act,
            num_envs=num_envs,
            batch_size=batch_size,
            device=device,
            **model_kwargs,
        )
    elif actor_type == "hepi":
        return ActorHEPI(
            n_obs=n_obs,
            n_act=n_act,
            num_envs=num_envs,
            batch_size=batch_size,
            device=device,
            **model_kwargs,
        )
    elif actor_type == "aegnn":
        return ActorAEGNN(
            n_obs=n_obs,
            n_act=n_act,
            num_envs=num_envs,
            batch_size=batch_size,
            device=device,
            init_scale=init_scale,
            **model_kwargs,
        )
    elif actor_type == "ponita":
        return ActorPONITA(
            n_obs=n_obs,
            n_act=n_act,
            num_envs=num_envs,
            batch_size=batch_size,
            device=device,
            robot="h1",
            **model_kwargs,
        )
    elif actor_type == "hegnn":
        return ActorHEGNN(
            n_obs=n_obs,
            n_act=n_act,
            num_envs=num_envs,
            batch_size=batch_size,
            device=device,
            init_scale=init_scale,
            env_name=env_name,
            **model_kwargs,
        )
    else:
        raise ValueError(
            f"Unsupported actor type: {actor_type}. Supported types are: egnn, mlp, mpnn, hepi, aegnn, ponita, hegnn"
        )
