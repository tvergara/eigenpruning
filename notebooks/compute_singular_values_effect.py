from transformer_lens import utils
from fancy_einsum import einsum
import einops
import torch
from notebooks.constants import PREACTIVATION_NAMES

def compute_aprox_effect(
    gradients,
    activations,
    singular_values,
    layers,
    parts
):
    effects = {}
    for layer in layers:
        for part in parts:
            grad = gradients[(layer, part)]
            decompositions = singular_values[(layer, part)]
            preactivation_name = PREACTIVATION_NAMES[part]
            preactivations = activations[utils.get_act_name(preactivation_name, layer)]

            u, s, v = decompositions
            rotated_activations = torch.roll(preactivations, shifts=1, dims=0)
            delta_activations = rotated_activations - preactivations

            if part == 'result':
                effect = calculate_importance_attn_out_component(
                    u, s, v,
                    delta_activations,
                    grad
                )
                effects[(layer, part)] = effect

            elif part in ['pre', 'mlp_out']:
                effect = calculate_importance_mlp_component(
                    u, s, v,
                    delta_activations,
                    grad
                )
                effects[(layer, part)] = effect
            elif part in ['k', 'q', 'v']:
                effect = calculate_importance_kqv_component(
                    u, s, v,
                    delta_activations,
                    grad
                )
                effects[(layer, part)] = effect
            else:
                raise ValueError('part not recognized')

    return effects


def calculate_importance_mlp_component(
    u, s, v,
    delta_activations,
    grad
):
    n_batches, n_tokens, _ = grad.shape
    n_singular_values = s.shape[-1]

    n_heads = 1
    u = einops.repeat(u, 'd_in d_singular_value -> head d_in d_singular_value', head=n_heads)
    s = einops.repeat(s, 'd_singular_value -> head d_singular_value', head=n_heads)
    v = einops.repeat(v, 'd_singular_value d_out -> head d_singular_value d_out', head=n_heads)
    grad = einops.repeat(grad, 'batch token d_out -> batch token head d_out', head=n_heads)
    delta_activations = einops.repeat(delta_activations, 'batch token d_in -> batch token head d_in', head=n_heads)

    importance_aproximation = calculate_importance_by_singular_value(
        n_batches,
        n_tokens,
        n_singular_values,
        n_heads,
        u,
        s,
        v,
        grad,
        delta_activations,
    )
    return importance_aproximation

def calculate_importance_kqv_component(
    u, s, v,
    delta_activations,
    grad
):
    n_batches, n_tokens, n_heads ,_ = grad.shape
    n_singular_values = s.shape[-1]

    repeated_delta_activations = einops.repeat(
        delta_activations,
        'batch token d_in -> batch token head d_in', head=n_heads
    )


    effect = calculate_importance_by_singular_value(
        n_batches,
        n_tokens,
        n_singular_values,
        n_heads,
        u,
        s,
        v,
        grad,
        repeated_delta_activations,
    )
    return effect

def calculate_importance_attn_out_component(
    u, s, v,
    delta_activations,
    grad
):
    n_batches, n_tokens, n_heads ,_ = grad.shape
    n_singular_values = s.shape[-1]

    effect = calculate_importance_by_singular_value(
        n_batches,
        n_tokens,
        n_singular_values,
        n_heads,
        u,
        s,
        v,
        grad,
        delta_activations,
    )
    return effect

def calculate_importance_by_singular_value(
    n_batches,
    n_tokens,
    n_singular_values,
    n_heads,
    u,                      # (head, d_in, d_singular_value)
    s,                      # (head, d_singular_value)
    v,                      # (head, d_singular_value, d_out)
    grad,                   # (batch, token, head, d_out)
    delta_activations,      # (batch, token, head, d_in)
):
    importance_aproximation = torch.zeros((n_batches, n_tokens, n_heads, n_singular_values))

    for singular_value in range(n_singular_values):
        u_singular_value = u[:, :, singular_value] # (head, d_in)
        s_singular_value = s[:, singular_value]    # (head)
        v_singular_value = v[:, singular_value, :] # (head, d_out)



        tmp = einsum('head d_in, batch token head d_in -> batch token head', u_singular_value, delta_activations)
        tmp = einsum('batch token head, head -> batch token head', tmp, s_singular_value)
        tmp = einsum('batch token head, head d_out -> batch token head d_out', tmp, v_singular_value)
        tmp = einsum('batch token head d_out, batch token head d_out -> batch token head', grad, tmp)

        importance_aproximation[:, :, :, singular_value] = tmp

    return importance_aproximation
