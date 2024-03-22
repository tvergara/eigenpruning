from transformer_lens import utils
from fancy_einsum import einsum
from notebooks.constants import PREACTIVATION_NAMES, PART_TO_PATH, PART_TO_BIAS_PATH
import einops

def compile_singular_values(
    model,
    singular_values,
    masks,
    activations
):
    for key, mask in masks.items():
        u, s, v = singular_values[key]
        if len(u.shape) == 2:
            u = einops.repeat(u, 'i j -> head i j', head=1)
            s = einops.repeat(s, 'i -> head i', head=1)
            v = einops.repeat(v, 'i j -> head i j', head=1)

        layer, part = key
        preactivation_name = PREACTIVATION_NAMES[part]
        preactivations = activations[utils.get_act_name(preactivation_name, layer)]
        __batch_size, tokens = preactivations.shape[:2]
        component, matrix_name = PART_TO_PATH[part]
        component = getattr(model.blocks[layer], component)
        weight_matrix = getattr(component, matrix_name).detach()

        component, bias_name = PART_TO_BIAS_PATH[part]
        component = getattr(model.blocks[layer], component)
        bias = getattr(component, bias_name).detach()

        mask = mask.to(s.device)
        s_mantained = s * mask
        s_removed = s * (1 - mask)


        new_weight_matrix = recompose_matrix(u, s_mantained, v)
        compiled_matrix = recompose_matrix(u, s_removed, v)

        avg_activation = preactivations.mean(dim=0)
        if part == 'result':
            delta_bias = result_component_delta_bias(
                avg_activation,
                compiled_matrix,
            )
        else:
            delta_bias = einsum(
                'token in, head in out -> head out',
                avg_activation,
                compiled_matrix
            )

        bias = bias.unsqueeze(0).repeat(tokens, *([1] * len(bias.shape)))  # Repeat 'tokens' times along the new dimension



        new_bias = bias + delta_bias

        weight_matrix = getattr(component, matrix_name)
        weight_matrix.data = new_weight_matrix.reshape(weight_matrix.data.shape)
        bias = getattr(component, bias_name)
        bias.data = new_bias

    return model


def recompose_matrix(
    u, # (head, in, singular_value)
    s, # (head, singular_value)
    v, # (head, singular_value, out)
):
    sv = s.unsqueeze(-1) * v
    usv = einsum('head in singular_value, head singular_value out -> head in out', u, sv)
    return usv

def result_component_delta_bias(
    avg_activation,
    compiled_matrix,
):
    return einsum(
        'token head in, head in out -> token out',
        avg_activation,
        compiled_matrix
    )


