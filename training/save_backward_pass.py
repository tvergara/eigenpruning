from transformer_lens import utils
from transformer_lens.hook_points import (HookPoint,)
import torch
from functools import partial
from jaxtyping import Float

def save_backward(
    value: Float[torch.Tensor, "batch pos head_index d_head"],
    hook: HookPoint,
    layer: int,
    component: str,
    cache: dict,
):

    cache[(layer, component)] = value

def add_hooks_to_save_backward_pass(model, components):
    cache = {}
    for component in components:
        for layer in range(model.cfg.n_layers):
            model.add_perma_hook(
                utils.get_act_name(component, layer),
                partial(save_backward, layer=layer, component=component, cache=cache),
                dir='bwd'
            )
    return cache
