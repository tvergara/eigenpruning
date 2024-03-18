from transformer_lens import utils
from transformer_lens.hook_points import (HookPoint,)
import torch
from functools import partial
from jaxtyping import Float

def save_backward(
    value: Float[torch.Tensor, "batch pos head_index d_head"],
    hook: HookPoint,
    layer: int,
    part: str,
    cache: dict,
):

    cache[(layer, part)] = value

def add_hooks_to_save_backward_pass(model, layers, parts):
    cache = {}
    for part in parts:
        for layer in layers:
            model.add_perma_hook(
                utils.get_act_name(part, layer),
                partial(save_backward, layer=layer, part=part, cache=cache),
                dir='bwd'
            )
    return cache
