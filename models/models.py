import transformer_lens


def get_model(model_name, device):
    model = transformer_lens.HookedTransformer.from_pretrained(
        model_name,
        device=device
    )
    model.set_use_attn_result(True)
    model.set_use_hook_mlp_in(True)
    return model



