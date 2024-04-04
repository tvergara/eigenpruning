from torch.utils.data import DataLoader
from training.decompose_model import decompose_model
from training.save_backward_pass import add_hooks_to_save_backward_pass
from tqdm import tqdm
from training.compute_effect import compute_effect
from training.get_mask import get_mask
from training.compile_singular_values import compile_singular_values

import torch.nn.functional as F
import torch

def find_circuit(
    model,
    dataset,
    components=['k'],
    batch_size=5,
    portion_to_trim=0.3,
):
    singular_values = decompose_model(model, components)
    gradients_cache = add_hooks_to_save_backward_pass(model, components)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    effect_by_singular_values = {}
    for examples in tqdm(dataloader, desc='finding circuit'):

        input_ids = examples['input_ids']
        attention_masks = examples['attention_mask']
        targets = examples['correct_token']

        input_ids = input_ids.to(model.cfg.device)
        attention_masks = attention_masks.to(model.cfg.device)
        targets = targets.to(model.cfg.device)

        logits, activations = model.run_with_cache(input_ids, attention_mask=attention_masks)
        model.zero_grad()
        logits = logits[:, -1, :]
        loss = F.cross_entropy(logits, targets)
        loss.backward()

        effect = compute_effect(
            gradients_cache,
            activations,
            singular_values,
            list(range(model.cfg.n_layers)),
            components
        )
        merge_effects(effect_by_singular_values, effect)

    mask = get_mask(effect_by_singular_values, portion_to_trim)
    model = compile_singular_values(model, singular_values, mask, activations)
    return model, mask


def merge_effects(base, new):
    for key, value in new.items():
        if key in base:
            base[key] = torch.cat((base[key], value), 0)
        else:
            base[key] = value
    return base
