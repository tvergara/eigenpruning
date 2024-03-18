import torch
import notebooks.decompose_model as dm
import notebooks.add_hooks as ah
import notebooks.compute_singular_values_effect as cs
import notebooks.get_singular_value_mask as gsvm
import notebooks.compile_singular_values as csv
from torch.utils.data import DataLoader
from tqdm import tqdm

import importlib
importlib.reload(dm)
importlib.reload(ah)
importlib.reload(cs)
importlib.reload(gsvm)
importlib.reload(csv)

def find_circuit(model, layers, parts, dataset, batch_size, backward_pass, iterations=1):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for i in range(iterations):
        print("Iteration", i)
        singular_values = dm.decompose_model(model, layers, parts)
        model = circuit_trim_iteration(model, layers, parts, dataloader, singular_values, backward_pass)
        # del model
        # import gc         # garbage collect library
        # gc.collect()
        # torch.cuda.empty_cache()
        # model = new_model

    return model

def circuit_trim_iteration(model,
                           layers,
                           parts,
                           dataloader,
                           singular_values,
                           backward_pass):
    gradients = ah.add_hooks_to_save_backward_pass(model, layers, parts)
    masks = {}
    effect_by_singular_values = {}
    for x, y in tqdm(dataloader):
        x.to(model.cfg.device)
        logits, activations = model.run_with_cache(x)
        model.zero_grad()
        backward_pass(logits, y)
        effect = cs.compute_aprox_effect(
            gradients,
            activations,
            singular_values,
            layers,
            parts
        )
        merge_effects(effect_by_singular_values, effect)

    masks = gsvm.get_singular_value_mask(effect_by_singular_values)
    model = csv.compile_singular_values(model, singular_values, masks, activations)
    return model

def merge_effects(base, new):
    for key, value in new.items():
        if key in base:
            base[key] = torch.cat((base[key], value), 0)
        else:
            base[key] = value
    return base
