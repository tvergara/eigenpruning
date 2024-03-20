from training.constants import COMPONENT_TO_OBJECT_PATH
from tqdm import tqdm

import torch


def decompose_model(model, components):
    singular_values = {}

    layers = list(range(model.cfg.n_layers))
    for layer in tqdm(layers, desc="Decomposing model"):
        for component in components:
            part, name = COMPONENT_TO_OBJECT_PATH[component]
            part = getattr(model.blocks[layer], part)
            weight_matrix = getattr(part, name).detach()
            u, s, v = torch.linalg.svd(weight_matrix, full_matrices=False)
            singular_values[(layer, component)] = (u, s, v)

    return singular_values
