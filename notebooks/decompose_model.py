import torch
from notebooks.constants import PART_TO_PATH
from tqdm import tqdm


def decompose_model(model, layers, parts):
    singular_values = {}

    for layer in tqdm(layers):
        for part in parts:
            component, name = PART_TO_PATH[part]
            component = getattr(model.blocks[layer], component)
            weight_matrix = getattr(component, name).detach()
            u, s, v = torch.linalg.svd(weight_matrix, full_matrices=False)
            singular_values[(layer, part)] = (u, s, v)

    return singular_values
