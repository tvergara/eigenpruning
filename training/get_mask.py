from training.constants import COMPONENTS

import torch

def get_mask(effect_by_singular_values, proportion_to_trim):
    value_tuples = []
    for key, value in effect_by_singular_values.items():
        layer, component = key
        batch_dim, token_dim = 0, 1
        max_values = torch.amax(value, dim=(batch_dim, token_dim)) # (head, singular_value)

        for head in range(max_values.shape[0]):
            for singular_value in range(max_values.shape[1]):
                max_value = max_values[head, singular_value].item()
                value_tuples.append((
                    max_value, layer, component, head, singular_value
                ))

    value_tuples.sort(key=lambda x: x[0])

    mask = {}
    for key, value in effect_by_singular_values.items():
        layer, component = key
        __n_batches, __tokens, heads, singular_values = value.shape
        mask[(layer, component)] = torch.ones(heads, singular_values)

    for component in COMPONENTS:
        tuples = [x for x in value_tuples if x[2] == component]
        top_n = int((len(tuples) * proportion_to_trim) // 1)
        for value_tuple in value_tuples[:top_n]:
            max_value, layer, part, head, singular_value = value_tuple
            mask[(layer, part)][head, singular_value] = 0

    return mask

