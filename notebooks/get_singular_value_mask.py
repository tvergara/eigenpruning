import torch
import matplotlib.pyplot as plt

KILL_SV_PER_COMPONENT = {
    'k': 4/6,
    'q': 4/6,
    'v': 4/6,
    'pre': 4/6,
    'mlp_out': 10/11,
    'result': 5/6
}

def get_singular_value_mask(effect_by_singular_values):
    value_tuples = []
    for key, value in effect_by_singular_values.items():
        layer, part = key
        max_values_by_batch, _ = torch.max(value, dim=0)
        max_values_by_batch_and_token, _ = torch.max(max_values_by_batch, dim=0)  # (head, singular_value)

        for head in range(max_values_by_batch_and_token.shape[0]):
            for singular_value in range(max_values_by_batch_and_token.shape[1]):
                max_value = max_values_by_batch_and_token[head, singular_value].item()
                value_tuples.append((max_value, layer, part, head, singular_value))

    value_tuples.sort(reverse=True, key=lambda x: x[0])

    masks = {}
    for key, value in effect_by_singular_values.items():
        layer, part = key
        __n_batches, __tokens, heads, singular_values = value.shape
        mask = torch.zeros(heads, singular_values)
        masks[(layer, part)] = mask


    for component in KILL_SV_PER_COMPONENT.keys():
        tuples = [x for x in value_tuples if x[2] == component]
        percentage = KILL_SV_PER_COMPONENT[component]
        top_n = int((len(tuples) * percentage) // 1)
        print(f"TOP_N for {component}: {top_n}")
        for value_tuple in value_tuples[:top_n]:
            max_value, layer, part, head, singular_value = value_tuple
            masks[(layer, part)][head, singular_value] = 1

        plt.figure(figsize=(10, 6))
        plt.plot([x[0] for x in tuples], marker='o')
        plt.title('Sorted Values of a One-Dimensional Tensor')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.grid(True)
        plt.savefig(f"{component}.png")
        plt.show()


    # # TOP_N = len(value_tuples)
    # TOP_N = len(value_tuples) * 2 // 3
    # print(f"TOP_N: {TOP_N}")
    # print(f"MAX: {len(value_tuples)}")


    # for value_tuple in value_tuples[:TOP_N]:
    #     max_value, layer, part, head, singular_value = value_tuple
    #     masks[(layer, part)][head, singular_value] = 1

    return masks

