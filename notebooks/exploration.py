from transformer_lens import utils
import transformer_lens
import torch.nn.functional as F
from transformer_lens.hook_points import (HookPoint,)
import torch

from jaxtyping import Float

device = torch.device("cuda")

cache_path = '/mnt/ialabnas/homes/tvergara'
model = transformer_lens.HookedTransformer.from_pretrained("gpt2-xl", cache_dir=cache_path)

model.to(device)

# Run the model and get logits and activations
logits, activations = model.run_with_cache("Hello World")
logits

activations

max_size = 2
dataset = []
for i in range(max_size):
    for j in range(max_size):
        label = f" {i + j}"
        model.to_single_token(label)
        dataset.append((f'{i} + {j} =', label))

dataset

logits, activations = model.run_with_cache(dataset[0][0])

# lets do a single forward/backward pass of the whole dataset
# and accumulate the gradients

X = list(map(lambda x: x[0], dataset))
y = torch.tensor(list(map(lambda x: model.to_single_token(x[1]), dataset)), dtype=torch.int64)
y = y.to(device)



saved_backward = None
def save_backward(
    value: Float[torch.Tensor, "batch pos head_index d_head"],
    hook: HookPoint
):
    global saved_backward
    saved_backward = value

LAYER = 0
model.reset_hooks()
model.add_perma_hook(utils.get_act_name('v', LAYER), save_backward, dir='bwd')

model.blocks[0].attn.W_V.shape
logits, activations = model.run_with_cache(X)
logits = logits[:, -1, :]

loss = F.cross_entropy(logits, y)
model.zero_grad()
loss.backward()



# Assuming `W_V` is your tensor with shape [25, 1600, 64]
U_list = []
S_list = []
V_list = []


W_V = model.blocks[LAYER].attn.W_V.detach()
model.blocks[LAYER].mlp.W_V.detach()
getattr(model.blocks[LAYER].attn, 'W_V').detach()

for i in range(W_V.shape[0]):
    # Perform SVD on each slice
    U, S, V = torch.linalg.svd(W_V[i], full_matrices=False)
    U_list.append(U)
    S_list.append(S)
    V_list.append(V)

    # construct the matrix with each singular value

matrix_list = []
for i in range(len(U_list)):
    U = U_list[i]
    S = S_list[i]
    V = V_list[i]  # V already transposed in TensorFlow, not in PyTorch
    matrices_for_singular_values = []

    for j in range(len(S)):
        sigma = S[j]
        u = U[:, j].unsqueeze(1)
        v = V[j, :].unsqueeze(0)  # For TensorFlow: V[:, j].unsqueeze(1)
        matrix = sigma * torch.mm(u, v)  # Use tf.matmul(u, v) for TensorFlow
        matrices_for_singular_values.append(matrix)

    matrix_list.append(matrices_for_singular_values)

# send list of lists to tensor
matrix_tensor = torch.stack([torch.stack(matrices) for matrices in matrix_list])
matrix_tensor.shape

# Stack the results
U = torch.stack(U_list)
S = torch.stack(S_list)
V = torch.stack(V_list)

# Check the shapes
print(U.shape)
print(S.shape)
print(V.shape)

test_activations = activations[f"blocks.{LAYER}.hook_resid_pre"]
test_activations.shape



import torch
from einops import rearrange, repeat

activations_aligned = rearrange(test_activations, 'b t m -> (b t) 1 1 m 1')

# We need to align the singular value matrices for batch and token dimensions
# The dimensions are now properly aligned for batch-wise multiplication
matrix_tensor_aligned = repeat(matrix_tensor, 'h s m d -> (b t) h s m d', b=max_size ** 2, t=5)

# Multiplication and summing over the 'model_dim'
# Resulting shape should be [225*5, 25, 64, 64], corresponding to [batch*token, head_number, singular_value, head_dim]
result = torch.sum(activations_aligned * matrix_tensor_aligned, axis=-2)

# Reshaping the result to separate batch and token dimensions
# Final shape: [225, 5, 25, 64, 64]
result_reshaped = rearrange(result, '(b t) h s d -> b t h s d', b=max_size ** 2, t=5)

result_reshaped.shape # [batch, token, head_number, singular_value, head_dim]


rotated = torch.roll(result_reshaped, shifts=1, dims=0)

saved_backward.shape # [batch, token, head_number, head_dim]
torch.einsum('bthsd,bthd->bths', result_reshaped, saved_backward).shape


delta = rotated - result_reshaped

res = torch.einsum('bthsd,bthd->bths', delta, saved_backward)
res.shape

max_values, _ = torch.max(res, dim=0)

tensor_flat = max_values.flatten().cpu()

import torch
import matplotlib.pyplot as plt


plt.hist(tensor_flat * 10e4)

plt.title("Distribution of Tensor Values")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.savefig('histogram.png')


sorted_tensor, indices = torch.sort(tensor_flat)

# Step 3: Plot the sorted values using Matplotlib
plt.figure(figsize=(10, 6))
plt.plot(sorted_tensor.numpy(), marker='o')
plt.title('Sorted Values of a One-Dimensional Tensor')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)
plt.savefig('dist.png')
plt.show()

