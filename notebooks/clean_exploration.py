import transformer_lens
import notebooks.find_circuit as fc
import notebooks.sum_dataset as sm
import notebooks.mult_dataset as mt
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from notebooks.simplified_ioi_dataset import IOI
import torch

import importlib
importlib.reload(fc)
importlib.reload(sm)
importlib.reload(mt)

device = torch.device('cuda:3')
cache_path = '/mnt/ialabnas/homes/tvergara'

# models
# microsoft/phi-2
# gpt2-xl
model = transformer_lens.HookedTransformer.from_pretrained(
    "microsoft/phi-2",
    cache_dir=cache_path,
    device=device
)
model.set_use_attn_result(True)
model.set_use_hook_mlp_in(True)

layers = list(range(model.cfg.n_layers))
# layers = [10]
# parts = ['k', 'q', 'v', 'pre', 'mlp_out', 'result']
# parts = ['k', 'q', 'v', 'mlp_out', 'result']
parts = ['k']

# dataset = IOI(model)
dataset = sm.IntegerSumDataset(8, model)
# dataset = mt.IntegerMultDataset(5, model)
batch_size = 5

train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)


def backward_pass(logits, labels):
    logits = logits[:, -1, :]
    loss = F.cross_entropy(logits, labels)
    loss.backward()

circuit = fc.find_circuit(model, layers, parts, train_dataset, batch_size, backward_pass, iterations=1)

test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
for x, y in test_data_loader:
    model_logits = model(x)
    model_logits = model_logits[:, -1, :]

    circuit_logits = circuit(x)
    circuit_logits = circuit_logits[:, -1, :]

    model_loss = F.cross_entropy(model_logits, y)
    circuit_loss = F.cross_entropy(circuit_logits, y)
    print('loss', model_loss, circuit_loss)

    # evaluate accuracy
    model_preds = model_logits.argmax(dim=-1)
    circuit_preds = circuit_logits.argmax(dim=-1)
    model_accuracy = (model_preds == y).float().mean()
    circuit_accuracy = (circuit_preds == y).float().mean()
    print('accuracy', model_accuracy, circuit_accuracy)


one_shot_dataset = sm.IntegerSumDataset(10, model, '2 + 2 = 4\n')
one_shot_data_loader = DataLoader(one_shot_dataset, batch_size=batch_size, shuffle=True)

for x, y in one_shot_data_loader:
    model_logits = model(x)
    model_logits = model_logits[:, -1, :]


    model_loss = F.cross_entropy(model_logits, y)

    # evaluate accuracy
    model_preds = model_logits.argmax(dim=-1)
    model_accuracy = (model_preds == y).float().mean()
    print(model_accuracy)

