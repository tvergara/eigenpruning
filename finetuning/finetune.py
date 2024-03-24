from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.nn.functional as F
import torch

def finetune(model, dataset, lr=0.01, batch_size=5, epochs=1):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for examples in tqdm(dataloader, desc='finetuning'):
        input_ids = examples['input_ids']
        attention_masks = examples['attention_mask']
        targets = examples['correct_token']

        input_ids = input_ids.to(model.cfg.device)
        attention_masks = attention_masks.to(model.cfg.device)
        targets = targets.to(model.cfg.device)

        logits = model(input_ids, attention_mask=attention_masks)
        model.zero_grad()
        logits = logits[:, -1, :]
        loss = F.cross_entropy(logits, targets)
        loss.backward()

        optimizer.step()






