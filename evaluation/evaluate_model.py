from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.nn.functional as F

def evaluate(model, dataset, batch_size=5):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_loss = 0.0
    correct_predictions = 0
    total_examples = 0
    model.eval()

    for examples in tqdm(dataloader, desc='Evaluating'):
        input_ids = examples['input_ids']
        attention_masks = examples['attention_mask']
        targets = examples['correct_token']

        input_ids = input_ids.to(model.cfg.device)
        attention_masks = attention_masks.to(model.cfg.device)
        targets = targets.to(model.cfg.device)

        logits = model(input_ids, attention_mask=attention_masks)
        logits = logits[:, -1, :]

        loss = F.cross_entropy(logits, targets, reduction='sum')
        total_loss += loss.item()

        preds = logits.argmax(dim=-1)
        correct_predictions += (preds == targets).sum().item()
        total_examples += targets.size(0)

    avg_loss = total_loss / total_examples
    accuracy = correct_predictions / total_examples

    print('Average loss:', avg_loss)
    print('Accuracy:', accuracy)
