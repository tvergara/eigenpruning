from models.models import get_model
from data.prepare_dataset import prepare_dataset

import argparse
import torch
import uuid

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gpt2')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--dataset', type=str, default='cb')
args = parser.parse_args()


experiment_id = uuid.uuid4()
device = torch.device(args.device)
model = get_model(args.model, device)
dataset = prepare_dataset(args.dataset, model.tokenizer)

print('starting experiment', experiment_id)
