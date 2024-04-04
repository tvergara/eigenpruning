from models.models import get_model
from data.prepare_dataset import prepare_datasets
from training.find_circuit import find_circuit
from evaluation.evaluate_model import evaluate
from finetuning.finetune import finetune

import argparse
import torch
import uuid

torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gpt2')
parser.add_argument('--cache_dir', type=str, default='/mnt/ialabnas/homes/tvergara')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--dataset', type=str, default='cb')
parser.add_argument('--decomposed_components', nargs='+', default=['k'])
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--portion_trim', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=1)
args, unknown = parser.parse_known_args()

experiment_id = uuid.uuid4()
device = torch.device(args.device)
model = get_model(args.model, device, args.cache_dir)
datasets = prepare_datasets(args.dataset, model)
print('starting experiment', experiment_id)
print('model:', args.model)
print('portion_trim:', args.portion_trim)
print('dataset:', args.dataset)

finetune(
    model,
    datasets['train'],
    lr=args.lr,
    batch_size=args.batch_size,
    epochs=args.epochs
)

model, mask = find_circuit(
    model,
    datasets['train'],
    components=args.decomposed_components,
    batch_size=args.batch_size,
    portion_to_trim=args.portion_trim,
)

evaluate(model, datasets['test'], batch_size=args.batch_size)
