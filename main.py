from models.models import get_model
from data.prepare_dataset import prepare_datasets
from training.find_circuit import find_circuit
from evaluation.evaluate_model import evaluate

import argparse
import torch
import uuid

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gpt2')
parser.add_argument('--cache_dir', type=str, default='/mnt/ialabnas/homes/tvergara')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--dataset', type=str, default='cb')
parser.add_argument('--decomposed_components', nargs='+', default=['k'])
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--portion_trim', type=float, default=0.3)
args = parser.parse_args()


experiment_id = uuid.uuid4()
device = torch.device(args.device)
model = get_model(args.model, device, args.cache_dir)
datasets = prepare_datasets(args.dataset, model.tokenizer)
print('starting experiment', experiment_id)

model, mask = find_circuit(
    model,
    datasets['train'],
    components=args.decomposed_components,
    batch_size=args.batch_size,
    portion_to_trim=args.portion_trim,
)

evaluate(model, datasets['test'], batch_size=args.batch_size)
