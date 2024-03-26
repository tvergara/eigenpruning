from torch.utils.data import Dataset, DataLoader, random_split
import torch

MAX_INT = 100

class IntegerSumDataset(Dataset):
    def __init__(self, max_int, model, prefix=''):
        self.data = []
        self.attention_masks = []
        self.labels = []
        self.prefix = prefix
        self._build_data(max_int, model)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'input_ids': self.data[idx],
            'attention_mask': self.attention_masks[idx],
            'correct_token': self.labels[idx]
        }

    def _build_data(self, max_int, model):
        for i in range(max_int):
            for j in range(max_int):
                result = i + j
                result = i + j
                str_label = ' ' + str(result)
                label =  model.to_single_token(str_label)
                label = torch.tensor(label)
                data_point = model.to_tokens(f'{i} + {j} =').squeeze()
                self.data.append(data_point)
                self.labels.append(torch.tensor(label))
                self.attention_masks.append(torch.ones_like(data_point))


def prepare_sum_dataset(model, train_size=0.8):
    dataset = IntegerSumDataset(MAX_INT, model)
    train_size = int(len(dataset) * train_size)
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])

    return {'train': train_data, 'test': test_data}
