from torch.utils.data import Dataset
import torch


class IntegerMultDataset(Dataset):
    def __init__(self, max_int, model, prefix=''):
        self.data = []
        self.labels = []
        self.prefix = prefix
        self._build_data(max_int, model)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

    def _build_data(self, max_int, model):
        for i in range(max_int):
            for j in range(max_int):
                label = f" {i * j}"
                label = torch.tensor(model.to_single_token(label))
                data_point = model.to_tokens(self.prefix + f'{i} * {j} =').squeeze()
                label = label.to(model.cfg.device)
                self.data.append(data_point)
                self.labels.append(torch.tensor(label))



