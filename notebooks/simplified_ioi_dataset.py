from notebooks.ioi_dataset import IOIDataset, BABA_TEMPLATES
from torch.utils.data import Dataset
import torch

class IOI(Dataset):
    def __init__(self, model, N=100, prompt_type=BABA_TEMPLATES):
        self.dataset = IOIDataset(N=N, prompt_type=prompt_type)
        self.data = []
        self.labels = []
        self._build_data(self.dataset, model)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

    def _build_data(self, dataset, model):
        max_len = 0
        for prompt in dataset.ioi_prompts:
            text = prompt['text'][:-len(prompt['IO'])]
            label = prompt['IO']
            assert prompt['text'][-len(label):] == label
            self.data.append(model.to_tokens(text).squeeze())
            self.labels.append(model.to_tokens(label).squeeze()[0])

            if len(self.data[-1]) > max_len:
                max_len = len(self.data[-1])

        for i in range(len(self.data)):
            text = self.data[i]
            pad = torch.zeros(max_len - len(text))
            self.data[i] = torch.cat((text, pad), 0)

