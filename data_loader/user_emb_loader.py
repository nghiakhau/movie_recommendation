from torch.utils.data import Dataset


class UserEmbeddingDataset(Dataset):
    def __init__(self, tr, te):
        super().__init__()
        self.tr = tr
        self.te = te

    def __getitem__(self, idx):
        return self.tr[idx], self.te[idx]

    def __len__(self):
        return len(self.tr)

