import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from src.preprocess import encode_sentence


class POSDataset(Dataset):
    def __init__(self, sentences, vocab, tag2idx):
        self.data = [encode_sentence(s, vocab, tag2idx) for s in sentences]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x), torch.tensor(y)

def collate_fn(batch):
    """
    Pads input and target sequences in a batch.
    """
    xs, ys = zip(*batch)
    xs_pad = pad_sequence(xs, batch_first=True, padding_value=0)
    ys_pad = pad_sequence(ys, batch_first=True, padding_value=-100)  # -100 for ignore index
    return xs_pad, ys_pad
