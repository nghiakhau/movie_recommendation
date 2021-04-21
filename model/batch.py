import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
import numpy as np


class BatchText:
    def __init__(self, text, attention_mask, label):
        self.size = len(text)
        self.text = torch.LongTensor(text)
        self.attention_mask = torch.LongTensor(attention_mask)
        self.label = torch.FloatTensor(label)

    def to_device(self, device):
        self.text = self.text.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.label = self.label.to(device)


class TokenizersCollateFn:
    def __init__(self, config):
        self.max_token = config['sequence_max_length']
        self.padding = config['padding']
        self.truncation = config['truncation']

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=config['pretrained_tokenizer_name_or_path'],
            cache_dir=config['tokenizer_dir_cache'],
            use_fast=True if config['tokenizer_use_fast'] == 1 else False
        )

    def __call__(self, batch):
        xs = []
        ys = []
        for (x, y) in batch:
            xs.append(x)
            ys.append(y)
        encoded = self.tokenizer.batch_encode_plus(
                        batch_text_or_text_pairs=xs,
                        add_special_tokens=True,
                        padding=self.padding,
                        truncation=self.truncation,
                        max_length=self.max_token,
                        return_tensors='pt',
                        return_attention_mask=True)
        sequences_padded = encoded.input_ids
        attention_masks_padded = encoded.attention_mask
        ys = torch.tensor(ys)

        return BatchText(sequences_padded, attention_masks_padded, ys)


class BatchUser:
    def __init__(self, tr_idx, te_idx, tr_idx_wo_pad, num_movie):
        batch_size = len(tr_idx)
        self.tr_idx = torch.LongTensor(tr_idx)
        self.te_idx = te_idx
        self.tr_idx_wo_pad = tr_idx_wo_pad
        self.label = torch.zeros(batch_size, num_movie)
        for i in range(batch_size):
            self.label[i, self.tr_idx_wo_pad[i]] = 1.0

    def to_device(self, device):
        self.tr_idx = self.tr_idx.to(device)
        self.label = self.label.to(device)


class UserCollateFn:
    def __init__(self, config):
        self.max_movie = config.max_movie
        self.num_movie = config.num_movie
        self.PADDING = config.num_movie
        self.CLS = config.num_movie + 1
        np.random.seed(config.seed)

    def __call__(self, batch):
        tr_idx = []
        te_idx = []
        tr_idx_wo_pad = []

        for (tr, te) in batch:
            np.random.shuffle(tr)
            tr = tr.tolist()
            if len(tr) > self.max_movie:
                tr = tr[:self.max_movie]

            tr_idx.append(torch.tensor([self.CLS] + tr))
            # tr_idx.append(torch.tensor(tr))
            tr_idx_wo_pad.append(tr)
            te_idx.append(te.tolist())

        "Padding idx"
        tr_idx = pad_sequence(sequences=tr_idx, batch_first=True, padding_value=self.PADDING)

        return BatchUser(tr_idx, te_idx, tr_idx_wo_pad, self.num_movie)


