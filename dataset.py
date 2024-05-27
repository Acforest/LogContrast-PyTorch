import torch
import numpy as np
import pandas as pd
from functools import partial
from torch.utils.data import Dataset, DataLoader
from utils import evolution


class LogDataset(Dataset):
    def __init__(self,
                 raw_data,
                 tokenizer,
                 max_seq_len: int = 128,
                 sup_ratio: float = 1.0,
                 noise_ratio: float = 0.0,
                 evo_ratio: float = 0.0):
        self.dataset = []
        dataset_len = len(raw_data['semantics'])
        for idx in range(dataset_len):
            log_semantics = raw_data['semantics'][idx]
            if np.random.rand() < evo_ratio:
                log_semantics = evolution(log_semantics, tokenizer)
            else:
                log_semantics = log_semantics.split()

            log_sequence = eval(raw_data['sequences'][idx])
            log_sequence_mask = [1] * len(log_sequence)
            if len(log_sequence) < max_seq_len:
                log_sequence += [0] * (max_seq_len - len(log_sequence))
                log_sequence_mask += [0] * (max_seq_len - len(log_sequence_mask))
            else:
                log_sequence = log_sequence[:max_seq_len]
                log_sequence_mask = log_sequence_mask[:max_seq_len]

            true_label = 0 if raw_data['labels'][idx] == 'Normal' else 1
            if np.random.rand() < sup_ratio:
                train_label = true_label
                if np.random.rand() < noise_ratio:
                    train_label = 0 if true_label == 1 else 1
            else:
                train_label = -1

            self.dataset.append((log_semantics, log_sequence, log_sequence_mask, true_label, train_label))

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


def collate_fn(batch, tokenizer):
    log_semantics, log_sequences, log_sequence_masks, true_labels, train_labels = map(list, zip(*batch))

    log_semantics = tokenizer(log_semantics,
                              padding=True,
                              truncation=True,
                              max_length=256,
                              is_split_into_words=True,
                              add_special_tokens=True,
                              return_tensors='pt')
    log_sequences = torch.tensor(log_sequences, dtype=torch.long)
    log_sequence_masks = torch.tensor(log_sequence_masks, dtype=torch.bool)
    true_labels = torch.tensor(true_labels, dtype=torch.long)
    train_labels = torch.tensor(train_labels, dtype=torch.long)

    return {
        'semantics': log_semantics,
        'sequences': log_sequences,
        'sequence_masks': log_sequence_masks,
        'true_labels': true_labels,
        'train_labels': train_labels,
    }


def load_data(tokenizer,
              data_dir: str,
              batch_size: int = 16,
              max_seq_len: int = 32,
              sup_ratio: float = 1.0,
              noise_ratio: float = 0.0,
              evo_ratio: float = 0.0):
    data_df = pd.read_csv(data_dir)
    raw_data = {
        'semantics': data_df['EventTemplateSequence'].values.tolist(),
        'sequences': data_df['EventIdSequence'].values.tolist(),
        'labels': data_df['Label'].values.tolist()
    }

    dataset = LogDataset(raw_data, tokenizer,
                         max_seq_len=max_seq_len,
                         sup_ratio=sup_ratio,
                         noise_ratio=noise_ratio,
                         evo_ratio=evo_ratio)
    log_collate_fn = partial(collate_fn, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size, collate_fn=log_collate_fn)

    return dataloader
