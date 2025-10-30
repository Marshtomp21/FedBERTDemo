from datasets import load_dataset
from transformers import RobertaTokenizer
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np

MODEL_NAME = 'roberta-base'
MAX_LENGTH = 128
MLM_PROBABILITY = 0.15

class WikiTextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.tokenizer = tokenizer
        self.max_length = MAX_LENGTH
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, MLM_PROBABILITY)

        masked_indices = torch.bernoulli(probability_matrix).bool()

        special_tokens_mask = self.tokenizer.get_special_tokens_mask(labels.tolist(), already_has_special_tokens=True)
        masked_indices &= ~torch.tensor(special_tokens_mask, dtype=torch.bool)

        input_ids[masked_indices] = self.tokenizer.mask_token_id

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

#还没写完