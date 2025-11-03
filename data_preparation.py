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

def get_dataloaders(num_clients, batch_size):
    print("Loading WikiText dataset...")

    train_dataset_raw = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    val_dataset_raw = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')

    train_texts = [line for line in train_dataset_raw['text'] if len(line.strip()) > 0][:10000]
    val_texts = [line for line in val_dataset_raw['text'] if len(line.strip()) > 0][:1000]

    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = WikiTextDataset(train_texts, tokenizer)
    val_dataset = WikiTextDataset(val_texts, tokenizer)

    #将数据分发给多个客户端

    total_train_size = len(train_dataset)
    total_val_size = len(val_dataset)
    train_indices = list(range(total_train_size))
    val_indices = list(range(total_val_size))
    train_split_size = total_train_size // num_clients
    val_split_size = total_val_size // num_clients
    client_dataloaders = []

    for i in range(num_clients):
        train_start, train_end = i * train_split_size, (i + 1) * train_split_size
        client_train_dataset = Subset(train_dataset, train_indices[train_start:train_end])
        train_loader = DataLoader(client_train_dataset, batch_size=batch_size, shuffle=True)

        val_start, val_end = i * val_split_size, (i + 1) * val_split_size
        client_val_dataset = Subset(val_dataset, val_indices[val_start:val_end])
        val_loader = DataLoader(client_val_dataset, batch_size=batch_size, shuffle=False)

        client_dataloaders.append({'train': train_loader, 'val': val_loader})
        print(f"客户端 {i} 分配到 {len(client_train_dataset)} 条训练数据, {len(client_val_dataset)} 条验证数据。")

    return client_dataloaders

if __name__ == "__main__":
    dataloaders = get_dataloaders(num_clients=5, batch_size=64)
    print(f"总共创建了 {len(dataloaders)} 个客户端数据加载器。")

