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

    raw_dataset = load_dataset("wikitext", 'wikitext-2-raw-v1', split="train")

    texts = [line for line in raw_dataset['text'] if len(line.strip()) > 0]

    texts = texts[:10000]

    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    full_dataset = WikiTextDataset(texts, tokenizer)

    #将数据分发给多个客户端

    total_size = len(full_dataset)
    indices = list(range(total_size))
    split_size = total_size // num_clients
    client_dataloaders = []

    for i in range(num_clients):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i != num_clients - 1 else total_size
        client_indices = indices[start_idx:end_idx]
        client_dataset = Subset(full_dataset, client_indices)
        loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
        client_dataloaders.append(loader)

        print(f"客户端 {i} 分配到了 {len(client_dataset)} 条数据。")

    return client_dataloaders

if __name__ == "__main__":
    dataloaders = get_dataloaders(num_clients=5, batch_size=8)
    print(f"总共创建了 {len(dataloaders)} 个客户端数据加载器。")

