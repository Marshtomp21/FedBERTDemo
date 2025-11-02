import io
import torch
import torch.nn as nn
import requests
from transformers import RobertaConfig, RobertaForMaskedLM

class ClientModel(nn.Module):
    def __init__(self):
        super(ClientModel, self).__init__()

        self.config = RobertaConfig.from_pretrained('roberta-base')

        pretrained_model = RobertaForMaskedLM.from_pretrained('roberta-base')

        self.embedding_layer = pretrained_model.roberta.embeddings

        self.lm_head = pretrained_model.lm_head

        print("Client model initialized with RoBERTa-base embeddings and LM head.")

    def forward_part1(self, input_ids, attention_mask=None, token_type_ids=None):
        embedding_output = self.embedding_layer(input_ids=input_ids,
                                          token_type_ids=token_type_ids)
        return embedding_output, attention_mask

    def forward_part2(self, hidden_states_from_server):
        prediction_scores = self.lm_head(hidden_states_from_server)

        return prediction_scores

class ClientTrainer:
    def __init__(self, client_id, dataloader, server_url, lr=1e-5):
        self.client_id = client_id
        self.dataloader = dataloader
        self.server_url = server_url

        self.model = ClientModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for i, batch in enumerate(self.dataloader['train']):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            self.optimizer.zero_grad()

            client_activations, attention_mask_to_server = self.model.forward_part1(input_ids, attention_mask)

            detached_activations = client_activations.detach().requires_grad_()
            buffer_fwd = io.BytesIO()
            torch.save({'activations': detached_activations, 'attention_mask': attention_mask_to_server}, buffer_fwd)
            response_fwd = requests.post(f"{self.server_url}/forward", data=buffer_fwd.getvalue())

            server_output = torch.load(io.BytesIO(response_fwd.content), weights_only=False)
            final_output = self.model.forward_part2(server_output)
            loss = self.loss_fn(final_output.view(-1, self.model.config.vocab_size), labels.view(-1))

            loss.backward()
            grad_to_server = server_output.grad

            buffer_bwd = io.BytesIO()
            torch.save(grad_to_server, buffer_bwd)
            response_bwd = requests.post(f"{self.server_url}/backward", data=buffer_bwd.getvalue())

            grad_from_server = torch.load(io.BytesIO(response_bwd.content), weights_only=False)
            client_activations.backward(gradient=grad_from_server)

            self.optimizer.step()
            total_loss += loss.item()
            #每个epoch仅训练几个step
            if i >= 5:
                break
        avg_loss = total_loss / (i + 1)
        print(f"Client {self.client_id} - Average Loss: {avg_loss:.4f}")
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(self.dataloader['val']):
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']

                client_activations, attention_mask_to_server = self.model.forward_part1(input_ids, attention_mask)

                buffer_fwd = io.BytesIO()
                torch.save({'activations': client_activations, 'attention_mask': attention_mask_to_server}, buffer_fwd)
                response_fwd = requests.post(f"{self.server_url}/forward_eval", data=buffer_fwd.getvalue())

                server_output = torch.load(io.BytesIO(response_fwd.content), weights_only=False)
                final_output = self.model.forward_part2(server_output)
                loss = self.loss_fn(final_output.view(-1, self.model.config.vocab_size), labels.view(-1))

                total_loss += loss.item()

        avg_loss = total_loss / len(self.dataloader['val'])
        print(f"Client {self.client_id} - Validate Average Loss: {avg_loss:.4f}")
        return avg_loss
