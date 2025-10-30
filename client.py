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

def train_step(client_model, optimizer, input_ids, attention_mask, labels, server_url):
    optimizer.zero_grad()
    client_activations, attention_mask_to_server = client_model.forward_part1(input_ids, attention_mask)

    detached_activations = client_activations.detach().requires_grad_()

    buffer = io.BytesIO()
    torch.save({'activations': detached_activations, 'attention_mask': attention_mask_to_server}, buffer)

    response = requests.post(f"{server_url}/forward", data=buffer.getvalue())

    server_output = torch.load(io.BytesIO(response.content), weights_only=False)
    final_output = client_model.forward_part2(server_output)

    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(final_output.view(-1, client_model.config.vocab_size), labels.view(-1))

    loss.backward()

    grad_to_server = server_output.grad

    grad_buffer = io.BytesIO()
    torch.save(grad_to_server, grad_buffer)
    response = requests.post(f"{server_url}/backward", data=grad_buffer.getvalue())

    grad_from_server = torch.load(io.BytesIO(response.content), weights_only=False)
    client_activations.backward(gradient=grad_from_server)

    optimizer.step()

    return loss.item()

if __name__ == '__main__':
    server_url = 'http://127.0.0.1:2778'

    client_model = ClientModel()
    client_optimizer = torch.optim.Adam(client_model.parameters(), lr=0.001)

    vocab_size = client_model.config.vocab_size
    mask_token_id = 103

    input_ids = torch.randint(0, vocab_size, (4, 16))
    labels = input_ids.clone()

    prob_matrix = torch.full(labels.shape, 0.15)
    masked_indices = torch.bernoulli(prob_matrix).bool()
    labels[~masked_indices] = -100
    input_ids[masked_indices] = mask_token_id
    attention_mask = torch.ones_like(input_ids)

    print("\n开始训练...")

    for step in range(10):
        try:
            loss = train_step(client_model, client_optimizer, input_ids, attention_mask, labels, server_url)
            print(f"Step {step + 1}/10, Loss: {loss:.4f}")
        except requests.exceptions.ConnectionError as e:
            print("\n错误：无法连接到服务器。请确保先运行了 'python server.py'。")
            print(f"服务器地址: {server_url}")
            break