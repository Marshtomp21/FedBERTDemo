import torch
import torch.nn as nn
import os
from transformers import RobertaConfig, RobertaForMaskedLM
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from flask import Flask, request
import io

class ServerModel(nn.Module):
    def __init__(self):
        super(ServerModel, self).__init__()

        self.config = RobertaConfig.from_pretrained("roberta-base")

        pretrained_model = RobertaForMaskedLM.from_pretrained("roberta-base")

        self.transformer_layers = pretrained_model.roberta.encoder

        print("Server model initialized with RoBERTa-base transformer layers.")

    def get_extended_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(
            dtype=self.transformer_layers.layer[0].attention.self.query.weight.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, hidden_states_from_client, attention_mask_from_client):
        extended_attention_mask = self.get_extended_attention_mask(attention_mask_from_client)

        encoder_outputs = self.transformer_layers(hidden_states_from_client, extended_attention_mask)

        hidden_states_to_client = encoder_outputs[0]

        return hidden_states_to_client

app = Flask(__name__)
server_model = ServerModel()
server_optimizer = torch.optim.Adam(server_model.parameters(), lr=0.001)

tensor_cache = {}

@app.route("/forward", methods=["POST"])
def forward_pass():
    data = request.get_data()
    buffer = io.BytesIO(data)

    received_tensors = torch.load(buffer, weights_only=False)

    client_activations = received_tensors['activations']
    attention_mask = received_tensors['attention_mask']

    client_activations.requires_grad = True

    server_optimizer.zero_grad()
    server_output = server_model(client_activations, attention_mask)

    tensor_cache['client_activations'] = client_activations
    tensor_cache['server_output'] = server_output

    buffer = io.BytesIO()
    torch.save(server_output, buffer)
    return buffer.getvalue()

@app.route("/backward", methods=["POST"])
def backward_pass():
    grad_from_client = torch.load(io.BytesIO(request.get_data()), weights_only=False)

    server_output = tensor_cache['server_output']
    client_activations = tensor_cache['client_activations']

    server_output.backward(gradient=grad_from_client)

    server_optimizer.step()

    grad_to_client = client_activations.grad

    buffer = io.BytesIO()
    torch.save(grad_to_client, buffer)
    return buffer.getvalue()

@app.route("/forward_eval", methods=["POST"])
def forward_eval_pass():
    data = request.get_data()
    buffer = io.BytesIO(data)
    received_tensors = torch.load(buffer, weights_only=False)

    client_activations = received_tensors['activations']
    attention_mask = received_tensors['attention_mask']

    with torch.no_grad():
        server_output = server_model(client_activations, attention_mask)

    buffer = io.BytesIO()
    torch.save(server_output, buffer)
    return buffer.getvalue()

@app.route("/save_model", methods=["GET"])
def save_model():
    torch.save(server_model.state_dict(), "server_model.pth")
    print("服务器模型已保存到 server_model.pth")
    return "Model saved", 200

#记得传到服务器上运行的时候修改port
if __name__ == "__main__":

    print("服务器正在 http://127.0.0.1:2778 运行...")

    app.run(host="0.0.0.0", port=2778)