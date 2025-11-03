import torch
import torch.nn as nn
import os
from transformers import RobertaConfig, RobertaForMaskedLM
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from flask import Flask, request, jsonify
import io
import threading
import copy

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
global_server_model = ServerModel()

client_models = {}
client_optimizers = {}
client_tensor_caches = {}
client_update_status = {}
lock = threading.Lock()

@app.route("/register_client", methods=["POST"])
def register_client():
    client_id = request.json['client_id']
    with lock:
        if client_id not in client_models:
            print(f"注册新客户端: {client_id}")
            client_models[client_id] = copy.deepcopy(global_server_model)
            client_optimizers[client_id] = torch.optim.Adam(client_models[client_id].parameters(), lr=1e-5)
            client_tensor_caches[client_id] = {}
            client_update_status[client_id] = False
    return jsonify({"status": "registered"})

@app.route("/forward", methods=["POST"])
def forward_pass():
    data = request.get_data()
    buffer = io.BytesIO(data)

    received_data = torch.load(buffer, weights_only=False)

    client_id = received_data['client_id']
    client_activations = received_data['activations'].requires_grad_()
    attention_mask = received_data['attention_mask']

    model = client_models[client_id]
    optimizer = client_optimizers[client_id]

    optimizer.zero_grad()
    server_output = model(client_activations, attention_mask)

    client_tensor_caches[client_id] = {'client_activations': client_activations, 'server_output': server_output}

    buffer_out = io.BytesIO()
    torch.save(server_output, buffer_out)
    return buffer_out.getvalue()

@app.route("/backward", methods=["POST"])
def backward_pass():
    data = request.get_data()
    buffer = io.BytesIO(data)
    received_data = torch.load(buffer, weights_only=False)

    client_id = received_data['client_id']
    grad_from_client = received_data['gradient']

    cache = client_tensor_caches[client_id]
    server_output = cache['server_output']
    client_activations = cache['client_activations']

    server_output.backward(gradient=grad_from_client)

    client_optimizers[client_id].step()

    with lock:
        client_update_status[client_id] = True

    grad_to_client = client_activations.grad

    buffer_out = io.BytesIO()
    torch.save(grad_to_client, buffer_out)
    return buffer_out.getvalue()

@app.route("/aggregate", methods=["POST"])
def aggregate_models():
    global global_server_model
    print("聚合客户端模型到全局服务器模型...")
    with lock:
        updated_params_list = [model.state_dict() for model in client_models.values()]

        if not updated_params_list:
            return "No models to aggregate", 400

        aggregated_params = copy.deepcopy(updated_params_list[0])
        for key in aggregated_params.keys():
            aggregated_params[key] = torch.stack([params[key].float() for params in updated_params_list]).mean(dim=0)

        global_server_model.load_state_dict(aggregated_params)

        for client_id in client_models.keys():
            client_models[client_id].load_state_dict(global_server_model.state_dict())

        print("聚合完成，已更新全局服务器模型和客户端模型。")

    return "Aggregation complete", 200

@app.route("/forward_eval", methods=["POST"])
def forward_eval_pass():
    data = request.get_data()
    buffer = io.BytesIO(data)
    received_tensors = torch.load(buffer, weights_only=False)

    client_activations = received_tensors['activations']
    attention_mask = received_tensors['attention_mask']

    with torch.no_grad():
        server_output = global_server_model(client_activations, attention_mask)

    buffer = io.BytesIO()
    torch.save(server_output, buffer)
    return buffer.getvalue()

@app.route("/save_model", methods=["GET"])
def save_model():
    torch.save(global_server_model.state_dict(), "server_model.pth")
    print("服务器模型已保存到 server_model.pth")
    return "Model saved", 200

#记得传到服务器上运行的时候修改port
if __name__ == "__main__":

    print("服务器正在 http://127.0.0.1:2778 运行...")

    app.run(host="0.0.0.0", port=2778)