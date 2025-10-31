from client import ClientTrainer
from data_preparation import get_dataloaders
import requests

NUM_CLIENTS = 5
BATCH_SIZE = 8
GLOBAL_ROUNDS = 5
LOCAL_EPOCHS = 1
SERVER_URL = "http://127.0.0.1:2778"

def run_sequential_training():
    print("Step 1: Preparing data loaders for clients...")
    client_dataloaders = get_dataloaders(num_clients=NUM_CLIENTS, batch_size=BATCH_SIZE)

    clients = [
        ClientTrainer(client_id = i, dataloader = loader, server_url = SERVER_URL)
        for i, loader in enumerate(client_dataloaders)
    ]

    print("Step 2: Starting Sequential Training...")

    for round_num in range(GLOBAL_ROUNDS):
        print(f"\n===== 全局轮次 {round_num + 1}/{GLOBAL_ROUNDS} =====")

        for client in clients:
            print(f"--- 客户端 {client.client_id} 训练中 ---")

            try:
                for epoch in range(LOCAL_EPOCHS):
                    client.train_epoch()
            except requests.exceptions.ConnectionError:
                print("服务器连接失败。")
                return
            except Exception as e:
                print (f"客户端 {client.client_id} 训练时发生错误: {e}")
                return

    print("训练完成。")

if __name__ == "__main__":
    run_sequential_training()
