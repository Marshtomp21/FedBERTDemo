from client import ClientTrainer
from data_preparation import get_dataloaders
import requests

NUM_CLIENTS = 5
BATCH_SIZE = 64
GLOBAL_ROUNDS = 10
LOCAL_EPOCHS = 50
SERVER_URL = "http://192.168.1.151:16805"

def run_sequential_training():
    print("Step 1: Preparing data loaders for clients...")
    client_dataloaders = get_dataloaders(num_clients=NUM_CLIENTS, batch_size=BATCH_SIZE)

    clients = [
        ClientTrainer(client_id = i, dataloader = loader, server_url = SERVER_URL)
        for i, loader in enumerate(client_dataloaders)
    ]
    best_val_loss = float('inf')
    print("Step 2: Starting Sequential Training...")

    for round_num in range(GLOBAL_ROUNDS):
        print(f"\n===== 全局训练轮次 {round_num + 1}/{GLOBAL_ROUNDS} =====")

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
        print(f"\n--- 全局轮次 {round_num + 1} 结束，开始评估 ---")
        total_val_loss = 0
        for client in clients:
            val_loss = client.evaluate()
            total_val_loss += val_loss
        avg_val_loss = total_val_loss / len(clients)
        print(f"===== 全局轮次 {round_num + 1} 平均验证损失: {avg_val_loss:.4f} =====")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"*** 发现新的最佳模型，验证损失为: {best_val_loss:.4f}。正在保存... ***")
            try:
                requests.get(f"{SERVER_URL}/save_model")
            except requests.exceptions.ConnectionError:
                print("服务器连接失败，无法保存模型。")
                return
    print("训练完成。")

if __name__ == "__main__":
    run_sequential_training()
