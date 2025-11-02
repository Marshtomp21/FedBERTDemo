from client import ClientTrainer
from data_preparation import get_dataloaders
import requests
import multiprocessing

NUM_CLIENTS = 5
BATCH_SIZE = 64
GLOBAL_ROUNDS = 10
LOCAL_EPOCHS = 50
SERVER_URL = "http://192.168.1.151:16805"

def client_task(client_id, dataloader):
    print(f"Client {client_id} starting training...")
    try:
        requests.post(f"{SERVER_URL}/register_client", json={'client_id': client_id})
    except:
        print(f"Client {client_id} failed to connect to server.")
        return

    trainer = ClientTrainer(client_id=client_id, dataloader=dataloader, server_url=SERVER_URL)
    for epoch in range(LOCAL_EPOCHS):
        print(f"客户端 {client_id}, 本地 Epoch {epoch + 1}/{LOCAL_EPOCHS}")
        trainer.train_epoch()

    print(f"Client {client_id} training finished.")

def run_parallel_training():
    print("Preparing data loaders for clients...")
    client_dataloaders = get_dataloaders(num_clients=NUM_CLIENTS, batch_size=BATCH_SIZE)

    for round_num in range(GLOBAL_ROUNDS):
        print(f"\n===== 全局训练轮次 {round_num + 1}/{GLOBAL_ROUNDS} =====")

        processes = []

        for i in range(NUM_CLIENTS):
            process = multiprocessing.Process(target=client_task, args=(f"client_{i}", client_dataloaders[i]))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

        print(f"===== 全局训练轮次 {round_num + 1} 完成 =====\n")
        print(f"正在尝试聚合模型...")
        try:
            response = requests.post(f"{SERVER_URL}/aggregate")
            print(f"聚合响应: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"错误: 无法连接到服务器进行聚合 - {e}")
            return

    print("\n所有全局训练轮次完成。")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    run_parallel_training()
