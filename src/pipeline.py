#!/usr/bin/env python

from alpha_net import ChessNet, train
from MCTS_chess import MCTS_self_play
import os
import pickle
import numpy as np
import torch
import torch.multiprocessing as mp

def get_gpu_id(process_id, num_gpus):
    return process_id % num_gpus

def clean_old_data(current_data_dir: str) -> None:
    if os.path.exists(current_data_dir):
        print(f"Cleaning up data in {current_data_dir}")
        for file in os.listdir(current_data_dir):
            os.remove(os.path.join(current_data_dir, file))
        print("Cleaned current_iter folder.")
    os.makedirs(current_data_dir, exist_ok=True) 
    print("Cleaned old iteration data.")

def validate_entry(entry):
    if len(entry) != 3:  # Should have state, policy, value
        return False
    state, policy, value = entry
    if state.shape != (8, 8, 22):  # Check state shape
        return False
    if len(policy) != 4672:  # Check policy vector size
        return False
    if not isinstance(value, (int, float)):  # Check value is scalar
        return False
    return True

def run_pipeline(model_dir: str, num_iterations: int = 10, processes_per_gpu: int = 4) -> None:
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    mp.set_start_method("spawn", force=True)

    os.makedirs(model_dir, exist_ok=True)

    num_processes = num_gpus * processes_per_gpu
    print(f"Running with {num_processes} total processes ({processes_per_gpu} per GPU)")

    for iteration in range(num_iterations):
        print(f"\nStarting iteration {iteration + 1}/{num_iterations}")

        
        current_data_dir = r"./datasets/current_iter/"
        clean_old_data(current_data_dir=current_data_dir)

        # Run MCTS self-play
        net_to_play = os.path.join(model_dir, f"current_net_trained{iteration}.pth.tar")
        if not os.path.exists(net_to_play):
            if iteration == 0:
                model = ChessNet()
                if torch.cuda.is_available():
                    model.cuda()
                torch.save({'state_dict': model.state_dict()}, net_to_play)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        nets_mcts = []
        for gpu_id in range(num_gpus):
            net = ChessNet()
            if torch.cuda.is_available():
                net.cuda(gpu_id)
            net.share_memory()
            net.eval()
            checkpoint = torch.load(net_to_play, map_location=f'cuda:{gpu_id}')
            net.load_state_dict(checkpoint['state_dict'])
            nets_mcts.append(net)

        print("Starting MCTS self-play...")
        processes1 = []
        for i in range(num_processes):
            gpu_id = get_gpu_id(i, num_gpus)
            p1 = mp.Process(target=MCTS_self_play, args=(nets_mcts[gpu_id], 50, i, gpu_id))
            p1.start()
            processes1.append(p1)
        for p1 in processes1:
            p1.join()

        del nets_mcts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        net_to_train = os.path.join(model_dir, f"current_net_trained{iteration}.pth.tar")
        
        datasets = []

        print(f"Loading data from {current_data_dir}")
        for idx, file in enumerate(os.listdir(current_data_dir)):
            filename = os.path.join(current_data_dir, file)
            try:
                with open(filename, 'rb') as fo:
                    try:
                        data = pickle.load(fo, encoding='bytes')
                        valid_data = [entry for entry in data if validate_entry(entry)]
                        if len(valid_data) != len(data):
                            print(f"Warning: Filtered {len(data) - len(valid_data)} invalid entries from {filename}")
                        datasets.extend(valid_data)
                    except (pickle.UnpicklingError, EOFError) as e:
                        print(f"Warning: Could not load or invalid pickle file {filename}. Error: {e}")
            except FileNotFoundError:
                print(f"Warning: File {filename} disappeared during loading.")
            except Exception as e:
                print(f"Error loading file {filename}: {e}")


        if not datasets:
            print("Error: No valid training data found!")
            continue

        print(f"Total valid training examples: {len(datasets)}")
        datasets = np.array(datasets, dtype=object)

        num_samples = len(datasets)
        indices = np.arange(num_samples)
        np.random.shuffle(indices) 

        shard_indices = np.array_split(indices, num_processes)
        dataset_shards = []
        original_datasets_np = np.array(datasets, dtype=object) # Chuyển đổi ở đây
        for shard_idx_list in shard_indices:
            if len(shard_idx_list) > 0: # Chỉ thêm shard nếu có dữ liệu
                dataset_shards.append(original_datasets_np[shard_idx_list])
        del original_datasets_np # Giải phóng bộ nhớ mảng lớn ban đầu
        del datasets # Giải phóng bộ nhớ list ban đầu

        if not dataset_shards:
            print("Error: No data shards created, possibly due to empty dataset.")
            continue

        actual_num_processes_train = len(dataset_shards)
        print(f"Training with {actual_num_processes_train} processes due to data sharding.")

        print("Starting neural network training...")

        nets_train = []
        for gpu_id in range(num_gpus):
            net = ChessNet()
            if torch.cuda.is_available():
                net.cuda(gpu_id)
            net.share_memory()
            net.train()
            checkpoint = torch.load(net_to_train, map_location=f'cuda:{gpu_id}')
            net.load_state_dict(checkpoint['state_dict'])
            nets_train.append(net)

        processes2 = []
        for i in range(actual_num_processes_train):
            gpu_id = get_gpu_id(i, num_gpus)
            shard_data = dataset_shards[i]
            if len(shard_data) == 0: # Skip if shard is empty
                print(f"Skipping process {i} due to empty data shard.")
                continue
            p2 = mp.Process(target=train, args=(nets_train[gpu_id], shard_data, 0, 200, i, gpu_id))
            p2.start()
            processes2.append(p2)
        for p2 in processes2:
            p2.join()

        # Save results - we save the last GPU's model state
        # Find the net associated with GPU 0 (or process 0 if you prefer)
        # Assuming nets_train is ordered by GPU ID or you can find the correct one
        net_to_save = None
        for i in range(actual_num_processes_train):
            if get_gpu_id(i, num_gpus) == 0:
                # Assuming nets_train list was populated matching GPU assignments
                # This logic might need adjustment depending on how nets_train is structured
                # A simpler way might be to just save nets_train[0] if it always exists and maps to gpu 0
                try:
                    net_to_save = nets_train[get_gpu_id(i, num_gpus)] # Assuming nets_train indices match gpu_id
                    break
                except IndexError: # Fallback if nets_train wasn't fully populated
                    pass
        if net_to_save is None and nets_train: # Fallback if GPU 0 wasn't used or failed
            net_to_save = nets_train[0]

        if net_to_save:
            save_as = os.path.join(model_dir, f"current_net_trained{iteration + 1}.pth.tar")
            # Ensure model is on CPU before saving to avoid device mismatches later
            torch.save({'state_dict': net_to_save.to('cpu').state_dict()}, save_as)
            print(f"Completed iteration {iteration + 1}, model saved as {save_as}")
        else:
            print("Warning: No network found to save.")
        
        print("Cleaning GPUs memory...", end = " ")
        for net in nets_train:
            if torch.cuda.is_available():
                torch.cuda.empty_cache();
        print("Done!")
        print(f"Completed iteration {iteration + 1}, model saved as {save_as}")

if __name__ == '__main__':
    model_dir = r"./model_data/"
    num_iterations = 10
    processes_per_gpu = 4

    run_pipeline(model_dir=model_dir, num_iterations=num_iterations, processes_per_gpu=processes_per_gpu)
    print("Pipeline completed successfully.")