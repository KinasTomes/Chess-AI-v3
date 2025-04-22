#!/usr/bin/env python

from alpha_net import ChessNet, train
from MCTS_chess import MCTS_self_play
import os
import pickle
import numpy as np
import torch
import torch.multiprocessing as mp
import chess

def get_gpu_id(process_id, num_gpus):
    return process_id % num_gpus

if __name__=="__main__":
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    
    for iteration in range(20):
        # Runs MCTS
        net_to_play="current_net_trained8_iter1.pth.tar"
        mp.set_start_method("spawn",force=True)
        
        # Create one network per GPU
        nets = []
        for gpu_id in range(num_gpus):
            net = ChessNet()
            if torch.cuda.is_available():
                net.cuda(gpu_id)
            net.share_memory()
            net.eval()
            current_net_filename = os.path.join("./model_data/", net_to_play)
            checkpoint = torch.load(current_net_filename, map_location=f'cuda:{gpu_id}')
            net.load_state_dict(checkpoint['state_dict'])
            nets.append(net)
            
        print("Starting MCTS self-play...")
        processes1 = []
        num_processes = 6  # Total number of processes
        
        for i in range(num_processes):
            gpu_id = get_gpu_id(i, num_gpus)
            p1 = mp.Process(target=MCTS_self_play, args=(nets[gpu_id], 50, i, gpu_id))
            p1.start()
            processes1.append(p1)
        for p1 in processes1:
            p1.join()
            
        # Runs Net training
        net_to_train="current_net_trained8_iter1.pth.tar"; save_as="current_net_trained8_iter1.pth.tar"
        # gather data
        datasets = []
        
        # Function to validate dataset entry
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
            
        # Load and validate data from each iteration
        for iter_num in range(3):  # Loading from iter0, iter1, iter2
            data_path = f"./datasets/iter{iter_num}/"
            if not os.path.exists(data_path):
                print(f"Warning: {data_path} does not exist, skipping...")
                continue
                
            for idx, file in enumerate(os.listdir(data_path)):
                filename = os.path.join(data_path, file)
                with open(filename, 'rb') as fo:
                    data = pickle.load(fo, encoding='bytes')
                    # Validate each entry before adding
                    valid_data = [entry for entry in data if validate_entry(entry)]
                    if len(valid_data) != len(data):
                        print(f"Warning: Filtered {len(data) - len(valid_data)} invalid entries from {filename}")
                    datasets.extend(valid_data)
        
        if not datasets:
            print("Error: No valid training data found!")
            continue
            
        print(f"Total valid training examples: {len(datasets)}")
        datasets = np.array(datasets, dtype=object)
        
        print("Starting neural network training...")
        mp.set_start_method("spawn",force=True)
        
        # Create one network per GPU for training
        nets = []
        for gpu_id in range(num_gpus):
            net = ChessNet()
            if torch.cuda.is_available():
                net.cuda(gpu_id)
            net.share_memory()
            net.train()
            current_net_filename = os.path.join("./model_data/", net_to_train)
            checkpoint = torch.load(current_net_filename, map_location=f'cuda:{gpu_id}')
            net.load_state_dict(checkpoint['state_dict'])
            nets.append(net)
        
        processes2 = []
        num_processes = 6  # Total number of processes
        
        for i in range(num_processes):
            gpu_id = get_gpu_id(i, num_gpus)
            p2 = mp.Process(target=train, args=(nets[gpu_id], datasets, 0, 200, i, gpu_id))
            p2.start()
            processes2.append(p2)
        for p2 in processes2:
            p2.join()
            
        # Save results - we save the last GPU's model state
        torch.save({'state_dict': nets[-1].state_dict()}, os.path.join("./model_data/", save_as))