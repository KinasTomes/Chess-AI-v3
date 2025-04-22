# --- START OF FILE alpha_net.py ---

#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg") # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
import os
import datetime
import numpy as np # Import numpy if not already imported

class board_data(Dataset):
    def __init__(self, dataset_shard): # dataset_shard = np.array of (s, p, v) for this shard
        # Check if the shard is empty or malformed before accessing columns
        if dataset_shard is None or dataset_shard.shape[0] == 0:
            print("Warning: Received empty or None dataset shard.")
            self.X = np.array([])
            self.y_p = np.array([])
            self.y_v = np.array([])
        elif dataset_shard.ndim < 2 or dataset_shard.shape[1] < 3:
             print(f"Warning: Received malformed dataset shard with shape {dataset_shard.shape}.")
             # Handle potential errors - maybe assign empty arrays or raise error
             self.X = np.array([])
             self.y_p = np.array([])
             self.y_v = np.array([])
             # Or raise ValueError("Dataset shard must have at least 3 columns")
        else:
            self.X = dataset_shard[:, 0]
            self.y_p = dataset_shard[:, 1]
            self.y_v = dataset_shard[:, 2]

    def __len__(self):
        # Return 0 if X is empty to avoid errors
        return len(self.X) if hasattr(self, 'X') else 0

    def __getitem__(self, idx):
        # Add checks for empty arrays to prevent index errors
        if not hasattr(self, 'X') or idx >= len(self.X):
             raise IndexError("Index out of bounds or dataset not initialized properly")

        # Ensure data types are consistent before returning
        state = self.X[idx]
        # Ensure state is numpy array before transposing if needed
        if isinstance(state, np.ndarray):
             state = state.transpose(2, 0, 1) # Assuming state is HWC (8,8,22), transpose to CHW
        else:
             # Handle cases where state might not be a numpy array as expected
             # Maybe convert or raise an error depending on expected input
             print(f"Warning: State at index {idx} is not a numpy array.")
             # Example: Convert if possible, or handle error
             # state = np.array(state).transpose(2,0,1) # Attempt conversion

        policy = self.y_p[idx]
        value = self.y_v[idx]

        # You might want to explicitly convert types here if necessary
        # e.g., ensure policy is float array, value is float
        # policy = np.array(policy, dtype=np.float32)
        # value = float(value)

        return state, policy, value

# --- ConvBlock, ResBlock, OutBlock, ChessNet, AlphaLoss remain the same ---

class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.action_size = 8*8*73
        self.conv1 = nn.Conv2d(22, 256, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

    def forward(self, s):
        s = s.view(-1, 22, 8, 8)  # batch_size x channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        return s

class ResBlock(nn.Module):
    def __init__(self, inplanes=256, planes=256, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out

class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(256, 1, kernel_size=1) # value head
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(8*8, 64)
        self.fc2 = nn.Linear(64, 1)

        self.conv1 = nn.Conv2d(256, 128, kernel_size=1) # policy head
        self.bn1 = nn.BatchNorm2d(128)
        # Use LogSoftmax for numerical stability during training
        # The loss function (cross-entropy typically expects logits or log-probabilities)
        # AlphaLoss seems to expect probabilities (using .exp() later), so LogSoftmax + exp is okay.
        # Alternatively, modify AlphaLoss to work with log-probabilities directly.
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(8*8*128, 8*8*73)

    def forward(self,s):
        # Value head
        v = self.conv(s) # Raw output
        v = self.bn(v)   # Batch norm
        v = F.relu(v)    # ReLU activation
        v = v.view(-1, 8*8)  # Flatten spatial dimensions
        v = self.fc1(v)
        v = F.relu(v)    # ReLU activation
        v = self.fc2(v)
        v = torch.tanh(v) # Tanh activation for value in [-1, 1]

        # Policy head
        p = self.conv1(s)
        p = self.bn1(p)
        p = F.relu(p)
        p = p.view(-1, 8*8*128) # Flatten
        p = self.fc(p)
        # No activation here if using nn.CrossEntropyLoss which includes Softmax
        # If using AlphaLoss which expects probabilities:
        p = self.logsoftmax(p).exp() # Apply LogSoftmax then exp to get probabilities
        return p, v

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv = ConvBlock()
        # Use nn.Sequential for cleaner residual blocks definition
        self.res_blocks = nn.Sequential(
            *[ResBlock() for _ in range(19)]
        )
        self.outblock = OutBlock()

    def forward(self,s):
        s = self.conv(s)
        s = self.res_blocks(s)
        p, v = self.outblock(s) # Unpack policy and value
        return p, v


class AlphaLoss(torch.nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, value_pred, value_target, policy_pred, policy_target):
        # Ensure targets are the correct type and shape
        value_target = value_target.view_as(value_pred).float()
        policy_target = policy_target.float()

        # Value loss: Mean Squared Error
        value_error = F.mse_loss(value_pred, value_target, reduction='none') # (Batch,)

        # Policy loss: Cross-Entropy between predicted policy distribution and target policy distribution
        # Assumes policy_pred are probabilities (output of softmax/exp(logsoftmax))
        # and policy_target are target probabilities (e.g., from MCTS visits)
        # Add small epsilon for numerical stability in log
        policy_error = -torch.sum(policy_target * torch.log(policy_pred + 1e-8), dim=1) # (Batch,)

        # Combine losses
        # The paper often includes L2 regularization term as well, but it's not here.
        # Taking the mean over the batch
        total_error = (value_error.squeeze() + policy_error).mean()
        return total_error

# Updated train function
def train(net, dataset_shard, epoch_start=0, epoch_stop=200, process_id=0, gpu_id=0): # Renamed parameters
    """Trains the network on a given data shard using a specific GPU."""
    print(f"Process {process_id} starting training on GPU {gpu_id} with {len(dataset_shard)} samples.")
    torch.manual_seed(process_id) # Use process_id for seeding if desired
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id) # Set the correct device for this process
        print(f"Process {process_id} set active device to GPU {gpu_id}")
    else:
         print(f"Warning: Process {process_id} running on CPU as CUDA is not available.")

    net.train() # Set the network to training mode
    criterion = AlphaLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.003) # Consider making lr configurable
    # Adjusted milestones for epoch_stop=200
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 140], gamma=0.2)

    # Use the provided data shard
    train_set = board_data(dataset_shard)
    if len(train_set) == 0:
        print(f"Process {process_id} received an empty dataset. Skipping training.")
        return # Exit if no data

    # Consider increasing batch size if GPU memory allows
    batch_size = 64 # Example larger batch size
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    losses_per_epoch = []
    for epoch in range(epoch_start, epoch_stop):

        total_loss_epoch = 0.0
        num_batches = 0
        for i, data in enumerate(train_loader, 0):
            state, policy_target, value_target = data

            # Move data to the assigned GPU
            if torch.cuda.is_available():
                state = state.cuda(gpu_id).float()
                policy_target = policy_target.float().cuda(gpu_id)
                value_target = value_target.cuda(gpu_id).float()
            else: # Handle CPU case
                 state = state.float()
                 policy_target = policy_target.float()
                 value_target = value_target.float()


            optimizer.zero_grad()
            # Forward pass
            policy_pred, value_pred = net(state)
            # Calculate loss
            # Ensure value_pred has the same shape as value_target for the loss function
            loss = criterion(value_pred.squeeze(), value_target, policy_pred, policy_target)
            loss.backward()
            optimizer.step()

            total_loss_epoch += loss.item()
            num_batches += 1

            # Logging periodically
            if i % 20 == 19: # Log less frequently for larger datasets/batch sizes
                avg_loss_batch = total_loss_epoch / num_batches # Average loss since last log or epoch start
                print(f'Proc {process_id} [Epoch: {epoch + 1}, { (i + 1) * batch_size }/{len(train_set)} points] Avg loss: {avg_loss_batch:.4f} (GPU: {gpu_id})')
                # Reset counters for next log interval within epoch if desired, or keep cumulative
                # total_loss_epoch = 0.0
                # num_batches = 0


        # Calculate average loss for the epoch
        avg_loss_this_epoch = total_loss_epoch / num_batches if num_batches > 0 else 0
        losses_per_epoch.append(avg_loss_this_epoch)
        if epoch >= 100: # Only print after 100 epochs
            print(f"Proc {process_id} Epoch {epoch + 1} finished. Avg Loss: {avg_loss_this_epoch:.4f}")

        # Step the scheduler after each epoch
        scheduler.step()

        # Optional: Early stopping based on loss trend for this process's shard
        if len(losses_per_epoch) > 100: # Check after enough epochs
             # Check if average loss over last 3 epochs vs average loss over earlier 3 epochs
             # is not improving significantly. Be careful with this on single shards.
             last_3_avg = sum(losses_per_epoch[-3:]) / 3
             prev_3_avg = sum(losses_per_epoch[-15:-12]) / 3 # Compare with a much earlier period
             if abs(last_3_avg - prev_3_avg) < 0.005: # Threshold for significant improvement
                 print(f"Process {process_id}: Early stopping triggered at epoch {epoch + 1}. Loss stagnated.")
                 break

    # --- Plotting (only for process 0 to avoid conflicts) ---
    # Ensure process_id corresponds to the intended primary process (often 0)
    if process_id == 0:
        print(f"Process {process_id} generating loss plot...")
        try:
            fig = plt.figure()
            ax = fig.add_subplot(111) # Use 111 for a single plot
            # Create x-axis values that match the number of epochs actually completed
            epochs_completed = list(range(1, len(losses_per_epoch) + 1))
            ax.plot(epochs_completed, losses_per_epoch) # Use plot for line graph
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Average Loss per Epoch")
            ax.set_title(f"Loss vs Epoch (Proc {process_id}, GPU {gpu_id})")
            # Ensure model_data directory exists
            plot_dir = "./model_data/"
            os.makedirs(plot_dir, exist_ok=True)
            plot_filename = os.path.join(plot_dir, f"Loss_vs_Epoch_{datetime.datetime.today().strftime('%Y-%m-%d')}_Proc{process_id}_GPU{gpu_id}.png")
            plt.savefig(plot_filename)
            plt.close(fig) # Close the figure to free memory
            print(f'Process {process_id} finished Training. Plot saved to {plot_filename}')
        except Exception as e:
            print(f"Warning: Could not create plot on Process {process_id} / GPU {gpu_id}: {str(e)}")
    else:
        print(f'Process {process_id} finished Training on GPU {gpu_id} (No plot generated).')


# --- END OF FILE alpha_net.py ---