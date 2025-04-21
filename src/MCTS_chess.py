#!/usr/bin/env python
import pickle
import os
import collections
import numpy as np
import math
import encoder_decoder as ed
import copy
import torch
import torch.multiprocessing as mp
from alpha_net import ChessNet
import datetime
import chess
from collections import defaultdict

def decode_n_move_pieces(board, move_idx):
    """Apply a move to a chess board.
    
    Args:
        board (chess.Board): The current board state
        move_idx (int): The encoded move index
        
    Returns:
        chess.Board: New board state after move
    """
    # Find the move that matches the encoding
    for move in board.legal_moves:
        if ed.encode_action(board, move) == move_idx:
            board.push(move)
            return board
            
    raise ValueError(f"No legal move found for index {move_idx}")

def do_decode_n_move_pieces(board, move_idx):
    """Apply a move to a chess board.
    
    Args:
        board (chess.Board): The current board state
        move_idx (int): The encoded move index
        
    Returns:
        chess.Board: New board state after move
    """
    return decode_n_move_pieces(board, move_idx)

class DummyNode:
    def __init__(self):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)
        self.visits = 0  # Add visits attribute for root node

class UCTNode():
    def __init__(self, game, move, parent=None):
        self.game = game  # chess.Board instance
        self.move = move  # The move that led to this node
        self.parent = parent or DummyNode()  # Use DummyNode if no parent
        self.children = {}  # Dict from moves to UCTNode
        self.unexplored_moves = list(game.legal_moves)
        self.player = game.turn  # True for white, False for black
        
        self.wins = 0
        self.visits = 0
        self._results = defaultdict(int)
        self.is_expanded = False
        self.child_priors = None
        self.action_idxes = []
        
    def maybe_add_child(self, move):
        """Add a move as a child of this node if not already present.
        
        Args:
            move: The move to add as a child
            
        Returns:
            UCTNode: The new or existing child node
        """
        if move not in self.children:
            copy_board = self.game.copy()
            copy_board = decode_n_move_pieces(copy_board, move)
            self.children[move] = UCTNode(copy_board, move, parent=self)
        return self.children[move]
    
    @property
    def number_visits(self):
        return self.visits

    @property
    def total_value(self):
        return self.wins
    
    def child_Q(self):
        if not self.visits:
            return 0
        return self.wins / self.visits
    
    def child_U(self):
        if self.child_priors is None:
            return 0
        return math.sqrt(self.parent.visits) * (
            abs(self.child_priors) / (1 + self.visits))
    
    def best_child(self):
        if not self.is_expanded:
            return None
            
        if not self.action_idxes:
            return None
            
        Q = self.child_Q()
        U = self.child_U()
        if isinstance(Q, (int, float)):
            Q = np.full_like(U, Q)
        scores = Q + U
        return self.action_idxes[np.argmax(scores[self.action_idxes])]
    
    def select_leaf(self):
        current = self
        while current.is_expanded and current.best_child() is not None:
            best_move = current.best_child()
            current = current.maybe_add_child(best_move)
        return current
    
    def add_dirichlet_noise(self, action_idxs, child_priors):
        valid_child_priors = child_priors[action_idxs]
        valid_child_priors = 0.75 * valid_child_priors + 0.25 * np.random.dirichlet(np.zeros([len(valid_child_priors)], dtype=np.float32) + 0.3)
        child_priors[action_idxs] = valid_child_priors
        return child_priors
    
    def expand(self, child_priors):
        self.is_expanded = True
        action_idxs = []
        c_p = child_priors
        
        for move in self.game.legal_moves:
            action_idx = ed.encode_action(self.game, move)
            action_idxs.append(action_idx)
            
        if not action_idxs:
            self.is_expanded = False
            
        self.action_idxes = action_idxs
        
        # Mask all illegal actions
        for i in range(len(child_priors)):
            if i not in action_idxs:
                c_p[i] = 0.0
                
        # Add dirichlet noise to child_priors in root node
        if self.parent.parent is None:
            c_p = self.add_dirichlet_noise(action_idxs, c_p)
            
        self.child_priors = c_p
    
    def backup(self, value_estimate: float):
        current = self
        while current.parent is not None:
            current.visits += 1
            if current.game.turn:  # White's turn
                current.wins += value_estimate
            else:  # Black's turn
                current.wins -= value_estimate
            current = current.parent

def UCT_search(game_state, num_reads, net):
    root = UCTNode(game_state, move=None, parent=None)  # Root node has no parent
    
    for _ in range(num_reads):
        leaf = root.select_leaf()
        
        # Skip if leaf is terminal
        if leaf.game.is_game_over():
            if leaf.game.is_checkmate():
                value_estimate = 1 if leaf.game.turn else -1
            else:
                value_estimate = 0  # Draw
            leaf.backup(value_estimate)
            continue
            
        # Get network evaluation
        encoded_s = ed.encode_board(leaf.game)
        encoded_s = encoded_s.transpose(2, 0, 1)
        encoded_s = torch.from_numpy(encoded_s).float().cuda()
        child_priors, value_estimate = net(encoded_s)
        child_priors = child_priors.detach().cpu().numpy().reshape(-1)
        value_estimate = value_estimate.item()
        
        # Expand and backup
        leaf.expand(child_priors)
        leaf.backup(value_estimate)
    
    # Select move with highest visit count
    visit_counts = np.array([root.children[move].visits if move in root.children else 0 
                            for move in range(4672)])
    return np.argmax(visit_counts), root

def get_policy(root):
    policy = np.zeros([4672], dtype=np.float32)
    for move, node in root.children.items():
        policy[move] = node.visits / root.visits
    return policy

def save_as_pickle(filename, data):
    save_dir = "./datasets/iter2/"
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
    completeName = os.path.join(save_dir, filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)

def MCTS_self_play(chessnet, num_games, cpu):
    for idxx in range(num_games):
        current_board = chess.Board()
        dataset = []  # to get state, policy, value for neural network training
        states = []
        value = 0
        
        while not current_board.is_game_over() and current_board.fullmove_number <= 100:
            # Check for threefold repetition
            draw_counter = 0
            current_fen = current_board.fen()
            for s in states:
                if current_fen == s:
                    draw_counter += 1
            if draw_counter == 3:  # draw by repetition
                break
                
            states.append(current_fen)
            board_state = copy.deepcopy(ed.encode_board(current_board))
            best_move, root = UCT_search(current_board, 777, chessnet)
            current_board = do_decode_n_move_pieces(current_board, best_move)
            policy = get_policy(root)
            dataset.append([board_state, policy])
            # print(current_board)
            # print(f"Move count: {current_board.fullmove_number}")
            # print()
            
            if current_board.is_checkmate():
                value = -1 if current_board.turn else 1
                break
                
        dataset_p = []
        for idx, data in enumerate(dataset):
            s, p = data
            if idx == 0:
                dataset_p.append([s, p, 0])
            else:
                dataset_p.append([s, p, value])
        
        save_as_pickle(f"dataset_cpu{cpu}_{idxx}_{datetime.datetime.today().strftime('%Y-%m-%d')}", dataset_p)

if __name__ == "__main__":
    net_to_play = "current_net_trained8_iter1.pth.tar"
    mp.set_start_method("spawn", force=True)
    net = ChessNet()
    cuda = torch.cuda.is_available()
    if cuda:
        net.cuda()
    net.share_memory()
    net.eval()
    print("Starting MCTS self-play...")
    
    current_net_filename = os.path.join("./model_data/", net_to_play)
    checkpoint = torch.load(current_net_filename)
    net.load_state_dict(checkpoint['state_dict'])
    
    processes = []
    for i in range(6):
        p = mp.Process(target=MCTS_self_play, args=(net, 50, i))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
