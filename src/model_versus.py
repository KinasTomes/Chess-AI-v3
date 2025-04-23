#!/usr/bin/env python

import torch
import chess
import numpy as np
from alpha_net import ChessNet
from MCTS_chess import UCT_search
import encoder_decoder as ed
import time

class ChessGame:
    def __init__(self, model1_path, model2_path, gpu_id=0):
        self.board = chess.Board()
        self.device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
        
        # Load models
        self.model1 = self.load_model(model1_path)
        self.model2 = self.load_model(model2_path)
        
    def load_model(self, model_path):
        net = ChessNet()
        if torch.cuda.is_available():
            net.cuda(self.device)
        net.eval()
        checkpoint = torch.load(model_path, map_location=self.device)
        net.load_state_dict(checkpoint['state_dict'])
        return net
    
    def get_move(self, model, num_simulations=100):
        # Get network evaluation
        encoded_s = ed.encode_board(self.board)
        encoded_s = encoded_s.transpose(2, 0, 1)
        encoded_s = torch.from_numpy(encoded_s).float().to(self.device)
        
        # Run MCTS
        best_move, _ = UCT_search(self.board, num_simulations, model)
        return best_move
    
    def play_move(self, move_idx):
        # Find the move that matches the encoding
        for move in self.board.legal_moves:
            if ed.encode_action(self.board, move) == move_idx:
                self.board.push(move)
                return True
        return False
    
    def play_game(self, num_simulations=100, verbose=True):
        move_count = 0
        start_time = time.time()
        
        while not self.board.is_game_over() and move_count < 200:  # Max 200 moves
            # Select model based on turn
            current_model = self.model1 if self.board.turn else self.model2
            
            # Get move from current model
            move_idx = self.get_move(current_model, num_simulations)
            
            # Play the move
            if not self.play_move(move_idx):
                print("Error: Invalid move generated")
                break
                
            move_count += 1
            
            if verbose:
                print(f"\nMove {move_count}:")
                print(self.board)
                print(f"Current evaluation: {self.get_evaluation()}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Print game result
        result = self.get_game_result()
        print(f"\nGame finished in {move_count} moves ({duration:.2f} seconds)")
        print(f"Result: {result}")
        return result
    
    def get_evaluation(self):
        # Get network evaluation
        encoded_s = ed.encode_board(self.board)
        encoded_s = encoded_s.transpose(2, 0, 1)
        encoded_s = torch.from_numpy(encoded_s).float().to(self.device)
        
        with torch.no_grad():
            _, value = self.model1(encoded_s)
            return value.item()
    
    def get_game_result(self):
        if self.board.is_checkmate():
            return "Black wins" if self.board.turn else "White wins"
        elif self.board.is_stalemate():
            return "Draw by stalemate"
        elif self.board.is_insufficient_material():
            return "Draw by insufficient material"
        elif self.board.is_fifty_moves():
            return "Draw by fifty-move rule"
        elif self.board.is_repetition():
            return "Draw by repetition"
        else:
            return "Game not finished"

def play_tournament(model_paths, num_games=10, num_simulations=100, gpu_id=0):
    """
    Play a tournament between multiple models
    
    Args:
        model_paths: List of paths to model files
        num_games: Number of games to play between each pair
        num_simulations: Number of MCTS simulations per move
        gpu_id: GPU to use
    """
    results = {}
    
    # Initialize results dictionary
    for model in model_paths:
        results[model] = {'wins': 0, 'losses': 0, 'draws': 0}
    
    # Play games between each pair of models
    for i, model1_path in enumerate(model_paths):
        for j, model2_path in enumerate(model_paths[i+1:], i+1):
            print(f"\nPlaying games between {model1_path} and {model2_path}")
            
            for game in range(num_games):
                print(f"\nGame {game + 1}/{num_games}")
                
                # Create new game
                game = ChessGame(model1_path, model2_path, gpu_id)
                
                # Play game
                result = game.play_game(num_simulations)
                
                # Update results
                if result == "White wins":
                    results[model1_path]['wins'] += 1
                    results[model2_path]['losses'] += 1
                elif result == "Black wins":
                    results[model1_path]['losses'] += 1
                    results[model2_path]['wins'] += 1
                else:  # Draw
                    results[model1_path]['draws'] += 1
                    results[model2_path]['draws'] += 1
    
    # Print tournament results
    print("\nTournament Results:")
    print("-" * 50)
    for model, stats in results.items():
        print(f"\nModel: {model}")
        print(f"Wins: {stats['wins']}")
        print(f"Losses: {stats['losses']}")
        print(f"Draws: {stats['draws']}")
        print(f"Win rate: {stats['wins']/(stats['wins']+stats['losses']+stats['draws'])*100:.2f}%")

if __name__ == "__main__":
    # Example usage
    model_paths = [
        "./model_data/current_net_trained9.pth.tar",
        "./model_data/current_net_trained10.pth.tar"
    ]
    
    play_tournament(
        model_paths=model_paths,
        num_games=10,  # Number of games between each pair
        num_simulations=200,  # MCTS simulations per move
        gpu_id=0  # GPU to use
    ) 