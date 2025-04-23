#!/usr/bin/env python

import torch
import chess
import numpy as np
from alpha_net import ChessNet
from MCTS_chess import UCT_search
import encoder_decoder as ed
import time

class HumanVsModel:
    def __init__(self, model_path, gpu_id=0, num_simulations=100):
        self.board = chess.Board()
        self.device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
        self.num_simulations = num_simulations
        
        # Load model
        self.model = self.load_model(model_path)
        
    def load_model(self, model_path):
        net = ChessNet()
        if torch.cuda.is_available():
            net.cuda(self.device)
        net.eval()
        checkpoint = torch.load(model_path, map_location=self.device)
        net.load_state_dict(checkpoint['state_dict'])
        return net
    
    def get_model_move(self):
        # Get network evaluation
        encoded_s = ed.encode_board(self.board)
        encoded_s = encoded_s.transpose(2, 0, 1)
        encoded_s = torch.from_numpy(encoded_s).float().to(self.device)
        
        # Run MCTS
        best_move, _ = UCT_search(self.board, self.num_simulations, self.model)
        
        # Convert move index to UCI
        for move in self.board.legal_moves:
            if ed.encode_action(self.board, move) == best_move:
                return move.uci()
        
        return None
    
    def get_evaluation(self):
        # Get network evaluation
        encoded_s = ed.encode_board(self.board)
        encoded_s = encoded_s.transpose(2, 0, 1)
        encoded_s = torch.from_numpy(encoded_s).float().to(self.device)
        
        with torch.no_grad():
            _, value = self.model(encoded_s)
            return value.item()
    
    def play_game(self):
        print("\nWelcome to Human vs AI Chess!")
        print("You are playing as White. Enter moves in UCI format (e.g., 'e2e4', 'g1f3')")
        print("Type 'quit' to exit the game")
        print("\nInitial position:")
        print(self.board)
        
        while not self.board.is_game_over():
            if self.board.turn:  # Human's turn (White)
                while True:
                    move = input("\nYour move (UCI format): ").strip().lower()
                    
                    if move == 'quit':
                        print("Game ended by user")
                        return
                    
                    try:
                        chess_move = chess.Move.from_uci(move)
                        if chess_move in self.board.legal_moves:
                            self.board.push(chess_move)
                            break
                        else:
                            print("Illegal move! Try again.")
                    except:
                        print("Invalid move format! Use UCI format (e.g., 'e2e4')")
            
            else:  # AI's turn (Black)
                print("\nAI is thinking...")
                start_time = time.time()
                ai_move = self.get_model_move()
                end_time = time.time()
                
                if ai_move:
                    self.board.push(chess.Move.from_uci(ai_move))
                    print(f"AI played: {ai_move} (took {end_time - start_time:.2f} seconds)")
                else:
                    print("Error: AI couldn't find a valid move")
                    return
            
            print("\nCurrent position:")
            print(self.board)
            print(f"Evaluation: {self.get_evaluation():.3f}")
        
        # Game over
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn else "White"
            print(f"\nCheckmate! {winner} wins!")
        elif self.board.is_stalemate():
            print("\nStalemate! Game is a draw.")
        elif self.board.is_insufficient_material():
            print("\nInsufficient material! Game is a draw.")
        elif self.board.is_fifty_moves():
            print("\nFifty-move rule! Game is a draw.")
        elif self.board.is_repetition():
            print("\nThreefold repetition! Game is a draw.")

def main():
    # Get model path from user
    model_path = r'model_data\current_net_trained10.pth.tar'
    
    # Get number of simulations
    while True:
        try:
            num_simulations = int(input("Enter number of MCTS simulations per move (recommended: 100-200): "))
            if num_simulations > 0:
                break
            print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
    
    # Get GPU ID
    while True:
        try:
            gpu_id = int(input("Enter GPU ID to use (0 for first GPU): "))
            if gpu_id >= 0:
                break
            print("Please enter a non-negative number")
        except ValueError:
            print("Please enter a valid number")
    
    # Create and start game
    game = HumanVsModel(model_path, gpu_id, num_simulations)
    game.play_game()

if __name__ == "__main__":
    main() 