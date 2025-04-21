#!/usr/bin/env python

import os.path
import torch
import numpy as np
from alpha_net import ChessNet as cnet
import chess
import encoder_decoder as ed
import copy
from MCTS_chess import UCT_search, do_decode_n_move_pieces
import pickle
import torch.multiprocessing as mp

def save_as_pickle(filename, data):
    completeName = os.path.join("./evaluator_data/",\
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)

class arena():
    def __init__(self,current_chessnet,best_chessnet):
        self.current = current_chessnet
        self.best = best_chessnet
    
    def play_round(self):
        if np.random.uniform(0,1) <= 0.5:
            white = self.current; black = self.best; w = "current"; b = "best"
        else:
            white = self.best; black = self.current; w = "best"; b = "current"
        current_board = chess.Board()
        states = []; dataset = []
        value = 0
        while not current_board.is_game_over() and current_board.fullmove_number <= 100:
            draw_counter = 0
            for s in states:
                if current_board.fen() == s:
                    draw_counter += 1
            if draw_counter == 3:  # draw by repetition
                break
            states.append(current_board.fen())
            board_state = copy.deepcopy(ed.encode_board(current_board))
            dataset.append(board_state)
            if current_board.turn:  # White to move
                best_move, _ = UCT_search(current_board, 777, white)
            else:  # Black to move
                best_move, _ = UCT_search(current_board, 777, black)
            current_board = do_decode_n_move_pieces(current_board, best_move)  # decode move and move piece(s)
            print(current_board, current_board.fullmove_number); print(" ")
            
            # Check for checkmate
            if current_board.is_checkmate():
                if current_board.turn:  # Black wins (White is in checkmate)
                    value = -1
                else:  # White wins (Black is in checkmate)
                    value = 1
                break
                
        dataset.append(value)
        if value == -1:
            return b, dataset
        elif value == 1:
            return w, dataset
        else:
            return None, dataset
    
    def evaluate(self, num_games, cpu):
        current_wins = 0
        for i in range(num_games):
            winner, dataset = self.play_round(); print("%s wins!" % winner)
            dataset.append(winner)
            if winner == "current":
                current_wins += 1
            save_as_pickle("evaluate_net_dataset_cpu%i_%i" % (cpu,i), dataset)
        print("Current_net wins ratio: %.3f" % current_wins/num_games)

def fork_process(arena_obj, num_games, cpu):  # make arena picklable
    arena_obj.evaluate(num_games, cpu)

if __name__=="__main__":
    mp.set_start_method("spawn",force=True)
    current_net="current_net.pth.tar"; best_net="current_net_trained.pth.tar"
    current_net_filename = os.path.join("./model_data/",\
                                current_net)
    best_net_filename = os.path.join("./model_data/",\
                                    best_net)
    current_chessnet = cnet()
    best_chessnet = cnet()
    checkpoint = torch.load(current_net_filename)
    current_chessnet.load_state_dict(checkpoint['state_dict'])
    checkpoint = torch.load(best_net_filename)
    best_chessnet.load_state_dict(checkpoint['state_dict'])
    cuda = torch.cuda.is_available()
    if cuda:
        current_chessnet.cuda()
        best_chessnet.cuda()
    current_chessnet.eval(); best_chessnet.eval()
    current_chessnet.share_memory(); best_chessnet.share_memory()
    
    processes = []
    for i in range(6):
        p = mp.Process(target=fork_process,args=(arena(current_chessnet,best_chessnet),50,i))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

        