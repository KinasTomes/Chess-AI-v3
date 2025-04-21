#!/usr/bin/env python

import numpy as np
import chess

def encode_board(board):
    """Encode a chess.Board into a numpy array representation.
    
    Args:
        board (chess.Board): The chess board to encode
        
    Returns:
        np.array: Encoded board state with shape [8,8,22]
    """
    encoded = np.zeros([8,8,22]).astype(int)
    
    # Piece placement encoding (first 12 planes)
    piece_dict = {
        'R': 0, 'N': 1, 'B': 2, 'Q': 3, 'K': 4, 'P': 5,
        'r': 6, 'n': 7, 'b': 8, 'q': 9, 'k': 10, 'p': 11
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            piece_idx = piece_dict[piece.symbol()]
            encoded[7-rank, file, piece_idx] = 1
    
    # Turn encoding (plane 12)
    if not board.turn:  # Black to move
        encoded[:,:,12] = 1
        
    # Castling rights (planes 13-16)
    if not board.has_queenside_castling_rights(chess.WHITE):
        encoded[:,:,13] = 1
    if not board.has_kingside_castling_rights(chess.WHITE):
        encoded[:,:,14] = 1
    if not board.has_queenside_castling_rights(chess.BLACK):
        encoded[:,:,15] = 1
    if not board.has_kingside_castling_rights(chess.BLACK):
        encoded[:,:,16] = 1
        
    # Move count and other metadata (planes 17-21)
    encoded[:,:,17] = board.fullmove_number
    # Note: repetitions are not directly available in chess.Board
    encoded[:,:,18] = 0  # repetitions_w
    encoded[:,:,19] = 0  # repetitions_b
    encoded[:,:,20] = board.halfmove_clock
    
    # En passant (plane 21)
    if board.ep_square is not None:
        encoded[:,:,21] = chess.square_file(board.ep_square)
    else:
        encoded[:,:,21] = -999
        
    return encoded

def decode_board(encoded):
    """Decode a numpy array representation back into a chess.Board.
    
    Args:
        encoded (np.array): Encoded board state with shape [8,8,22]
        
    Returns:
        chess.Board: The decoded chess board
    """
    board = chess.Board(fen=chess.STARTING_FEN)
    board.clear()
    
    # Decode piece placement
    decoder_dict = {
        0: 'R', 1: 'N', 2: 'B', 3: 'Q', 4: 'K', 5: 'P',
        6: 'r', 7: 'n', 8: 'b', 9: 'q', 10: 'k', 11: 'p'
    }
    
    for i in range(8):
        for j in range(8):
            for k in range(12):
                if encoded[i,j,k] == 1:
                    piece = chess.Piece.from_symbol(decoder_dict[k])
                    square = chess.square(j, 7-i)  # Convert to chess square
                    board.set_piece_at(square, piece)
    
    # Decode turn
    board.turn = not bool(encoded[0,0,12])
    
    # Decode castling rights
    board.clear_castling_rights()
    if not encoded[0,0,13]:
        board.set_castling_right(chess.WHITE, True, queen_side=True)
    if not encoded[0,0,14]:
        board.set_castling_right(chess.WHITE, True, queen_side=False)
    if not encoded[0,0,15]:
        board.set_castling_right(chess.BLACK, True, queen_side=True)
    if not encoded[0,0,16]:
        board.set_castling_right(chess.BLACK, True, queen_side=False)
    
    # Decode move counts
    board.fullmove_number = int(encoded[0,0,17])
    board.halfmove_clock = int(encoded[0,0,20])
    
    # Decode en passant
    ep_file = int(encoded[0,0,21])
    if ep_file != -999:
        rank = 5 if board.turn else 2
        board.ep_square = chess.square(ep_file, rank)
    
    return board

def encode_action(board, move):
    """Encode a chess move into a one-hot vector.
    
    Args:
        board (chess.Board): The current board state
        move (chess.Move): The move to encode
        
    Returns:
        int: Index of the encoded move (0-4671)
    """
    # Convert move to index based on from_square, to_square and promotion
    from_square = move.from_square
    to_square = move.to_square
    promotion = move.promotion
    
    # Calculate base index for the move
    from_rank = chess.square_rank(from_square)
    from_file = chess.square_file(from_square)
    to_rank = chess.square_rank(to_square)
    to_file = chess.square_file(to_square)
    
    # Encode normal moves
    if not promotion:
        # Queen-like moves (horizontal, vertical, diagonal)
        if from_rank == to_rank or from_file == to_file or abs(from_rank - to_rank) == abs(from_file - to_file):
            if from_rank == to_rank:  # Horizontal
                idx = 14 + (to_file - from_file + 7)
            elif from_file == to_file:  # Vertical
                idx = 0 + (to_rank - from_rank + 7)
            elif from_rank - to_rank == from_file - to_file:  # Diagonal
                idx = 28 + (to_rank - from_rank + 7)
            else:  # Anti-diagonal
                idx = 42 + (to_rank - from_rank + 7)
        # Knight moves
        elif (abs(from_rank - to_rank) == 2 and abs(from_file - to_file) == 1) or \
             (abs(from_rank - to_rank) == 1 and abs(from_file - to_file) == 2):
            knight_moves = [
                (2,-1), (2,1), (1,-2), (-1,-2),
                (-2,1), (-2,-1), (-1,2), (1,2)
            ]
            move_diff = (to_rank - from_rank, to_file - from_file)
            idx = 56 + knight_moves.index(move_diff)
    else:
        # Encode promotions
        prom_type = {chess.QUEEN: 'queen', chess.ROOK: 'rook', 
                    chess.BISHOP: 'bishop', chess.KNIGHT: 'knight'}[promotion]
        base_idx = {'rook': 64, 'knight': 65, 'bishop': 66}[prom_type]
        if abs(from_file - to_file) == 0:  # Straight promotion
            idx = base_idx
        elif to_file - from_file == -1:  # Capture left
            idx = base_idx + 3
        else:  # Capture right
            idx = base_idx + 6
            
    return idx

def decode_action(board, move_idx):
    """Decode a move index back into a chess move.
    
    Args:
        board (chess.Board): The current board state
        move_idx (int): Index of the encoded move (0-4671)
        
    Returns:
        List[Tuple]: List containing [(from_square, to_square, promotion)]
    """
    # First try to find a legal move that matches the encoding
    for move in board.legal_moves:
        if encode_action(board, move) == move_idx:
            from_square = move.from_square
            to_square = move.to_square
            promotion = move.promotion
            return [(chess.square_rank(from_square), chess.square_file(from_square)),
                   (chess.square_rank(to_square), chess.square_file(to_square)),
                   chess.piece_symbol(promotion) if promotion else None]
    
    raise ValueError(f"No legal move found for index {move_idx}")
