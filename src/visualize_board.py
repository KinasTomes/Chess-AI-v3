#!/usr/bin/env python

import matplotlib.pyplot as plt
from matplotlib.table import Table
import pandas as pd
import numpy as np
import chess
import chess.svg
from cairosvg import svg2png
from PIL import Image
import io

def view_board(board, fmt='{:s}', bkg_colors=['yellow', 'white']):
    """Visualize a chess board using matplotlib.
    
    Args:
        board (chess.Board): The chess board to visualize
        fmt (str): Format string for cell text
        bkg_colors (list): List of two colors for alternating squares
        
    Returns:
        matplotlib.figure.Figure: The figure containing the board visualization
    """
    # Convert chess.Board to 8x8 array of piece symbols
    board_array = np.full((8, 8), ' ', dtype=str)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            board_array[7-rank, file] = piece.symbol()
    
    # Create pandas DataFrame
    data = pd.DataFrame(board_array, columns=['A','B','C','D','E','F','G','H'])
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=[7,7])
    ax.set_axis_off()
    tb = Table(ax, bbox=[0,0,1,1])
    nrows, ncols = data.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i,j), val in np.ndenumerate(data):
        idx = [j % 2, (j + 1) % 2][i % 2]
        color = bkg_colors[idx]
        tb.add_cell(i, j, width, height, text=fmt.format(val), 
                    loc='center', facecolor=color)

    # Add row labels (1-8)
    for i in range(8):
        tb.add_cell(i, -1, width, height, text=str(8-i), loc='right', 
                    edgecolor='none', facecolor='none')

    # Add column labels (A-H)
    for j, label in enumerate(data.columns):
        tb.add_cell(-1, j, width, height/2, text=label, loc='center', 
                           edgecolor='none', facecolor='none')
    tb.set_fontsize(24)
    ax.add_table(tb)
    return fig

def save_board_svg(board, path):
    """Save a chess board as an SVG file using chess.svg.
    
    Args:
        board (chess.Board): The chess board to save
        path (str): Path where to save the SVG file
    """
    svg_data = chess.svg.board(board)
    with open(path, 'w') as f:
        f.write(svg_data)
        
def save_board_png(board, path, size=400):
    """Save a chess board as a PNG file using chess.svg and cairosvg.
    
    Args:
        board (chess.Board): The chess board to save
        path (str): Path where to save the PNG file
        size (int): Size of the output image in pixels
    """
    svg_data = chess.svg.board(board)
    png_data = svg2png(bytestring=svg_data.encode('utf-8'), output_width=size, output_height=size)
    img = Image.open(io.BytesIO(png_data))
    img.save(path)