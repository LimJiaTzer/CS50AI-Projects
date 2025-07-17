"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    x_count = 0
    o_count = 0
    for i in range(3):
        for j in range(3):
            if board[i][j]==X:
                x_count+=1
            elif board[i][j]==O:
                o_count += 1
    if x_count == 0 and o_count == 0:
        return X
    return X if o_count==x_count else O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possible_actions = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] is None:
                possible_actions.add((i, j))
    return possible_actions

def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    i, j = action
    p = player(board)
    resulting_board = [row[:] for row in board]
    if resulting_board[i][j] is None:
        resulting_board[i][j] = p
        return resulting_board
    else:
        raise ValueError("Not a valid action: position already occupied")


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # checks all horizontal
    for row in board:
        if row[0] is not None and row[0] == row[1] == row[2]:
            return row[0]
    # checks all vertical
    for col in range(3):
        if board[0][col] is not None and board[0][col]==board[1][col]==board[2][col]:
            return board[0][col]
    if board[1][1] is not None and (board[0][0]==board[1][1]==board[2][2] or board[0][2]==board[1][1]==board[2][0]):
        return board[1][1]
    return None



def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None:
        return True   
    for i in range(3):
        for j in range(3):
            if board[i][j] is None:
                return False
    return True

def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    status = winner(board)
    if status is None:
        return 0
    elif status == X:
        return 1
    else:
        return -1



def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None
    def finding_moves(board, alpha, beta):
        if terminal(board):
            return utility(board), None

        p_turn = player(board)

        if p_turn == X:
            max_no = -10**9
            for action in actions(board):
                no, _ = finding_moves(result(board, action), alpha, beta)
                if no>max_no:
                    optimal_action = action
                    max_no = no
                alpha = max(alpha, max_no)
                if beta<=alpha:
                    break
            return max_no, optimal_action

        if p_turn == O:
            min_no = 10**9
            for action in actions(board):
                no, _ = finding_moves(result(board, action), alpha, beta)
                if no<min_no:
                    optimal_action = action
                    min_no = no
                beta = min(beta, min_no)
                if beta<=alpha:
                    break
            return min_no, optimal_action
    _, optimal_action = finding_moves(board, float('-inf'), float('inf'))
    return optimal_action

