LIVING_REWARD = -0.1
DISCOUNTING_RATE = 0.2
LEARNING_RATE = 0.1
GOAL_REWARD = 100
FORBIDDEN_REWARD = -100
EPISILON = 0.1
MAX_ITERATIONS = 10000
START_SQUARE_IDX = 2

import sys
import copy
import random

# get user input
user_inputs = sys.argv[1].split(' ')
goal_loc_1, goal_loc_2, forbidden_loc, wall_loc, print_option = int(user_inputs[0]), int(user_inputs[1]), int(user_inputs[2]), int(user_inputs[3]), user_inputs[4]
try:
    print_square_loc = user_inputs[5]
except:
    pass

class Square:
    def __init__(self, idx):
        self.idx = idx
        self.square_type = None
        self.reward = 0
        # north, east, west, south
        self.Q_vals = [0, 0, 0, 0]
    
    def init_goal(self):
        self.square_type = 'G'
        self.reward = GOAL_REWARD

    def init_forbidden(self):
        self.square_type = 'F'
        self.reward = FORBIDDEN_REWARD
    
    def init_wall(self):
        self.square_type = 'W'

def move_to_square_idx(curr_square_idx, direction):
    # direction: 0 -> north; 1 -> east; 2 -> west; 3 -> south;
    if direction == 0:
        return curr_square_idx + 4 if 1 <= curr_square_idx <= 12 else curr_square_idx
    elif direction == 1:
        return curr_square_idx if curr_square_idx == 4 or curr_square_idx == 8 or curr_square_idx == 12 or curr_square_idx == 16 else curr_square_idx + 1
    elif direction == 2:
        return curr_square_idx if curr_square_idx == 1 or curr_square_idx == 5 or curr_square_idx == 9 or curr_square_idx == 13 else curr_square_idx - 1
    else:
        return curr_square_idx - 4 if 5 <= curr_square_idx <= 16 else curr_square_idx

def q_update(s_Q_vals, direction, reward, s_prime_Q_vals):
    # direction: 0 -> north; 1 -> east; 2 -> west; 3 -> south;
    return (1 - LEARNING_RATE) * s_Q_vals[direction] + LEARNING_RATE * (reward + LIVING_REWARD + DISCOUNTING_RATE * max(s_prime_Q_vals))

def is_convergent(old_board, new_board, iteration):
    if iteration == 0:
        return False
    for i in range(16):
        old_Q_vals = old_board[i].Q_vals
        new_Q_vals = new_board[i].Q_vals
        for j in range(4):
            print(round(new_Q_vals[j], 2), ' - ', round(old_Q_vals[j], 2), round(new_Q_vals[j], 2) - round(old_Q_vals[j], 2))
            print()
            if abs(round(new_Q_vals[j], 2) - round(old_Q_vals[j], 2)) > 0:
                return False
    print('iteration', iteration)
    return True

# construct the board
board = []
for i in range(1, 17):
    if i == goal_loc_1 or i == goal_loc_2:
        goal_sqaure = Square(i)
        goal_sqaure.init_goal()
        board.append(goal_sqaure)
    elif i == forbidden_loc:
        forbidden_square = Square(i)
        forbidden_square.init_forbidden()
        board.append(forbidden_square)
    elif i == wall_loc:
        wall_square = Square(i)
        wall_square.init_wall()
        board.append(wall_square)
    else:
        board.append(Square(i))

iteration = 0
new_board = copy.deepcopy(board)
curr_square = new_board[START_SQUARE_IDX-1]
while not is_convergent(board, new_board, iteration) and iteration < MAX_ITERATIONS:
# while iteration < MAX_ITERATIONS:
    board = copy.deepcopy(new_board)
    while not (curr_square.square_type == 'G' or curr_square.square_type == 'F'):
        # direction: 0 -> north; 1 -> east; 2 -> west; 3 -> south.
        if random.uniform(0, 1) < EPISILON:
            # agent chooses an action randomly
            move_direction = random.randint(0, 3)
        else:
            move_direction = curr_square.Q_vals.index(max(curr_square.Q_vals))
        move_to_square = new_board[move_to_square_idx(curr_square.idx, move_direction) - 1]
        if move_to_square.square_type == 'G' or move_to_square.square_type == 'F':
            print() 
        # update Q value
        curr_square.Q_vals[move_direction] = q_update(curr_square.Q_vals, move_direction, move_to_square.reward, move_to_square.Q_vals)
        if move_to_square.square_type == 'W':
            # keep current_sqaure as the new current_square
            pass
        else:
            curr_square = move_to_square
    curr_square = new_board[START_SQUARE_IDX-1]
    iteration += 1

direction_mappings = {0: '↑', 1: '→', 2:'←', 3: '↓'}

if print_option == 'p':
    for square in new_board:
        if square.square_type != 'W' and square.square_type != 'F' and square.square_type != 'G':
            print(square.idx, direction_mappings[square.Q_vals.index(max(square.Q_vals))])
else:
    for q_val_iter in range(4):
        print(direction_mappings[q_val_iter], new_board[print_square_loc - 1][q_val_iter])
