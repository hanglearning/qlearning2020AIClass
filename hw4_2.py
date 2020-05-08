import sys
import random

# Global Variables
LIVING_REWARD = -0.1
GAMMA = 0.2
ALPHA = 0.1
GOAL_SQUARE_REWARD = 100
FORBIDDEN_SQUARE_REWARD = -100
EPSILON = 0.1
MAX_ITERATIONS = 10000
START_SQUARE_INDEX = 2

# parse inputs
user_inputs = sys.argv[1].split(' ')

class Square:
    def __init__(self, index, reward=0, square_state="basic"):
        self.index = index
        self.reward = reward
        # either goal, forbidden, wall, and default to basic
        self.square_state = square_state
        # corresponding to ↑, →, ←, ↓ for easy printing
        self.q_values = [0,0,0,0]

class GameBoard:
    def __init__(self):
        # game board saved to a list
        self.board = []
        for square_index in range(1, 17):
            if square_index == int(user_inputs[0]) or square_index == int(user_inputs[1]):
                # construct a goal square
                self.board.append(Square(square_index, reward=GOAL_SQUARE_REWARD, square_state="goal"))
            elif square_index == int(user_inputs[2]):
                # construct a forbidden square
                self.board.append(Square(square_index, reward=FORBIDDEN_SQUARE_REWARD, square_state="forbidden"))
            elif square_index == int(user_inputs[3]):
                # construct a wall square
                self.board.append(Square(square_index, square_state="wall"))
            else:
                # basic square
                self.board.append(Square(square_index))
    
    def agent_choose_action(self, current_square):
        # return an integer indicating the action chosen - 0, 1, 2, 3 corresponds to ↑, →, ←, ↓
        return random.randint(0, 3) if random.uniform(0, 1) < EPSILON else current_square.q_values.index(max(current_square.q_values))
    
    def agent_move(self, current_square, chosen_action):
        # return the moved-into square
        # returning back the current_square when the agent bumps to the edge and stays put
        if chosen_action == 0:
            return self.board[current_square.index + 4 - 1] if current_square.index < 13 else current_square
        elif chosen_action == 1:
            return current_square if current_square.index in [4, 8, 12, 16] else self.board[current_square.index + 1 - 1]
        elif chosen_action == 2:
            return current_square if current_square.index in [1, 5, 9, 13] else self.board[current_square.index - 1 - 1]
        else:
            return self.board[current_square.index - 4 - 1] if current_square.index > 4 else current_square

    # main function
    def q_learning(self):
        for game_iteration in range(MAX_ITERATIONS):
            current_square = self.board[START_SQUARE_INDEX - 1]
            # q_value_convergence_check_list used to abort the iterations if q values all converge(within 2 digit precision change)
            q_value_convergence_check_list = []
            while current_square.square_state != 'goal' and current_square.square_state != 'forbidden':
                # chosen_action = 0, 1, 2, 3 corresponds to ↑, →, ←, ↓
                chosen_action = self.agent_choose_action(current_square)
                moved_into_square = self.agent_move(current_square, chosen_action)
                # update Q values
                old_q_value = current_square.q_values[chosen_action]
                current_square.q_values[chosen_action] = (1 - ALPHA) * old_q_value + ALPHA * (moved_into_square.reward + LIVING_REWARD + GAMMA * max(moved_into_square.q_values))
                # record the Q value change difference rounded to 2 decimal points
                q_value_convergence_check_list.append(abs(round(current_square.q_values[chosen_action], 2) - round(old_q_value, 2)))
                # stay put if moved into a wall
                current_square = current_square if moved_into_square.square_state == 'wall' else moved_into_square
            if max(q_value_convergence_check_list) == 0:
                break

    def print_policy(self):
        q_values_list_index_action_mapping = {0: '↑', 1: '→', 2:'←', 3: '↓'}
        if user_inputs[4] == 'p':
            for square in self.board:
                if square.square_state not in ['goal', 'forbidden', 'wall']:
                    print(square.index, q_values_list_index_action_mapping[square.q_values.index(max(square.q_values))])
        else:
            for q_value_iter in range(4):
                print(q_values_list_index_action_mapping[q_value_iter], self.board[int(user_inputs[5]) - 1].q_values[q_value_iter])

game = GameBoard()
game.q_learning()
game.print_policy()