import numpy as np
import ann
from random import choice
import re
import timeit


class Game:
    def __init__(self, player1, player2, field_size=3):
        self.field = np.array(np.zeros((field_size, field_size)))
        self.turn = 1 - (2 * np.random.randint(0, 2))
        self.player_one = player1
        self.player_two = player2

    def print(self):
        print()
        for i in range(0, self.field.shape[0] - 1):
            self.__print_line(i)
            print('---' * self.field.shape[0] + '-' * (self.field.shape[0] - 1))
        self.__print_line(self.field.shape[0] - 1)

    def __print_line(self, i):
        for j in range(0, self.field.shape[1] - 1):
            print('', self.translate(self.field[i][j]), '|', end='')
        print('', self.translate(self.field[i][-1]))

    def translate(self, value):
        return {
            1: 'X',
            -1: 'O',
            0: ' '
        }[np.round(value)]

    def __set(self, position, value):
        if self.is_legal_move(position) and (value == 1 or value == -1):
            self.field[position] = value

    def is_legal_move(self, position):
        return self.field[position] == 0

    def is_won(self):
        # horizontal
        for x in range(0, self.field.shape[0]):
            row = self.__check_row(x)
            if row != 0:
                return row
        # vertical
        for y in range(0, self.field.shape[1]):
            col = self.__check_col(y)
            if col != 0:
                return col
        # diagonal
        diag = self.__check_diagonal()
        if diag != 0:
            return diag

        diag2 = self.__check_diagonal2()
        if diag2 != 0:
            return diag2

        return 0

    def __check_row(self, x):
        value = self.field[x][0]
        for y in range(1, self.field.shape[1]):
            if value != self.field[x][y]:
                return 0
        return value

    def __check_col(self, y):
        value = self.field[0][y]
        for x in range(1, self.field.shape[0]):
            if value != self.field[x][y]:
                return 0
        return value

    def __check_diagonal(self):
        value = self.field[0, 0]
        for xy in range(1, self.field.shape[0]):
            if value != self.field[xy][xy]:
                return 0
        return value

    def __check_diagonal2(self):
        value = self.field[-1, 0]
        for xy in range(1, self.field.shape[0]):
            if value != self.field[self.field.shape[0] - 1 - xy, xy]:
                return 0
        return value

    def is_finished(self):
        return self.is_won() or not self.moves_left()

    def moves_left(self):
        return 0 in self.field

    def __fast_is_won3x3(self, x, y):
        return Game.fast_is_won3x3(self.field, x, y)

    def get_free_spots(self):
        return self.free_spots(self.field)

    @staticmethod
    def free_spots(field):
        where_result = np.where(field == 0);
        return [(x, y) for x, y in zip(where_result[0], where_result[1])]

    @staticmethod
    def get_other_sign(sign):
        return -1 if sign == 1 else 1

    @staticmethod
    def fast_is_won3x3(field, x, y):
        if field[x][y] == field[x][0] == field[x][1] == field[x][2]:
            return True

        if field[x][y] == field[0][y] == field[1][y] == field[2][y]:
            return True

        if x == y and field[0][0] == field[1][1] == field[2][2]:
            return True

        if x + y == 2 and field[0][2] == field[1][1] == field[2][0]:
            return True

    def get_my_move_values(self, my_sign):
        return self.get_move_values(self.field, my_sign)

    @staticmethod
    def get_move_values(field, my_sign):
        move_values = []
        possible_moves = Game.free_spots(field)

        for current_move in possible_moves:
            new_array = np.copy(field)
            new_array[current_move[0]][current_move[1]] = my_sign
            # find the value of the move
            move_values.append(
                Game.min_max_iteration(new_array, my_sign, Game.get_other_sign(my_sign), 1, len(possible_moves)))

        return possible_moves, move_values

    @staticmethod
    def min_max_iteration(field, my_sign, current_sign, depth, maximum_depth):
        free_spots = Game.free_spots(field)
        if len(free_spots) == 0:
            return 0

        total = []
        for spot in free_spots:
            new_array = np.copy(field)
            new_array[spot[0]][spot[1]] = current_sign
            if Game.fast_is_won3x3(new_array, spot[0], spot[1]):
                if my_sign is current_sign:
                    total.append(1000 // depth)
                else:
                    total.append(-1000 // (maximum_depth - depth))
            else:
                total.append(Game.min_max_iteration(new_array, my_sign, Game.get_other_sign(current_sign), depth + 1, maximum_depth))
    
        if my_sign is current_sign:
            return np.amax(total)
        else:
            return np.amin(total)

    def run(self):
        winner = 0
        print(self.player_one.__class__.__name__, 'is', self.translate(1), ' - ',
              self.player_two.__class__.__name__, 'is', self.translate(-1), ' - ',
              self.translate(self.turn), 'starts')
        while self.moves_left() and winner == 0:
            start = timeit.default_timer()

            if self.turn == 1:
                move = self.player_one.make_move(self, self.turn)
            elif self.turn == -1:
                move = self.player_two.make_move(self, self.turn)

            end = timeit.default_timer()
            self.__set(move, self.turn)

            self.print()
            print()
            print(self.player_one.__class__.__name__ if self.turn == 1 else self.player_two.__class__.__name__,
                  'turn took', np.round(end - start, 3), 'seconds')

            if self.__fast_is_won3x3(move[0], move[1]):
                winner = self.turn
            else:
                self.turn = Game.get_other_sign(self.turn)

        print()
        if winner != 0:
            print('the winner is:', self.translate(winner))
        else:
            print('the game was a tie')

    def run_fast(self):
        winner = 0
        print(self.player_one.__class__.__name__, 'is', self.translate(1), ' - ',
              self.player_two.__class__.__name__, 'is', self.translate(-1), ' - ',
              self.translate(self.turn), 'starts')
        while self.moves_left() and winner == 0:
            if self.turn == 1:
                move = self.player_one.make_move(self, self.turn)
            elif self.turn == -1:
                move = self.player_two.make_move(self, self.turn)

            self.__set(move, self.turn)

            if self.__fast_is_won3x3(move[0], move[1]):
                winner = self.turn
            else:
                self.turn = Game.get_other_sign(self.turn)
        if winner != 0:
            print('the winner is:', self.translate(winner))
        else:
            print('the game was a tie')
        return winner


class HumanPlayer:
    def make_move(self, game, my_sign):
        game.print()
        while True:
            input_string = input(game.translate(my_sign) + " make your move:")
            user_input = [int(s) for s in re.split('[ +,-/|]', input_string) if s.isdigit()]
            if len(user_input) == 1:
                user_input = int(user_input[0])
                if 0 <= user_input < game.field.size:
                    my_move = tuple((user_input // game.field.shape[0], user_input % game.field.shape[1]))
                    if game.is_legal_move(my_move):
                        return my_move

            elif len(user_input) == len(game.field.shape):
                my_tuple = tuple(map(int, user_input))
                if self.__comp_tuple(my_tuple, game.field.shape):
                    if game.is_legal_move(my_tuple):
                        return my_tuple
            else:
                continue

    @staticmethod
    def __comp_tuple(a, b):
        for ai, bi in zip(a, b):
            if bi <= ai:
                return False
        return True


class AIPlayer:
    def make_move(self, game, my_sign):
        while (True):
            move = self.evaluate(game, my_sign)
            if game.is_legal_move(move):
                return move

    def evaluate(self, game, my_sign):
        raise NotImplementedError("Should have implemented this")

    @staticmethod
    def random_free_spot(field):
        free_spots = Game.free_spots(field)
        return free_spots[np.random.randint(0, len(free_spots))]


class RandomAIPlayer(AIPlayer):
    def evaluate(self, game, my_sign):
        # picks a random free field
        return self.random_free_spot(game.field)


class BlockingAIPlayer(AIPlayer):
    def evaluate(self, game, my_sign):
        free_spots = AIPlayer.free_spots(game.field)
        other_sign = Game.get_other_sign(my_sign)

        # check if we would win
        for spot in free_spots:
            new_array = np.copy(game.field)
            new_array[spot[0]][spot[1]] = my_sign
            if game.fast_is_won3x3(new_array, spot[0], spot[1]):
                return spot

        # check if the enemy would win
        for spot in free_spots:
            new_array = np.copy(game.field)
            new_array[spot[0]][spot[1]] = other_sign
            if Game.fast_is_won3x3(new_array, spot[0], spot[1]):
                return spot

        return AIPlayer.random_free_spot(game.field)


class MinMaxAIPlayer(AIPlayer):
    def __init__(self):
        self.my_sign = None
        self.maximum_depth = None

    def evaluate(self, game, my_sign):

        self.my_sign = my_sign

        possible_moves, move_values = game.get_my_move_values(my_sign)
        return possible_moves[self.pick_random_maximum(move_values)]

    @staticmethod
    def pick_random_maximum(move_values):
        # find the best move from the values
        maximum = np.argmax(move_values)
        indices = []
        # there might be multiple equal moves
        for i in range(0, len(move_values)):
            if move_values[i] == move_values[maximum]:
                indices.append(i)
        # pick a random move if there are more than one
        rand = np.random.randint(0, len(indices))
        return indices[rand]


class FastMinMaxAIPlayer(MinMaxAIPlayer):
    def evaluate(self, game, my_sign):
        possible_moves = game.get_free_spots()
        maximum_depth = len(possible_moves)

        # short cut - empty board takes 10 seconds to solve and all values are 0
        if maximum_depth == 9:
            return choice([(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)])

        move_values = []
        for current_move in possible_moves:
            new_array = np.copy(game.field)
            new_array[current_move[0]][current_move[1]] = my_sign
            # if we win with the next move, we don't need to think any further
            # would be better in an extra loop
            if Game.fast_is_won3x3(new_array, current_move[0], current_move[1]):
                return current_move
            # find the value of the move
            move_values.append(
                game.min_max_iteration(new_array, my_sign, Game.get_other_sign(my_sign), 1, maximum_depth))

        return possible_moves[self.pick_random_maximum(move_values)]


class NNAIPlayer(AIPlayer):
    def __init__(self):
        self.neural_net = ann.NeuralNet(9,27,9,ann.TanH());

    def train(self):
        file_input = eval(open('moves.txt','r').read())

        moves_input = []
        moves_output = []

        for single_input,single_output in file_input:
            moves_input.append(np.array(single_input[0]))
            moves_output.append(np.array(single_output[0]))

        moves_output[-1] = moves_input[-1] #fix for strange parsing bug

        moves_input= np.array(moves_input)
        moves_output = np.array(moves_output)

        print(len(moves_input),len(moves_output))

        self.neural_net.trainsingle(moves_input,moves_output , 1000)


    def evaluate(self, game, my_sign):
        return evaluatefield(game.field,my_sign)

    def evaluatefield(self, field, my_sign):
        field = np.copy(field)

        flattened_field = field.reshape(1,9)

        answer = self.neural_net.forward(np.array(flattened_field))

        picked = MinMaxAIPlayer.pick_random_maximum(answer[0])

        print(answer[0].reshape(3,3))

        return picked // 3, picked %3        

    def print(self):
        self.neural_net.print()

def test():
    player1 = FastMinMaxAIPlayer()
    player2 = FastMinMaxAIPlayer()
    result = 0
    iterations = 0
    while result is 0 and iterations < 1000:
        iterations += 1
        result = Game(player1, player2).run_fast()

#Game(FastMinMaxAIPlayer(), FastMinMaxAIPlayer()).run()

test_player = NNAIPlayer()
test_field = np.array(np.zeros((3,3)))
test_field[0][0] = 1
result = test_player.evaluatefield(test_field.reshape(1,9),-1)
test_player.train()
result = test_player.evaluatefield(test_field.reshape(1,9),-1)