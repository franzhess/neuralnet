from tictactoe import Game
import numpy as np

def get_move_values(field, my_sign):
    move_values = []
    possible_moves = Game.free_spots(field)

    for current_move in possible_moves:
        new_array = np.copy(field)
        new_array[current_move[0]][current_move[1]] = my_sign
        # find the value of the move
        move_values.append(
            min_max_iteration(new_array, my_sign, Game.get_other_sign(my_sign), 1, len(possible_moves)))

    
    flattened_field = field.reshape(1,9).astype(int).tolist()
    if str(flattened_field) not in allmoves:
        allmoves[str(flattened_field)] = flattened_field,move_values

    return possible_moves, move_values

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
                total.append(100000 / depth)
            else:
                total.append(-100000 / (maximum_depth - depth))
        else:
            total.append(min_max_iteration(new_array, my_sign, Game.get_other_sign(current_sign), depth + 1, maximum_depth))

    

    flattened_field = field.reshape(1,9).astype(int).tolist()

    minimum = np.amin(total)
    maximum = np.amax(total)
    normalize_range = maximum-minimum
    weights = np.array(-np.ones((3,3)))

    for index in range(0,len(free_spots)):
        if normalize_range < 0.0001:
            weights[free_spots[index][0]][free_spots[index][1]] = 1
        else:
            weights[free_spots[index][0]][free_spots[index][1]] = ((2 * (total[index] - minimum) / normalize_range) - 1)

    if str(flattened_field) not in allmoves and my_sign is current_sign:
       allmoves[str(flattened_field)] = flattened_field, weights.reshape(1,9).tolist()

    if my_sign is current_sign:
        return np.amax(total)
    else:
        return np.amin(total)


allmoves = dict()
get_move_values(np.array(np.zeros((3,3))),1)

print(len(allmoves))
with open('moves.txt','w+') as f:
    f.write(str(list(allmoves.values())))