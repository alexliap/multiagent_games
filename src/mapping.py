import itertools


def get_state_mapping(environment_size: tuple, possible_tile_states: tuple):
    state_row_dict = {}
    i = 0
    for comb in itertools.product(
        possible_tile_states, repeat=environment_size[0] * environment_size[1]
    ):
        state_row_dict[str(comb)] = i
        i += 1

    return state_row_dict


def get_desired_states(environment_size: tuple, possible_tile_states: tuple):
    # currently it works only when desired states are 2, i.e. only two colors used
    # and only for 3x3 environment
    size = environment_size[0] * environment_size[1]
    non_empty_tile_states = [i for i in possible_tile_states if i > 0]
    if len(non_empty_tile_states) == 2:
        desired_states = [[], []]
        for i in range(2):
            for j in range(size):
                if j % 2 == 0:
                    desired_states[i] += [1 + i]
                else:
                    desired_states[i] += [2 - i]

    for i in range(len(desired_states)):
        desired_states[i] = tuple(desired_states[i])

    return desired_states
