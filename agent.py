import random

import numpy as np

from mapping import get_state_mapping


class Agent:
    def __init__(
        self, environment_size: tuple, possible_tile_states: tuple, actions: tuple
    ):
        self.no_of_tiles = environment_size[0] * environment_size[1]
        # get state to row mapping
        self.state_row_mapping = get_state_mapping(
            environment_size, possible_tile_states
        )
        # calculate action space
        self.action_space = len(actions) * self.no_of_tiles
        # initialize Q table
        self.q_table = self.init_q_table(len(self.state_row_mapping), self.action_space)

        self.epsilon = 1

    def init_q_table(self, state_space: int, action_space: int):
        return np.zeros((state_space, action_space))

    def action(self, state: tuple):
        viable_actions = self.get_viable_actions(state)
        random_num = random.uniform(0, 1)
        if random_num > self.epsilon:
            # get max available value on for a state
            best_index = np.argmax(
                self.q_table[self.state_row_mapping[str(state)]][viable_actions]
            )
            # get best action from the viable ones
            best_action = viable_actions[best_index]

            action_dict = {"tile": best_action // 2, "action": (best_action % 2) + 1}
        else:
            action_dict = {
                "tile": random.choice(viable_actions) // 2,
                "action": (random.choice(viable_actions) % 2) + 1,
            }

        return action_dict

    def get_viable_actions(self, state: tuple):
        free_tiles = [i for i, x in enumerate(state) if x == 0]
        viable_actions = []
        for item in free_tiles:
            viable_actions.append(2 * item)
            viable_actions.append((2 * item) + 1)

        return viable_actions

    def update_table(self):
        pass
