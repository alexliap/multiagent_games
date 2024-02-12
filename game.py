import numpy as np
from tqdm import tqdm

from agent import Agent


class GameBOne:
    """
    This Game class refers to the B.1 game rules, where agents play one after the other and
    have knowledge of their other agents actions.
    """

    def __init__(
        self, environment_size: tuple, possible_tile_states: tuple, actions: tuple
    ):
        self.state = self.reset_state(environment_size)

        self.get_state()

        # create the 2 agents (it's self play so the 2nd agent is the same one)
        self.agent_1 = Agent(environment_size, possible_tile_states, actions)

    def reset_state(self, environment_size: tuple):
        return tuple(np.zeros((environment_size[0], environment_size[1])).flatten())

    def play(self, epochs):
        exploration_end = int(0.8 * epochs)
        linear_decay_rate = 100 / exploration_end
        for i in tqdm(range(epochs), bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"):
            while self.check_board():
                # get 1st agent's action
                action_1 = self.agent_1.action(self.state)
                # update state based on the above action
                self.state = self.update_state(action_1)
                # reward agent if the move was correct, otherwise penalize him

                # get 2nd agent's action
                action_2 = self.agent_1.action(self.state)
                # update state based on the above action
                self.state = self.update_state(action_2)
                # reward agent if the move was correct, otherwise penalize him

            # update epslion for the agents
            self.agent_1.epsilon -= linear_decay_rate

    def check_board(self):
        # check if all tiles are colored in order to end episode
        if np.max(self.state) != 0:
            return True
        else:
            return False

    def update_state(self, action: dict):
        # turn state from tuple to list
        state = list(self.state)
        # update its value
        state[action["tile"]] = action["action"]
        # return it to tuple again
        return tuple(state)

    # def get_state():
    #     for i in range(self.state.shape[0]):
    #         if np.max(state[i]) > 0:
    #             self.colored.append(i)
    #         else:
    #             self.blank.append(i)
