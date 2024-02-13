import numpy as np
from tqdm import tqdm

from agent import Agent


class GameBOne:
    """
    This Game class refers to the B.1 game rules, where agents play one after the other and
    have knowledge of their other agents actions.
    """

    def __init__(
        self,
        environment_size: tuple,
        possible_tile_states: tuple,
        actions: tuple,
        reward: int,
        neg_reward: int = 0,
    ):
        self.reward = reward
        self.neg_reward = neg_reward

        self.environment_size = environment_size

        self.state = self.reset_state(self.environment_size)

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
                action_legitimacy = self.check_action(action_1)
                reward = self.give_reward(action_legitimacy)
                # update agent's Q table
                self.agent_1.update_table(reward)
                # in case first action caused the episode to end
                if self.check_board():
                    continue
                # get 2nd agent's action
                action_2 = self.agent_1.action(self.state)
                # update state based on the above action
                self.state = self.update_state(action_2)
                # reward agent if the move was correct, otherwise penalize him
                action_legitimacy = self.check_action(action_2)
                reward = self.give_reward(action_legitimacy)
                # update agent's Q table
                self.agent_1.update_table(reward)

            # update epslion for the agents
            self.agent_1.epsilon -= linear_decay_rate
            # reset the state
            self.state = self.reset_state(self.environment_size)

    def check_action(self, action: dict):
        # check if the action was correct or not
        tile = action["tile"]
        wrong_action = False
        if tile == 0:
            neighbors = [1, 3]
        elif tile == 1:
            neighbors = [0, 2, 4]
        elif tile == 2:
            neighbors = [1, 5]
        elif tile == 3:
            neighbors = [0, 6, 4]
        elif tile == 4:
            neighbors = [1, 7, 3, 5]
        elif tile == 5:
            neighbors = [2, 8, 4]
        elif tile == 6:
            neighbors = [3, 7]
        elif tile == 7:
            neighbors = [6, 4, 8]
        else:
            neighbors = [7, 5]

        for item in neighbors:
            if self.state[item] != action["action"]:
                wrong_action = True

        return wrong_action

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

    def give_reward(self, action_legitimacy: bool):
        if action_legitimacy:
            return self.reward
        else:
            return self.neg_reward
