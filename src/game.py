import random

import numpy as np
from tqdm import tqdm

from agent import Agent


class BaseGame:
    """
    Base Game class with methods every game is going to use regardless of the rules.
    """

    def __init__(self, environment_size: tuple, reward: int, neg_reward: int = 0):
        self.reward = reward
        self.neg_reward = neg_reward

        self.environment_size = environment_size

        self.state = self.reset_state(self.environment_size)

    def reset_state(self, environment_size: tuple):
        return tuple(
            np.zeros((environment_size[0], environment_size[1])).flatten().astype(int)
        )

    def check_board(self):
        # check if all tiles are colored in order to end episode
        if np.min(self.state) != 0:
            return False
        else:
            return True

    def give_reward(self, action_legitimacy: bool):
        if action_legitimacy:
            return self.neg_reward
        else:
            return self.reward

    def update_state(self, action: dict):
        # turn state from tuple to list
        state = list(self.state)
        # update its value
        state[action["tile"]] = action["action"]
        # return it to tuple again
        return tuple(state)

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
            if self.state[item] == action["action"]:
                wrong_action = True
                break

        return wrong_action


class GameBOne(BaseGame):
    """
    This Game class refers to the B.1 game rules, where agents play one after the other and
    have knowledge of other agents actions.
    """

    def __init__(
        self,
        environment_size: tuple,
        possible_tile_states: tuple,
        actions: tuple,
        lr: float,
        gamma: float,
        reward: int,
        neg_reward: int = 0,
    ):
        super().__init__(environment_size, reward)

        # create the 2 agents (it's self play so the 2nd agent is the same one)
        self.agent_1 = Agent(environment_size, possible_tile_states, actions, lr, gamma)

    def play(self, epochs):
        exploration_end = int(0.8 * epochs)
        linear_decay_rate = 1 / exploration_end
        for i in tqdm(range(epochs), bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"):
            print(f"Epoch: {i}")
            while self.check_board():
                # get 1st agent's action
                action_1 = self.agent_1.action(self.state)
                # update state based on the above action
                new_state = self.update_state(action_1)
                # reward agent if the move was correct, otherwise penalize him
                action_legitimacy = self.check_action(action_1)
                reward = self.give_reward(action_legitimacy)

                print(self.state, action_1, new_state, reward)
                # update agent's Q table
                self.agent_1.update_table(self.state, action_1, new_state, reward)
                # in case first action caused the episode to end
                self.state = new_state
                if not self.check_board():
                    break
                # get 2nd agent's action
                action_2 = self.agent_1.action(self.state)
                # update state based on the above action
                new_state = self.update_state(action_2)
                # reward agent if the move was correct, otherwise penalize him
                action_legitimacy = self.check_action(action_2)
                reward = self.give_reward(action_legitimacy)
                # update agent's Q table
                self.agent_1.update_table(self.state, action_2, new_state, reward)

                print(self.state, action_2, new_state, reward)

                self.state = new_state

            # update epsilon for the agents
            self.agent_1.epsilon -= linear_decay_rate
            # reset the state
            self.state = self.reset_state(self.environment_size)


class GameBTwo(BaseGame):
    """
    This Game class refers to the B.2 game rules, where agents play one after the other and
    have only know the results of their own actions.
    """

    def __init__(
        self,
        environment_size: tuple,
        possible_tile_states: tuple,
        actions: tuple,
        lr: float,
        gamma: float,
        reward: int,
        neg_reward: int = 0,
    ):
        super().__init__(environment_size, reward)

        # create the 2 agents (it's self play so the 2nd agent is the same one)
        self.agent_1 = Agent(environment_size, possible_tile_states, actions, lr, gamma)

    def play(self, epochs: int):
        exploration_end = int(0.8 * epochs)
        linear_decay_rate = 1 / exploration_end
        for i in tqdm(range(epochs), bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"):
            print(f"Epoch: {i}")
            state_1 = self.state
            state_2 = self.state
            while self.check_perspective(state_1) and self.check_perspective(state_2):
                # get 1st agent's action
                action_1 = self.agent_1.action(state_1)
                if self.state[action_1["tile"]] != 0:
                    reward_1 = -1
                    new_state_1 = self.change_perspective(state_1, action_1)
                else:
                    # regular sequence of functions like B.1
                    new_state_1 = self.update_perspective(state_1, action_1)
                    action_legitimacy = self.check_action(action_1)
                    reward_1 = self.give_reward(action_legitimacy)
                    self.state = self.update_state(action_1)
                print(state_1, action_1, new_state_1, reward_1)
                # update agent's Q table
                self.agent_1.update_table(state_1, action_1, new_state_1, reward_1)
                state_1 = new_state_1
                # self.state = new_state_1
                action_2 = self.agent_1.action(state_2)
                if self.state[action_2["tile"]] != 0:
                    reward_2 = -1
                    new_state_2 = self.change_perspective(state_2, action_2)
                else:
                    # regular sequence of functions like B.1
                    new_state_2 = self.update_perspective(state_2, action_2)
                    action_legitimacy = self.check_action(action_2)
                    reward_2 = self.give_reward(action_legitimacy)
                    self.state = self.update_state(action_2)
                print(state_2, action_2, new_state_2, reward_2)
                # update agent's Q table
                self.agent_1.update_table(state_2, action_2, new_state_2, reward_2)
                state_2 = new_state_2
            # update epsilon for the agents
            self.agent_1.epsilon -= linear_decay_rate
            # reset the state
            self.state = self.reset_state(self.environment_size)

    def change_perspective(self, state: tuple[int], action: dict):
        # this is called when the agent tries to color an already colored tile
        # and just keeps the state as is, but update the agent's knowledge about
        # the tile he tried to color
        new_state = state
        new_state = list(new_state)
        new_state[action["tile"]] = self.state[action["tile"]]
        new_state = tuple(new_state)

        return new_state

    def check_perspective(self, state):
        # check if all tiles are colored in order to end episode
        if np.min(state) != 0:
            return False
        else:
            return True

    def update_perspective(self, state: tuple[int], action: dict):
        # turn state from tuple to list
        list_state = list(state)
        # update its value
        list_state[action["tile"]] = action["action"]
        # return it to tuple again
        return tuple(list_state)


class GameAOne(BaseGame):
    """
    This Game class refers to the A.1 game rules, where agents act simultaneously and
    have knowledge of other agent's actions.
    """

    def __init__(
        self,
        environment_size: tuple,
        possible_tile_states: tuple,
        actions: tuple,
        lr: float,
        gamma: float,
        reward: int,
        neg_reward: int = 0,
    ):
        super().__init__(environment_size, reward)

        # create the 2 agents (it's self play so the 2nd agent is the same one)
        self.agent_1 = Agent(environment_size, possible_tile_states, actions, lr, gamma)

    def play(self, epochs: int):
        exploration_end = int(0.8 * epochs)
        linear_decay_rate = 1 / exploration_end
        for i in tqdm(range(epochs), bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"):
            print(f"Epoch: {i}")
            while self.check_board():
                # get the agents' actions
                action_1 = self.agent_1.action(self.state)
                action_2 = self.agent_1.action(self.state)
                # check if they try to color the same tile
                # if they do, choose one action in random and penalize the other
                if action_1["tile"] == action_2["tile"]:
                    starting_state = self.state
                    number = random.choice([1, 2])
                    alt_reward = -1
                    if number == 1:
                        action = action_1
                        wrong_action = action_2
                    elif number == 2:
                        action = action_2
                        wrong_action = action_1
                    # reward agent if the move was correct, otherwise penalize him
                    action_legitimacy = self.check_action(action)
                    reward = self.give_reward(action_legitimacy)
                    # update state based on the above action
                    self.state = self.update_state(action)
                    # update agent's Q table
                    self.agent_1.update_table(
                        starting_state, action, self.state, reward
                    )
                    # the other agents move is penalized
                    self.agent_1.update_table(
                        starting_state, wrong_action, self.state, alt_reward
                    )

                    print(starting_state, action, self.state, reward)
                    print(starting_state, wrong_action, self.state, alt_reward)
                else:
                    starting_state = self.state
                    # reward agent if the move was correct, otherwise penalize him
                    action_legitimacy_1 = self.check_action(action_1)
                    reward_1 = self.give_reward(action_legitimacy_1)
                    # reward agent if the move was correct, otherwise penalize him
                    action_legitimacy_2 = self.check_action(action_2)
                    reward_2 = self.give_reward(action_legitimacy_2)
                    # update state based on the above actions
                    self.state = self.update_state(action_1)
                    self.state = self.update_state(action_2)

                    self.agent_1.update_table(
                        starting_state, action_1, self.state, reward_1
                    )
                    self.agent_1.update_table(
                        starting_state, action_2, self.state, reward_2
                    )

                    print(starting_state, action_1, self.state, reward_1)
                    print(starting_state, action_2, self.state, reward_2)

            # update epsilon for the agents
            self.agent_1.epsilon -= linear_decay_rate
            # reset the state
            self.state = self.reset_state(self.environment_size)


game = GameBTwo(
    environment_size=(3, 3),
    possible_tile_states=(0, 1, 2),
    actions=(1, 2),
    lr=1e-4,
    gamma=0.95,
    reward=1,
    neg_reward=0,
)

game.play(600)

# game = GameBOne(
#     environment_size=(3, 3),
#     possible_tile_states=(0, 1, 2),
#     actions=(1, 2),
#     lr=1e-4,
#     gamma=0.95,
#     reward=1,
#     neg_reward=0,
# )

# game.play(2000)

# game = GameAOne(
#     environment_size=(3, 3),
#     possible_tile_states=(0, 1, 2),
#     actions=(1, 2),
#     lr=1e-4,
#     gamma=0.95,
#     reward=1,
#     neg_reward=0,
# )

# game.play(4000)
