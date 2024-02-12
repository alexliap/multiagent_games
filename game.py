from agent import Agent
import math
import numpy as np
from tqdm import tqdm

class GameBOne:
    """
    This Game class refers to the B.1 game rules, where agents play one after the other and 
    have knowledge of their other agents actions.
    """
    def __init__(state_space: int, action_space: int):
        self.state=self.reset_state()
        
        # self.exploration_end=int(0.8*epochs)
        
        self.get_state()
        
        # create the 2 agents
        self.agent_1=Agent(state_space, action_space)
        # self.agent_2=Agent(state_space, action_space)
        
    def reset_state(self):
        return tuple(np.zeros((state_space, action_space)).flatten())
    
    def play(self, epochs):
        exploration_end=int(0.8*epochs)
        linear_decay_rate=100/exploration_end
        for i in tqdm(range(epochs), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            while self.check_board():
                # get 1st agent's action
                action_1=self.agent_1.action(self.state)
                # update state based on the above action
                self.state=self.update_state(action_1)
                # reward agent if the move was correct, otherwise penalize him
                
                # get 2nd agent's action
                action_2=self.agent_1.action(self.state)
                # update state based on the above action
                self.state=self.update_state(action_2)
                # reward agent if the move was correct, otherwise penalize him
                
            
            
    def check_board(self):
        # check if all tiles are colored in order to end episode
        if np.max(self.state) != 0:
            return True
        else:
            return False      
            
    def update_state(self, action: dict):
        # turn state from tuple to list
        state=list(self.state)
        # update its value
        state[action['tile']]=action['action']
        # return it to tuple again
        return tuple(state)
        
        
        
    # def get_state():
    #     for i in range(self.state.shape[0]):
    #         if np.max(state[i]) > 0:
    #             self.colored.append(i)
    #         else:
    #             self.blank.append(i)
                