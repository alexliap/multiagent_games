from agent import Agent
import math

class Game:
    def __init__(state_space: int, action_space: int, epochs: int, ):
        self.state=tuple(np.zeros((state_space, action_space)).flatten())
        
        self.end_of_exploration=math.ceil(epochs*0.8)
        
        self.blanks=[]
        self.colored=[]
        self.get_state()
        
        # create the 2 agents
        self.agent_1=Agent(state_space, action_space)
        self.agent_2=Agent(state_space, action_space)
        
    def reset_state():
        self.state=np.zeros((9, 2))
        
    def get_state():
        for i in range(self.state.shape[0]):
            if np.max(state[i]) > 0:
                self.colored.append(i)
            else:
                self.blank.append(i)
                