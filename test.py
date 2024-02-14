import numpy as np
import random
from random import choice


environment_rows = 3
environment_columns = 3
#initialize the enviroment with all blocks as white
enviroment = np.full((environment_rows, environment_columns), 'white')

epsilon = 1

#All the possible actions of the agent based on the enviroment
possible_actions = {(1,(0,0)):'green',(2,(0,0)):'red', (3,(0,1)):'green',(4,(0,1)):'red',(5,(0,2)):'green',(6,(0,2)):'red',
                    (7,(1,0)):'green',(8,(1,0)):'red', (9,(1,1)):'green',(10,(1,1)):'red',(11,(1,2)):'green',(12,(1,2)):'red',
                    (13,(2,0)):'green',(14,(2,0)):'red', (15,(2,1)):'green',(16,(2,1)):'red',(17,(2,2)):'green',(18,(2,2)):'red',}

#Agent Knows environment
for episode in range(100):
    #Agent knows the remaining white blocks
    possible_possistions = np.argwhere(enviroment == 'white')
    #initialize a reward dictionary
    rewards = []
    if (possible_possistions.size !=0):
        #exploration phase
        if (epsilon>=0.3):
            key, value = random.choice(list(possible_actions.items()))
            enviroment[key[1][0],key[1][1]] = value
            epsilon = epsilon-0.01
            reward=0
            #check the positions of colored blocks in order to reward
            positions_of_green_blocks = np.argwhere(enviroment == 'green')
            positions_of_red_blocks = np.argwhere(enviroment == 'red')
            #the side blocks that we want to check
            check_list = [[key[1][0]+1,key[1][1]], 
                              [key[1][0]-1,key[1][1]],
                              [key[1][0],key[1][1]-1],
                              [key[1][0],key[1][1]+1]]
            #if the selection is green and there is no red in side blocks then reward
            if (value == 'green'):
                for i in check_list:
                    for j in positions_of_red_blocks.tolist():
                        if i == j:
                            reward = 'a'
                        else:
                            reward = 0
                        rewards.append(reward)
            #if the selection is red and there is no green in side blocks then reward            
            if (value == 'red'):
                for k in check_list:
                    for l in positions_of_green_blocks.tolist():
                        if k == l:
                            reward = 'a'
                        else:
                            reward = 0
                        rewards.append(reward)
            print(rewards)        
        #exploitation phase  
        else:
            pass
    pass
  
