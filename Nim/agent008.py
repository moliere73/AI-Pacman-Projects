import random
from agent import Agent

class Agent008(Agent):
    def __init__(self, name, initial_num_pieces):
        Agent.__init__(self, name, initial_num_pieces)
        
    def getAction(self, state):
        if state % 3 == 2 :
            return 2
        elif state % 3 == 1 : 
            return 1
        else:
            # Doesn't matter, so choose random
            action = random.randrange(1,3)
            return min(action, state) # Don't choose more that what is left

