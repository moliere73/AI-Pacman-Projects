import random
from agent import Agent

class Agent002(Agent):
    def __init__(self, name, initial_num_pieces):
        Agent.__init__(self, name, initial_num_pieces)
        
    def getAction(self, state):
        action = 2
        return min(action, state)  # Don't choose more that what is left

