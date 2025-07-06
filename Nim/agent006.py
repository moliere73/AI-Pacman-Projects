import random
from agent import Agent

class Agent006(Agent):
    def __init__(self, name, initial_num_pieces):
        Agent.__init__(self, name, initial_num_pieces)
        
    def getAction(self, state):
        if state % 3 == 2 :
            return 1
        else : 
            return 2
