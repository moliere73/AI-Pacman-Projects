import random
from agent import Agent

class Agent003(Agent):
    def __init__(self, name, initial_num_pieces):
        Agent.__init__(self, name, initial_num_pieces)
        self.prev_state = self.initial_num_pieces
        
    def getAction(self, state):
        # If at the start of the game
        if state == self.initial_num_pieces :
            action = random.randrange(1,3)
            self.prev_state = state-action
            return action
        else :
            prev_move = self.prev_state-state
            action = prev_move; # Same
            action = min(action, state); # Don't choose more that what is left
            self.prev_state = state-action
            return action

    def win(self):
        super().win()
        self.prev_state = self.initial_num_pieces

    def lose(self):
        super().lose()
        self.prev_state = self.initial_num_pieces


