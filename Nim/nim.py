'''
Play game where players (Agents) take turns selecting
one or two of the remaining pieces. The player that
takes the last piece wins.
'''

import random
from agent import Agent
from agent001 import Agent001
from agent002 import Agent002
from agent003 import Agent003
from agent004 import Agent004
from agent005 import Agent005
from agent006 import Agent006
from agent007 import Agent007
from agent008 import Agent008
from agentRL import AgentRL

class Nim:
    def __init__(self, agents, init_num_pieces):
        self.agents = agents
        self.num_pieces = init_num_pieces
        
    def play(self, display=True):
        turn = random.randrange(0,2)
        
        if display:
            print("Nim!!\n")
        
        while self.num_pieces > 0:
            
            if display:
                print("{} pieces remaining".format(self.num_pieces))
            
            num_selected = self.agents[turn].getAction(self.num_pieces)

            if num_selected != 1 and num_selected != 2 :
                print("Bogus! Can't select {}".format(num_selected))
                return

            self.num_pieces = self.num_pieces - num_selected
            
            if display:
                print("\t{} picks {}".format(self.agents[turn], num_selected))
            
            turn = (turn+1) % len(self.agents)            

        self.agents[(turn+1) % len(self.agents)].win()
        self.agents[turn].lose()
        
        if display:
            print("{} wins!\n".format(self.agents[(turn+1) % len(self.agents)]))

if __name__ == '__main__':
    initial_num_pieces = 11

    hal = Agent001("HAL", initial_num_pieces)
    walle = Agent007("WALL-E", initial_num_pieces)
    bb8 = AgentRL("BB-8", initial_num_pieces)

    game = Nim((walle, hal), initial_num_pieces)
    game.play()
    
