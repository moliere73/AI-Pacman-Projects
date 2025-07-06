from agent001 import Agent001
from agent002 import Agent002
from agent003 import Agent003
from agent004 import Agent004
from agent005 import Agent005
from agent006 import Agent006
from agent007 import Agent007
from agent008 import Agent008
from agentRL import AgentRL
from nim import Nim

initial_num_pieces = 11

walle = Agent007("WALL-E", initial_num_pieces)
bb8 = AgentRL("BB-8", initial_num_pieces)

agents = (walle, bb8)

game = Nim(agents, initial_num_pieces)
game.play()

for agent in agents:
    agent.print_stats()

