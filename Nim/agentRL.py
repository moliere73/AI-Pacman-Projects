import random
from agent import Agent

class AgentRL(Agent):
    def __init__(self, name, initial_num_pieces, epsilon=1.0):
        Agent.__init__(self, name, initial_num_pieces)
        
        self.epsilon = epsilon 
        self._explore_mode = True
        
        self.num_games_dict = {}
        self.num_wins_dict = {}
        for state in range(1,initial_num_pieces+1):
            for action in (1,2):
                self.num_games_dict[(state, action)] = 0
                self.num_wins_dict[(state, action)] = 0
        
        # List of (state, action) pairs this agent has played in the current game.
        # When winning or losing, we use this history to update the num_wins_dict and num_games_dict.
        # Should reset to [] after winning or losing.
        self.current_game_history = []
        
    def getAction(self, state):
        if self._explore_mode == True and random.random() < self.epsilon:
            action = random.randrange(1,3)
            action = min(action, state) # Don't choose more that what is left
        else:
            probability_win_one = self.getWinProbability(state, 1)
            probability_win_two = self.getWinProbability(state, 2)
            
            if probability_win_one > probability_win_two:
                action = 1
            elif probability_win_one < probability_win_two:
                action = 2
            else:
                # Both the same, so choose random
                action = random.randrange(1,3)
                action = min(action, state) # Don't choose more that what is left
            
        # Keep track of moves
        self.current_game_history.append((state, action))
        
        return action
        
    def getWinProbability(self, state, action):
        num_games = self.num_games_dict[(state, action)]
        if num_games > 0:
            prob = self.num_wins_dict[(state, action)] * 1.0 / num_games
        else:
            prob = 0.5
        return prob
    
    def win(self):
        # Call Agent.win()
        super().win()

        for (state,action) in self.current_game_history:
            self.num_games_dict[(state, action)] += 1
            self.num_wins_dict[(state, action)] += 1
        self.current_game_history = []
            
    def lose(self):
        # Call Agent.lose()
        super().lose()

        for (state,action) in self.current_game_history:
            self.num_games_dict[(state, action)] += 1
        self.current_game_history = []
    
    @property
    def explore_mode(self):
        return self._explore_mode
        
    @explore_mode.setter
    def explore_mode(self, explore_mode):
        self._explore_mode = explore_mode

    def print_stats(self):
        # Call Agent.print_stats
        super().print_stats()

        agent_string = "\t{} Statistics\n".format(self.name)

        agent_string += "\tState"
        for action in (1,2):
            agent_string += "\tact={}".format(action)
        agent_string += "\n"
    
        agent_string += "\t-----"
        for action in (1,2):
            agent_string += "\t-----".format(action)
        agent_string += "\n"
    
        for state in range(self.initial_num_pieces,0,-1):
            agent_string += "\t{}".format(state)

            for action in (1,2):
                num_games = self.num_games_dict[(state,action)]
                if num_games > 0:
                    percent = self.num_wins_dict[(state,action)] * 100.0 / num_games
                    agent_string += "\t{:0.1f}".format(percent)
                else:
                    agent_string += "\tN/A"        
            agent_string += "\n"

        print(agent_string)

