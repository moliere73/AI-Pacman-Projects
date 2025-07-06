# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math
from typing import Optional, Tuple

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.q_values = util.Counter()

    def getQValue(self, state: Tuple[int,int], action: str) -> float:
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise

          state: represents the current state
          action: represents action taken
          return: float, representing Q(state,action)
        """
        "*** YOUR CODE HERE ***"
        return self.q_values[(state, action)]


    def computeValueFromQValues(self, state: Tuple[int, int]) -> float:
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.

          state: represents the current state
          return: float, representing V(state) = max_{a} Q(state,a)
        """
        "*** YOUR CODE HERE ***"
        legal_actions = self.getLegalActions(state)
        if not legal_actions:  # Empty list of legal actions
            return 0.0
        return max(self.getQValue(state, action) for action in legal_actions)

    def computeActionFromQValues(self, state: Tuple[int,int]) -> Optional[str]:
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.

          state: represents the current state
          return: action, representing best action to take at state according to Q-values, None if not possible
        """
        "*** YOUR CODE HERE ***"
        legal_actions = self.getLegalActions(state)
        if not legal_actions:  # Empty list of legal actions
            return None

        # Find action(s) with the highest Q-value
        max_value = self.computeValueFromQValues(state)
        best_actions = [action for action in legal_actions if self.getQValue(state, action) == max_value]

        # Break ties randomly
        return random.choice(best_actions) if best_actions else None

    def update(self, state: Tuple[int,int], action: str, nextState: Tuple[int,int], reward: float) -> None:
        """
          The parent class calls this upon observing a
          (state => action => nextState and reward) transition.
          You should do your Q-value update here.

          NOTE: You should never call this function,
          it will be called on your behalf

          state: represents the current state
          action: represents action taken
          nextState: represents the resulting state (s')
          reward: float, represents the immediate reward gained, R(s,a,s')
          this method should update class variables, return nothing
        """
        "*** YOUR CODE HERE ***"
        sample = reward + self.discount * self.computeValueFromQValues(nextState)
        self.q_values[(state, action)] = ((1 - self.alpha) * self.getQValue(state, action)) + (self.alpha * sample)
    
    def getAction(self, state: Tuple[int,int]) -> str:
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)

          state: represents the current state
          return: action, representing the action taken according to the epsilon greedy policy
        """
        # Pick Action
        #legalActions = self.getLegalActions(state)
        #action = None
        legal_actions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"

        if not legal_actions:  # Empty list of legal actions
            return action

        if util.flipCoin(self.epsilon):  # With probability self.epsilon, take a random action
            action = random.choice(legal_actions)
        else:  # Take the best action according to Q-values
            action = self.computeActionFromQValues(state)
        return action

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

class QLearningAgentCountExploration(QLearningAgent):
    def __init__(self, k=2, **args):
        self.visitCount = util.Counter() 
        self.k = k
        QLearningAgent.__init__(self, **args)
    
    # Feel free to add helper functions here
    "*** YOUR CODE HERE ***"

    def update(self, state: Tuple[int,int], action: str, nextState: Tuple[int,int], reward: float) -> None:
        """
          The parent class calls this upon observing a
          (state => action => nextState and reward) transition.
          You should do your Q-value update here.

          You should update the visit count in this function 
          for the current state action pair.

          NOTE: You should never call this function,
          it will be called on your behalf

          state: represents the current state
          action: represents action taken
          nextState: represents the resulting state (s')
          reward: float, represents the immediate reward gained, R(s,a,s')
          this method should update class variables, return nothing
        """
        "*** YOUR CODE HERE ***"
        # Increment the visit count for the (state, action) pair
        self.visitCount[(state, action)] += 1

        # Compute the exploration function for each action in the next state
        f_values = [self.computeFValue(nextState, a) for a in self.getLegalActions(nextState)]

        # Use the modified Q-update equation with the exploration function values
        sample = reward + self.discount * max(f_values, default=0)  # Use default=0 for terminal states with no actions
        self.q_values[(state, action)] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * sample

    def computeFValue(self, state, action):
        """
        Compute the exploration function value for a (state, action) pair.
        """
        u = self.getQValue(state, action)  # The current Q-value
        n = self.visitCount[(state, action)]  # The visit count
        return u + self.k / (n + 1)


    def getAction(self, state: Tuple[int,int]) -> str:
        """
          Compute the action to take in the current state.
          Break ties randomly.

          state: represents the current state
          return: action, representing the action taken according to the visit count based exploration policy
        """
        action = None
        "*** YOUR CODE HERE ***"
        """
        Compute the action to take in the current state, including exploration.
        With probability self.epsilon, we take a random action.
        With probability 1 - self.epsilon, take the best policy action.
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:  # No legal actions available
            return None

        # Choose action based on exploration function rather than just Q-values
        f_values = [(self.computeFValue(state, action), action) for action in legalActions]
        best_value, best_action = max(f_values, key=lambda x: x[0])

        # Choose the best action with probability 1 - epsilon and choose a random action otherwise
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        else:
            return best_action




class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state: Tuple[int,int], action: str) -> float:
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
          
          state: represents the current state
          action: represents action taken
          return: float, representing Q_w(state,action)
        """
        "*** YOUR CODE HERE ***"
        # Get the feature vector for the current state and action
        features = self.featExtractor.getFeatures(state, action)
        # Compute the dot product of the feature vector and the weights
        q_value = sum(self.weights[feature] * value for feature, value in features.items())
        return q_value
        

    def update(self, state: Tuple[int,int], action: str, nextState: Tuple[int,int], reward: float) -> None:
        """
          Should update your weights based on transition

          state: represents the current state
          action: represents action taken
          nextState: represents the resulting state (s')
          reward: float, represents the immediate reward gained, R(s,a,s')
          this method should update class variables, return nothing
        """
        "*** YOUR CODE HERE ***"
        # Compute the difference part of the update equation
        difference = (reward + self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(state, action)
        # Get the features from the feature extractor
        features = self.featExtractor.getFeatures(state, action)
        # Update each feature's weight using the difference and feature value
        for feature in features:
            self.weights[feature] += self.alpha * difference * features[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)
        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
