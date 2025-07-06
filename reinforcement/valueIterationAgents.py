# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
from typing import Tuple
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def getValue(self, state):
        """
        Returns the value of the state as computed in value iteration.
        If the state is not found, returns 0.0 as a default.
        """
        return self.values.get(state, 0.0)

    def runValueIteration(self) -> None:
        """
        Runs self.iterations iterations of value iteration
        updates self.values, does not return anything
        """
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for _ in range(self.iterations):
        # Create a temporary copy of values to reference old values during updates
            temp_values = self.values.copy()

            for state in self.mdp.getStates():
                # Initialize a list to store all the Q-values of possible actions from this state
                action_values = []

                if not self.mdp.isTerminal(state):
                    for action in self.mdp.getPossibleActions(state):
                        q_value = sum(prob * (self.mdp.getReward(state, action, next_state) + 
                                            self.discount * temp_values[next_state])
                                    for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action))
                        action_values.append(q_value)

                # Update the value of the state to the max of the computed Q-values, if any actions were available
                self.values[state] = max(action_values, default=0.0)

    def computeQValueFromValues(self, state: Tuple[int,int], action: str) -> float:
        """
        Computes the Q-value of a state-action pair by summing the value of all possible next states.
        """
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.

          state: a state in the mdp
          action: the action we took at that state
          return: float representing Q(state,action)
        """
        QValue = 0
        for state_next, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, state_next)
            QValue += prob * (reward + self.discount * self.getValue(state_next))
        return QValue

    def computeActionFromValues(self, state: Tuple[int,int]) -> str:
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.

          state: state in mdp
          return: action type, represents best action from state (None if state is terminal)
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)
    
        # Check if there are no legal actions (e.g., terminal state)
        if not actions:
            return None

        # Compute the Q-value for each action and choose the action with the highest Q-value
        QValues = {action: sum(prob * (self.mdp.getReward(state, action, next_state) + 
                                        self.discount * self.values[next_state])
                                for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action))
                    for action in actions}

        # Find the action with the maximum Q-value
        max_action = max(QValues, key=QValues.get, default=None)

        return max_action


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
    

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        # Since this class inherits from ValueIterationAgent, call its __init__
        # This is necessary if the superclass does more than just runValueIteration
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        """
        Runs self.iterations iterations of async value iteration, only updating one state in each iteration
        updates self.values, does not return anything
        """ 
        "*** YOUR CODE HERE ***"
        
        states = self.mdp.getStates()
        for i in range(self.iterations):
            state = states[i % len(states)]  # Cyclically pick a state
            
            if not self.mdp.isTerminal(state):
                max_value = float('-inf')
                for action in self.mdp.getPossibleActions(state):
                    q_value = self.computeQValueFromValues(state, action)
                    if q_value > max_value:
                        max_value = q_value
                if max_value != float('-inf'):  # Update the value if there are actions
                    self.values[state] = max_value
        

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self) -> None:
        """
        Runs self.iterations iterations of prioritized sweeping value iteration
        updates self.values, does not return anything
        """
        # Compute predecessors of all states
        predecessors = {}
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        if prob > 0:
                            if nextState not in predecessors:
                                predecessors[nextState] = set()
                            predecessors[nextState].add(state)

        # Initialize an empty priority queue
        priority_queue = util.PriorityQueue()

        # Initialize a dictionary to hold the current diffs for states in the queue
        current_diffs = {}

        # For each non-terminal state, find the difference between
        # the current value of the state and the highest Q-value across all possible actions
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                max_q_value = max(self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state))
                diff = abs(self.values[state] - max_q_value)
                priority_queue.push(state, -diff)
                current_diffs[state] = diff

        # Update states' values in the queue for the specified number of iterations
        for iteration in range(self.iterations):
            if priority_queue.isEmpty():
                break

            # Pop the state off the priority queue
            state = priority_queue.pop()
            current_diffs.pop(state, None)  # Remove this state's diff from the current_diffs

            # If the state is not a terminal state, update its value
            if not self.mdp.isTerminal(state):
                self.values[state] = max(self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state))

            # For each predecessor of the popped state, update its priority in the queue
            for p in predecessors.get(state, set()):
                if not self.mdp.isTerminal(p):
                    max_q_value = max(self.computeQValueFromValues(p, action) for action in self.mdp.getPossibleActions(p))
                    diff = abs(self.values[p] - max_q_value)
                    # If diff is greater than theta and the state's diff has improved, push it into the queue
                    if diff > self.theta and diff > current_diffs.get(p, -1):
                        priority_queue.push(p, -diff)
                        current_diffs[p] = diff  # Update the current diff for this state
            
        