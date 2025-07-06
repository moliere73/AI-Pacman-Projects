# search.py
# ---------
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

import util
import sys
import copy

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def goalTest(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getActions(self, state):
        """
        Given a state, returns available actions.
        Returns a list of actions
        """        
        util.raiseNotDefined()

    def getResult(self, state, action):
        """
        Given a state and an action, returns resulting state.
        """
        util.raiseNotDefined()

    def getCost(self, state, action):
        """
        Given a state and an action, returns step cost, which is the incremental cost 
        of moving to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

class Node:
    """
    Search node object for your convenience.

    This object uses the state of the node to compare equality and for its hash function,
    so you can use it in things like sets and priority queues if you want those structures
    to use the state for comparison.

    Example usage:
    >>> S = Node("Start", None, None, 0)
    >>> A1 = Node("A", S, "Up", 4)
    >>> B1 = Node("B", S, "Down", 3)
    >>> B2 = Node("B", A1, "Left", 6)
    >>> B1 == B2
    True
    >>> A1 == B2
    False
    >>> node_list1 = [B1, B2]
    >>> B1 in node_list1
    True
    >>> A1 in node_list1
    False
    """
    def __init__(self, state, parent, action, path_cost):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other):
        return self.state == other.state

    def __ne__(self, other):
        return self.state != other.state


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def breadthFirstSearch(problem):
    visited = set()  
    queue = util.Queue()  
    initial = problem.getStartState()  
    root = Node(initial, None, None, 1) 
    queue.push(root)  

    while True:
        if queue.isEmpty():  
            return []
        currentNode = queue.pop() 
        if problem.goalTest(currentNode.state): 
            return reconstructPath(problem, currentNode)  
        visited.add(currentNode)  
        for action in problem.getActions(currentNode.state):  
            nextState = problem.getResult(currentNode.state, action)
            nextNode = Node(nextState, currentNode, action, currentNode.path_cost + 1)
            if (not queue.contains(nextNode) and 
                nextNode not in visited):  
                queue.push(nextNode)  

def reconstructPath(problem, currentNode):
    trajectory = []  # List to store the path from goal to start
    while problem.getStartState() != currentNode.state:  # Trace back from goal to start
        trajectory.insert(0, currentNode.action)  # Insert action at the beginning of the trajectory
        currentNode = currentNode.parent  # Move to the parent node
    return trajectory  # Return the path from start to goal

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def depthFirstLimitedSearch(problemInstance, maxDepth):
    visitedNodes = set()  
    explorationStack = util.Stack()  
    initialState = problemInstance.getStartState()  
    initialNode = Node(initialState, None, None, 1)  
    explorationStack.push(initialNode)  

    while True:
        if explorationStack.isEmpty():  # Check if there are no more nodes to explore
            return []
        currentNode = explorationStack.pop()  # Pop the next node to explore from the stack
        if problemInstance.goalTest(currentNode.state):  
            return reconstructPath(problemInstance, currentNode) 
        visitedNodes.add(currentNode)  
        if currentNode.path_cost == maxDepth:  # Limit exploration to the specified depth
            continue
        for action in problemInstance.getActions(currentNode.state):  
            nextStep = problemInstance.getResult(currentNode.state, action)
            nextNode = Node(nextStep, currentNode, action, currentNode.path_cost + 1)
            if (not explorationStack.contains(nextNode) and 
                nextNode not in visitedNodes):  # Ensure the node is neither in the stack nor visited
                explorationStack.push(nextNode)  

def iterativeDeepeningSearch(problem):
    """
    Executes the Depth-First Search algorithm iteratively with increasing depth limits to
    find the shortest path to the goal. Starts with a depth limit of 1, increasing the depth
    by one with each iteration until the goal is found.

    This approach combines the space efficiency of DFS with the completeness of BFS, making
    it particularly useful for searching large spaces where the depth of the solution is unknown.

    Note: "nodes expanded" refers to the count of nodes where getActions is invoked. To align with
    grading metrics, ensure to increment depth post goal checking but prior to executing getActions.
    """
    currentDepth = 1  # Initialize depth counter
    while True:  # Infinite loop to increase depth progressively
        searchOutcome = depthFirstLimitedSearch(problem, currentDepth)
        if searchOutcome:  # If a solution is found at the current depth
            return searchOutcome  # Return the solution
        currentDepth += 1  # Increment depth for the next iteration


    """
    Perform DFS with increasingly larger depth. Begin with a depth of 1 and increment depth by 1 at every step.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.goalTest(problem.getStartState()))
    print("Actions from start state:", problem.getActions(problem.getStartState()))

    Then try to print the resulting state for one of those actions
    by calling problem.getResult(problem.getStartState(), one_of_the_actions)
    or the resulting cost for one of these actions
    by calling problem.getCost(problem.getStartState(), one_of_the_actions)

    Note: In the autograder, "nodes expanded" is equivalent to the nodes on which getActions 
    was called. To make the autograder happy, do the depth check after the goal test but before calling getActions.

    """


def aStarSearch(problem, heuristic=nullHeuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    Conducts the A* search algorithm to identify the most efficient path to the goal,
    prioritizing by lowest cost from start combined with heuristic estimate to the goal.
    "*** YOUR CODE HERE ***"
    """
    checkedStates = set()  
    entryPoint = problem.getStartState()  
    initialNode = Node(entryPoint, None, None, 0)  
    frontier = util.PriorityQueueWithFunction(lambda node: node.path_cost + heuristic(node.state, problem))

    frontier.push(initialNode)  

    while not frontier.isEmpty():  
        currentNode = frontier.pop()  # Remove the node with the highest priority (lowest cost + heuristic)
        if problem.goalTest(currentNode.state):  
            return backtrackToStart(currentNode)  # Backtrack to construct the path taken to reach the goal

        if currentNode.state not in checkedStates:  # Ensure each state is processed only once
            checkedStates.add(currentNode.state)  # Mark the current state as checked
            for action in problem.getActions(currentNode.state): 
                successor = problem.getResult(currentNode.state, action)
                totalCost = currentNode.path_cost + problem.getCost(currentNode.state, action)
                successorNode = Node(successor, currentNode, action, totalCost)
                if successor not in checkedStates:
                    frontier.push(successorNode)  # Add successor to the frontier for further exploration

    raise Exception("Failed to find a solution")  
    # If the loop exits without finding a goal

def backtrackToStart(node):
    """
    Constructs the path from the goal node back to the start node by tracing parent links.
    """
    path = []
    while node.parent is not None:  # Traverse back from goal to start
        path.append(node.action)
        node = node.parent
    return path[::-1]  
    # Return the path in start to goal order
    

# Abbreviations
bfs = breadthFirstSearch
astar = aStarSearch
ids = iterativeDeepeningSearch
