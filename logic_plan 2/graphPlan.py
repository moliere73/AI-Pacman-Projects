# graphPlan.py
# ------------
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


"""
In graphPlan.py, you will implement graph plan planning methods which are called by
Pacman agents (in graphPlanAgents.py).
"""

import util
import sys
import logic
import game
from graphplanUtils import *

OPEN = "Open"
FOOD = "Food"
PACMAN = "Pacman"

class PlanningProblem:
    """
    This class outlines the structure of a planning problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the planning problem.
        """
        util.raiseNotDefined()

    def getGhostStartStates(self):
        """
        Returns a list containing the start state for each ghost.
        Only used in problems that use ghosts (FoodGhostPlanningProblem)
        """
        util.raiseNotDefined()
        
    def getGoalState(self):
        """
        Returns goal state for problem. Note only defined for problems that have
        a unique goal state such as PositionPlanningProblem
        """
        util.raiseNotDefined()

def tinyMazePlan(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def modelToString(model):
    """Converts the model to a string for printing purposes. The keys of a model are 
    sorted before converting the model to a string.
    
    model: Either a boolean False or a dictionary of Expr symbols (keys) 
    and a corresponding assignment of True or False (values). This model is the output of 
    a call to logic.pycoSAT.
    """
    if model == False:
        return "False" 
    else:
        # Dictionary
        modelList = sorted(model.items(), key=lambda item: str(item[0]))
        return str(modelList)

"""
    Create and solve the pacman navigation problem. 
    You will create instances, variables, and operators for pacman's actions 
    'North','South','East','West'
    Operators contain lists of preconditions, add effects, and delete effects
    which are all composed of propositions (boolean descriptors of the environment)
    Operators will test the current state propositions to determine whether all
    the preconditions are true, and then add and delete state propositions 
    to update the state.

    o_west = Operator('West', # the name of the action
                      [],     # preconditions
                      [],     # add effects   
                      []      # delete effects                         
                      )  
                      
    A GraphPlan problem requires a list of all instances, all operators, the start state and 
    the goal state. You must create these lists for GraphPlan.solve.
"""
def positionGraphPlan(problem):
    width, height = problem.getWidth(), problem.getHeight()
    walls = problem.walls     # if walls[x][y] is True, then that means there is a wall at (x,y)

    start_x = problem.startState[0]
    start_y = problem.startState[1]

    goal_x = problem.goal[0]
    goal_y = problem.goal[1]

    """
    Create your variables with unique string names
    Each variable has a type
    vname = Variable('name',TYPE)
    TYPES = INT, PACMAN, OPEN, FOOD
    """
    # Instances
    allinstances = [] # Make sure you fill this with ALL the instances you define

    "*** YOUR CODE HERE ***"

    # Variables
    "*** YOUR CODE HERE ***"

    start_state = [] # Make sure you fill this with ALL the starting propositions

    goal_state = [] # Make sure you fill this with ALL the propositions required for the goal

    alloperators = [] # Make sure you fill this with ALL the operators you define

    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

    prob1 = GraphPlanProblem('goto_xy',allinstances,alloperators,start_state, goal_state)
    prob1.solve()
    
    # some functions for debugging 
    #prob1.dump()
    #prob1.display()
    
    actions = prob1.getactions()
    "*** YOUR CODE HERE ***"
    return []
    

"""
Now use the operators for moving along with an eat operator you must create to eat 
all the food in the maze.
"""
def foodGraphPlan(problem):
    width, height = problem.getWidth(), problem.getHeight()
    walls = problem.walls     # if walls[x][y] is True, then that means there is a wall at (x,y)

    start_x = problem.start[0][0]
    start_y = problem.start[0][1]

    foodlist = problem.start[1].asList() 
    
    """The same as the previous question:
    Operators contain lists of preconditions, add effects, and delete effects
    which are all composed of propositions (boolean descriptors of the environment)
    Operators will test the current state propositions to determine whether all
    the preconditions are true, and then add and delete state propositions 
    to update the state.

    TYPES = INT, PACMAN, OPEN, FOOD
    """
    # Instances
    allinstances = [] # Make sure you fill this with ALL the instances you define

    "*** YOUR CODE HERE ***"

    # Variables
    "*** YOUR CODE HERE ***"

    start_state = [] # Make sure you fill this with ALL the starting propositions

    goal_state = [] # Make sure you fill this with ALL the propositions required for the goal

    alloperators = [] # Make sure you fill this with ALL the operators you define


    # Operators 

    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

    prob1 = GraphPlanProblem('eatfood',allinstances,
                             alloperators, start_state, goal_state)
    
    prob1.solve()
    actions = prob1.getactions()
    # functions to help with debugging
    #prob1.dump()
    #prob1.display()
    "*** YOUR CODE HERE ***"

    return []



# Abbreviations
pgp = positionGraphPlan
fgp = foodGraphPlan

# Sometimes the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(100000)
    
