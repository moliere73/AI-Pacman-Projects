# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
def findNearestFood(pacmanPosition, foodGrid):
    nearestFoodDistance = float('inf')  # Initialize with infinity
    nearestFoodPosition = None  # Initialize with None
    for x in range(foodGrid.width):
        for y in range(foodGrid.height):
            if foodGrid[x][y]:  # If there's food at (x, y)
                distance = util.manhattanDistance(pacmanPosition, (x, y))
                if distance < nearestFoodDistance:
                    nearestFoodDistance = distance
                    nearestFoodPosition = (x, y)
    return nearestFoodPosition, nearestFoodDistance


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        currentGameState: The current search state

        action: Direction; The action taken

        returns: float; a heuristic for the given (state,action) pair

        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Extract basic information from the current game state
        
        currentPosition = currentGameState.getPacmanPosition()
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPosition = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        ghostPositions = [ghost.getPosition() for ghost in newGhostStates]

        # Calculate the score of the successor state
        score = successorGameState.getScore()

        # Bonus for eating food
        nearestFoodPosition, nearestFoodDistance = findNearestFood(currentPosition, newFood)
        if nearestFoodPosition and util.manhattanDistance(newPosition, nearestFoodPosition) < nearestFoodDistance:
            score += 20  # Encourage moving towards the nearest food

        for ghostPosition, scaredTime in zip(ghostPositions, newScaredTimes):
            ghostDistance = util.manhattanDistance(newPosition, ghostPosition)
            if scaredTime > 0 and ghostDistance < scaredTime:
                score += 200 / ghostDistance  # Encourage chasing scared ghosts
            elif ghostDistance <= 1:
                score -= 500  # Penalize getting too close to a non-scared ghost

        return score


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent & AlphaBetaPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 7)
    """

    def getAction(self, gameState):
        """
        gameState: the current state

        returns: Direction; the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        def minimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState), None

            if agentIndex == 0:
                return maxNode(agentIndex, depth, gameState)
            else:
                return minNode(agentIndex, depth, gameState)

        def maxNode(agentIndex, depth, gameState):
            maxEval = float("-inf"), None
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                eval, _ = minimax(1, depth, successor)
                if eval > maxEval[0]:
                    maxEval = eval, action
            return maxEval

        def minNode(agentIndex, depth, gameState):
            minEval = float("inf"), None
            nextAgent = agentIndex + 1
            if nextAgent == gameState.getNumAgents():
                nextAgent = 0  
                depth += 1 
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                eval, _ = minimax(nextAgent, depth, successor)
                if eval < minEval[0]:
                    minEval = eval, action
            return minEval

        _, action = minimax(0, 0, gameState)
        return action
        #util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 8)
    """

    def getAction(self, gameState):
        """
        gameState: the current state

        returns: Direction; the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState), None
            
            if agentIndex == 0:
                return maxNode(agentIndex, depth, gameState)
            else:
                return expectNode(agentIndex, depth, gameState)

        def maxNode(agentIndex, depth, gameState):
            maxEval = float("-inf"), None
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                eval, _ = expectimax(1, depth, successor)
                if eval > maxEval[0]:
                    maxEval = eval, action
            return maxEval

        def expectNode(agentIndex, depth, gameState):
            nextAgent = agentIndex + 1
            if nextAgent == gameState.getNumAgents():
                nextAgent = 0  
                depth += 1  
            
            totalEval = 0
            actions = gameState.getLegalActions(agentIndex)
            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                eval, _ = expectimax(nextAgent, depth, successor)
                totalEval += eval
            
            averageEval = totalEval / len(actions) if actions else 0
            return averageEval, None

        _, action = expectimax(0, 0, gameState)
        return action

def betterEvaluationFunction(currentGameState):
    """
    currentGameState: the current state

    returns: float; the evaluation of the state

    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 9).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return float("inf")  # Maximize the utility for winning states
    if currentGameState.isLose():
        return -float("inf")  # Minimize the utility for losing states

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    capsules = currentGameState.getCapsules()

    # Feature 1: Score from the game state
    score = currentGameState.getScore()

    # Feature 2: Distance to the closest food
    foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
    if foodDistances:
        closestFoodDistance = min(foodDistances)
        score += 1.0 / closestFoodDistance

    # Feature 3: Ghost proximity and scared status
    ghostDistanceSum = 0
    for ghostState in newGhostStates:
        dist = manhattanDistance(newPos, ghostState.getPosition())
        if ghostState.scaredTimer:
            score += 10.0 / dist
        else:
            # Penalize positions close to active ghosts
            ghostDistanceSum += dist
    if ghostDistanceSum > 0:
        score -= 1.0 / ghostDistanceSum

    # Feature 4: Number of capsules left
    score -= 3 * len(capsules)

    # Feature 5: Number of food left
    score -= 4 * len(newFood.asList())

    return score

# Abbreviation
better = betterEvaluationFunction

