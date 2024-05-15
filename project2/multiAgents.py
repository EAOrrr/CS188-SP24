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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
        """
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
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        if newFood.count() == 0:
            return 10000
        # minimum manhattan distance from newPos to nearest food
        # the smaller foodDist is, the better
        foodDist = min(d for d in [manhattanDistance(newPos, food) for food in newFood.asList()]) 
        foodDistScore = - 10 * foodDist
        # minimum manhattan distance from newPos to nearest ghost
        # the largest, the better, only need to worry when ghost is near
        ghostPOS = [ghost.getPosition() for ghost in newGhostStates]
        ghostDist = min(d for d in [manhattanDistance(newPos, ghostPos) for ghostPos in ghostPOS] + [9])
        # danger
        if ghostDist < 4:
            ghostScore = -10000
        else:
            ghostScore = 0

        # win score if eat food
        if currentGameState.getNumFood() - newFood.count():
            foodRemainScore = 1000
        else:
            foodRemainScore = 0
        
        # randomScore to break ties
        randomScore = random.randint(-2, 3)
        # print(foodDistScore, ghostScore, foodRemainScore, randomScore)
        return  foodDistScore + ghostScore + foodRemainScore + randomScore

def scoreEvaluationFunction(currentGameState: GameState):
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
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

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
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def max_value(gamestate: GameState, agentIndex: int, depth: int):
            # agentIndex default = 0
            possibleAction = gamestate.getLegalActions()

            # depth = 0 or already lost: no need to further explore step
            if len(possibleAction) == 0 or depth == 0 or gamestate.isLose():
                return None, self.evaluationFunction(gamestate)
            # movement exist
            bestAction = possibleAction[0]

            # if already win, return action and score
            if gamestate.isWin():
                return bestAction, self.evaluationFunction(gamestate)

            # else: find the best action using minimax algorithm
            value =  float('-inf')
            for action in possibleAction:
                successor = gamestate.generateSuccessor(agentIndex, action)
                _, successor_value = min_value(successor, agentIndex + 1, depth)
                if value < successor_value:
                    bestAction = action
                    value = successor_value

            return bestAction, value
        
        def min_value(gamestate: GameState, agentIndex: int, depth: int):
            possibleAction = gamestate.getLegalActions(agentIndex)
            # movement does not exist or already lost or depth = 0 or 
            # already win(since we don't care the ghost actual movement)
            # action = None is OK 
            if not possibleAction or depth == 0 or gamestate.isLose() or gamestate.isWin():
                return None, self.evaluationFunction(gamestate)
            # movement exist: find the best action using minimax algorithm
            value, bestAction = float('inf'), None
            for action in gamestate.getLegalActions(agentIndex):
                successor = gamestate.generateSuccessor(agentIndex, action)
                if agentIndex == gamestate.getNumAgents() - 1:
                    # the last ghost agent, next agent will be pacman agent.
                    # increase depth
                    _, successor_value = max_value(successor, 0, depth - 1)
                    if value > successor_value:
                        bestAction = action
                        value = successor_value
                else:
                    # next agent is ghost agent, whose aim is to minimize gamestate's score
                    _, successor_value = min_value(successor, agentIndex + 1, depth)
                    if value > successor_value:
                        bestAction = action
                        value = successor_value
            return bestAction, value
        # return minimax(gameState, self.depth)
        action, _ = max_value(gameState, 0, self.depth)
        return action
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #############################################################
        # alpha: max-value's best solution on path to root          #
        # beta : min-value's best solution on path to root          #
        #############################################################
         
        def max_value(state:GameState, agentIndex, alpha, beta, depth):
            possibleAction = state.getLegalActions(agentIndex)
            if not possibleAction or state.isLose() or depth == 0:
                return None, self.evaluationFunction(state)
                
            value, bestAction = float('-inf'), possibleAction[0]
            for action in possibleAction:
                successor = state.generateSuccessor(agentIndex, action)
                _, successorValue = min_value(successor, agentIndex + 1, alpha, beta, depth)
                if value < successorValue:
                    value, bestAction = successorValue, action
                # prune:
                # value will only inc during max_value process
                # value > beta means it has been already larger than the optimal solution
                # no need to further explore
                if value > beta:
                    return bestAction, value
                # update value
                alpha = max(alpha, value)
            return bestAction, value
            

        def min_value(state:GameState, agentIndex, alpha, beta, depth):
            possibleAction = state.getLegalActions(agentIndex)
            if not possibleAction or state.isLose() or depth == 0:
                return None, self.evaluationFunction(state)
            value, bestAction = float('inf'), possibleAction[0]
            for action in possibleAction:
                successor = state.generateSuccessor(agentIndex, action)
                if agentIndex == state.getNumAgents() - 1:
                    _, successorValue = max_value(successor, 0, alpha, beta, depth - 1)
                    if successorValue < value:
                        value, bestAction = successorValue, action
                else:
                    _, successorValue = min_value(successor, agentIndex + 1, alpha, beta, depth)
                    if successorValue < value:
                        value, bestAction = successorValue, action
                # prune
                # value will only dec during min_value process
                # value < alpha means it has been already smaller than the optimal solution
                # no need to further explore
                if value < alpha:
                    return bestAction, value
                # update beta
                beta = min(value, beta)
            return bestAction, value
        action, _ = max_value(gameState, 0, float('-inf'), float('inf'), self.depth)
        return action
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def max_value(state:GameState, agentIndex, depth):
            possibleAction = state.getLegalActions()
            if not possibleAction or state.isLose() or depth == 0:
                return None, self.evaluationFunction(state)
            value, bestAction = float('-inf'), possibleAction[0]
            for action in possibleAction:
                successor = state.generateSuccessor(agentIndex, action)
                successorValue = exp_value(successor, agentIndex + 1, depth)
                if successorValue > value:
                    value, bestAction = successorValue, action
            return bestAction, value

        def exp_value(state:GameState, agentIndex, depth):
            possibleAction = state.getLegalActions(agentIndex)
            if not possibleAction or state.isLose() or depth == 0 or state.isWin():
                return self.evaluationFunction(state)
            value = 0
            for action in possibleAction:
                successor = state.generateSuccessor(agentIndex, action)
                if agentIndex + 1 == state.getNumAgents():
                    _, successorValue = max_value(successor, 0, depth - 1)
                else:
                    successorValue = exp_value(successor, agentIndex + 1, depth)
                value += successorValue
            return value / len(possibleAction)
        action, _ = max_value(gameState, 0, self.depth) 
        return action
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    pacmanPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    ghostPos = [ghostState.getPosition() for ghostState in ghostStates]

    if food.count() == 0:
        return 100000

    # foodRemainScore
    # the less food the state has, the better the state is 
    foodRemainScore = - 100 * currentGameState.getNumFood()

    # distScore
    # foodDistScore
    # the more the food is near to pacman, the better the state is
    # print(pacmanPos, food)
    minFoodDist = min(manhattanDistance(pacmanPos, f) for f in food.asList())
    minFoodDistScore = - 10 * minFoodDist

    # ghost score
    # the more the ghost is near to pacman, the worse the state is
    ghost = sum([manhattanDistance(pacmanPos, ghost) for ghost in ghostPos] + [9])
    if ghost < 4 * len(ghostStates):
        ghostScore = -10000
    else:
        ghostScore = 0

    # ghost scared score
    # the more the ghost is scared, the better the state is
    ghostScaredScore = sum(scaredTimes)
    

    return foodRemainScore + minFoodDistScore + ghostScore + ghostScaredScore + currentGameState.getScore()
    
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
