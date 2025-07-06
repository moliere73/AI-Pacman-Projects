# inference.py
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


import itertools
import random
import busters
import game

from util import manhattanDistance, raiseNotDefined


class DiscreteDistribution(dict):
    """
    A DiscreteDistribution models belief distributions and weight distributions
    over a finite set of discrete keys.
    """
    def __getitem__(self, key):
        self.setdefault(key, 0)
        return dict.__getitem__(self, key)

    def copy(self):
        """
        Return a copy of the distribution.
        """
        return DiscreteDistribution(dict.copy(self))

    def argMax(self):
        """
        Return the key with the highest value.
        """
        if len(self.keys()) == 0:
            return None
        all = list(self.items())
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def total(self):
        """
        Return the sum of values for all keys.
        """
        return float(sum(self.values()))

    def normalize(self):
        """
        Normalize the distribution such that the total value of all keys sums
        to 1. The ratio of values for all keys will remain the same. In the case
        where the total value of the distribution is 0, do nothing.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> dist.normalize()
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0)]
        >>> dist['e'] = 4
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0), ('e', 4)]
        >>> empty = DiscreteDistribution()
        >>> empty.normalize()
        >>> empty
        {}
        """
        "*** YOUR CODE HERE ***"
        if not self:  # This checks if the dictionary is empty
            return
        total = sum(self.values())
        if total == 0:  # Do nothing if the total is zero
            return
        for key, value in self.items():
            self[key] = value / total


    def sample(self):
        """
        Draw a random sample from the distribution and return the key, weighted
        by the values associated with each key.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> N = 1000000.0
        >>> samples = [dist.sample() for _ in range(int(N))]
        >>> round(samples.count('a') * 1.0/N, 1)  # proportion of 'a'
        0.2
        >>> round(samples.count('b') * 1.0/N, 1)
        0.4
        >>> round(samples.count('c') * 1.0/N, 1)
        0.4
        >>> round(samples.count('d') * 1.0/N, 1)
        0.0
        """
        "*** YOUR CODE HERE ***"
        keys = list(self.keys())
        values = list(self.values())
        return random.choices(keys, weights=values, k=1)[0]


class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    """
    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, ghostAgent):
        """
        Set the ghost agent for later access.
        """
        self.ghostAgent = ghostAgent
        self.index = ghostAgent.index
        self.obs = []  # most recent observation position

    def getJailPosition(self):
        return (2 * self.ghostAgent.index - 1, 1)

    def getPositionDistributionHelper(self, gameState, pos, index, agent):
        try:
            jail = self.getJailPosition()
            gameState = self.setGhostPosition(gameState, pos, index + 1)
        except TypeError:
            jail = self.getJailPosition(index)
            gameState = self.setGhostPositions(gameState, pos)
        pacmanPosition = gameState.getPacmanPosition()
        ghostPosition = gameState.getGhostPosition(index + 1)  # The position you set
        dist = DiscreteDistribution()
        if pacmanPosition == ghostPosition:  # The ghost has been caught!
            dist[jail] = 1.0
            return dist
        pacmanSuccessorStates = game.Actions.getLegalNeighbors(pacmanPosition, \
                gameState.getWalls())  # Positions Pacman can move to
        if ghostPosition in pacmanSuccessorStates:  # Ghost could get caught
            mult = 1.0 / float(len(pacmanSuccessorStates))
            dist[jail] = mult
        else:
            mult = 0.0
        actionDist = agent.getDistribution(gameState)
        for action, prob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            if successorPosition in pacmanSuccessorStates:  # Ghost could get caught
                denom = float(len(actionDist))
                dist[jail] += prob * (1.0 / denom) * (1.0 - mult)
                dist[successorPosition] = prob * ((denom - 1.0) / denom) * (1.0 - mult)
            else:
                dist[successorPosition] = prob * (1.0 - mult)
        return dist

    def getPositionDistribution(self, gameState, pos, index=None, agent=None):
        """
        Return a distribution over successor positions of the ghost from the
        given gameState. You must first place the ghost in the gameState, using
        setGhostPosition below.
        """
        if index == None:
            index = self.index - 1
        if agent == None:
            agent = self.ghostAgent
        return self.getPositionDistributionHelper(gameState, pos, index, agent)

    def getObservationProb(self, noisyDistance, pacmanPosition, ghostPosition, jailPosition):
        """
        Return the probability P(noisyDistance | pacmanPosition, ghostPosition).
        """
        "*** YOUR CODE HERE ***"
        # Check if the ghost is in jail
        if ghostPosition == jailPosition:
            # If the ghost is in jail, the sensor should deterministically read None
            if noisyDistance is None:
                return 1.0
            else:
                return 0.0
        else:
            # If the ghost is not in jail, the sensor reading should not be None
            if noisyDistance is None:
                return 0.0
            # Compute the true distance from Pacman to the ghost
            trueDistance = manhattanDistance(pacmanPosition, ghostPosition)
            # Get the observation probability of the noisy distance given the true distance
            return busters.getObservationProbability(noisyDistance, trueDistance)

    def setGhostPosition(self, gameState, ghostPosition, index):
        """
        Set the position of the ghost for this inference module to the specified
        position in the supplied gameState.

        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observe.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[index] = game.AgentState(conf, False)
        return gameState

    def setGhostPositions(self, gameState, ghostPositions):
        """
        Sets the position of all ghosts to the values in ghostPositions.
        """
        for index, pos in enumerate(ghostPositions):
            conf = game.Configuration(pos, game.Directions.STOP)
            gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
        return gameState

    def observe(self, gameState):
        """
        Collect the relevant noisy distance observation and pass it along.
        """
        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.index:  # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.update(obs, gameState)

    def initialize(self, gameState):
        """
        Initialize beliefs to a uniform distribution over all legal positions.
        """
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.allPositions = self.legalPositions + [self.getJailPosition()]
        self.initializeUniformly(gameState)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        """
        Set the belief state to a uniform prior belief over all positions.
        """
        raise NotImplementedError

    def update(self, observation, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        raise NotImplementedError

    def predict(self, gameState):
        """
        Predict beliefs for the next time step from a gameState.
        """
        raise NotImplementedError

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        raise NotImplementedError


class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward algorithm updates to
    compute the exact belief function at each time step.
    """
    def initializeUniformly(self, gameState):
        """
        Begin with a uniform distribution over legal ghost positions (i.e., not
        including the jail position).
        """
        self.beliefs = DiscreteDistribution()
        for p in self.legalPositions:
            self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def update(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        self.allPositions is a list of the possible ghost positions, including
        the jail position. You should only consider positions that are in
        self.allPositions.

        The update model is not entirely stationary: it may depend on Pacman's
        current position. However, this is not a problem, as Pacman's current
        position is known.
        """
        "*** YOUR CODE HERE ***"
        pacmanPosition = gameState.getPacmanPosition()  # Get this once as it does not change
        jailPosition = self.getJailPosition()  # Same for the jail position
        # Temporary distribution to accumulate new probabilities
        newBeliefs = DiscreteDistribution()
        for ghostPosition in self.allPositions:
            # Calculate the likelihood of the observation given the ghost's position
            prob = self.getObservationProb(observation, pacmanPosition, ghostPosition, jailPosition)
            
            # Update the belief for the ghostPosition
            newBeliefs[ghostPosition] = self.beliefs[ghostPosition] * prob
        # Normalize the updated beliefs to ensure they sum to 1
        newBeliefs.normalize()
        # Replace old beliefs with the updated and normalized ones
        self.beliefs = newBeliefs
        

    def predict(self, gameState):
        """
        Predict beliefs in response to a time step passing from the current
        state.

        The transition model is not entirely stationary: it may depend on
        Pacman's current position. However, this is not a problem, as Pacman's
        current position is known.
        """
        "*** YOUR CODE HERE ***"
        newBeliefs = DiscreteDistribution()
        for oldPos in self.allPositions:
            # Get the distribution over new positions for the ghost given its previous position
            newPosDist = self.getPositionDistribution(gameState, oldPos)
            # Update the new belief distribution based on the probability of moving to each new position
            for newPos, prob in newPosDist.items():
                if newPos in self.allPositions:  # Ensure newPos is a valid position
                    newBeliefs[newPos] += self.beliefs[oldPos] * prob
        # Normalize the new beliefs to ensure they sum to 1
        newBeliefs.normalize()
        # Update the beliefs with the normalized new beliefs
        self.beliefs = newBeliefs

    def getBeliefDistribution(self):
        return self.beliefs


class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.
    """
    def __init__(self, ghostAgent, numParticles=300):
        InferenceModule.__init__(self, ghostAgent)
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initializeUniformly(self, gameState):
        """
        Initialize a list of particles. Use self.numParticles for the number of
        particles. Use self.legalPositions for the legal board positions where
        a particle could be located. Particles should be evenly (not randomly)
        distributed across positions in order to ensure a uniform prior. Use
        self.particles for the list of particles.
        """
        #self.particles = []
        "*** YOUR CODE HERE ***"
        self.particles = []
        # Repeat each position in the legal positions list to approximately
        # achieve an even distribution based on the number of particles.
        num_legal_positions = len(self.legalPositions)
        particles_per_position = self.numParticles // num_legal_positions
        extra_particles = self.numParticles % num_legal_positions

        # Start by adding particles_per_position particles for each legal position
        for position in self.legalPositions:
            self.particles += [position] * particles_per_position

        # Distribute extra particles
        extra_positions = random.sample(self.legalPositions, extra_particles)
        for position in extra_positions:
            self.particles.append(position)

        # Shuffle to ensure randomness of initial particle order (optional)
        random.shuffle(self.particles)

    def update(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the estimated Manhattan distance to the ghost you are
        tracking.
        """
        "*** YOUR CODE HERE ***"
        pacmanPosition = gameState.getPacmanPosition()
        jailPosition = self.getJailPosition()
        weights = DiscreteDistribution()
        # Step 1: Calculate weights for each particle
        for particle in self.particles:
            prob = self.getObservationProb(observation, pacmanPosition, particle, jailPosition)
            weights[particle] += prob  # Increment weight for this particle position
        # Step 2: Check if all weights are zero
        if weights.total() == 0:
            # Reinitialize if all particles have zero probability
            self.initializeUniformly(gameState)
        else:
            # Normalize the weights
            weights.normalize()
            # Step 3: Resample particles based on the computed weights
            new_particles = []
            for _ in range(self.numParticles):
                new_particles.append(weights.sample())
            # Replace the old particles with the new resampled particles
            self.particles = new_particles

    def predict(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        "*** YOUR CODE HERE ***"
        new_particles = []
        # For each particle, get the distribution over new positions
        for oldPos in self.particles:
            newPosDist = self.getPositionDistribution(gameState, oldPos)
            # Sample a new position from the distribution and append to new_particles
            new_particles.append(newPosDist.sample())

        # Update the particles with the newly predicted positions
        self.particles = new_particles

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution.
        """
        "*** YOUR CODE HERE ***"
        belief_distribution = DiscreteDistribution()
        # Count occurrences of each position in particles
        for particle in self.particles:
            belief_distribution[particle] += 1
        # Normalize to convert counts to probabilities
        belief_distribution.normalize()
        return belief_distribution


class JointParticleFilter(ParticleFilter):
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """
    def __init__(self, numParticles=600):
        self.setNumParticles(numParticles)

    def initialize(self, gameState, legalPositions):
        """
        Store information about the game, then initialize particles.
        """
        self.numGhosts = gameState.getNumAgents() - 1
        self.ghostAgents = []
        self.legalPositions = legalPositions
        self.initializeUniformly(gameState)

    def initializeUniformly(self, gameState):
        """
        Initialize particles to be consistent with a uniform prior. Particles
        should be evenly distributed across positions in order to ensure a
        uniform prior.
        """
        #self.particles = []
        "*** YOUR CODE HERE ***"
        self.particles = []
        # Step 1: Create all possible combinations of ghost positions
        all_possible_combinations = list(itertools.product(self.legalPositions, repeat=self.numGhosts))
        # Step 2: Shuffle the permutations to ensure random order
        random.shuffle(all_possible_combinations)
        # Step 3: Evenly distribute the particles among all combinations
        num_combinations = len(all_possible_combinations)
        for i in range(self.numParticles):
            self.particles.append(all_possible_combinations[i % num_combinations])

    def addGhostAgent(self, agent):
        """
        Each ghost agent is registered separately and stored (in case they are
        different).
        """
        self.ghostAgents.append(agent)

    def getJailPosition(self, i):
        return (2 * i + 1, 1)

    def observe(self, gameState):
        """
        Resample the set of particles using the likelihood of the noisy
        observations.
        """
        observation = gameState.getNoisyGhostDistances()
        self.update(observation, gameState)

    def update(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the estimated Manhattan distances to all ghosts you
        are tracking.
        """
        "*** YOUR CODE HERE ***"
        pacmanPosition = gameState.getPacmanPosition()
        weights = DiscreteDistribution()
        
        for particle in self.particles:
            weight = 1.0
            # Update the weight for each ghost based on observation
            for i in range(self.numGhosts):
                # If a ghost is in jail, it has a fixed position, and the observation should always be None.
                jailPosition = self.getJailPosition(i)
                ghostPosition = particle[i]
                if ghostPosition == jailPosition:
                    # If the ghost is in jail, the observation must be None.
                    if observation[i] is not None:
                        weight = 0
                    continue

                # Update the weight based on how likely the observation is given the ghost's position.
                weight *= self.getObservationProb(observation[i], pacmanPosition, ghostPosition, jailPosition)
            weights[particle] += weight

        # Check for the case where all particles have zero weight.
        if weights.total() == 0:
            self.initializeUniformly(gameState)
        else:
            # Normalize and resample
            weights.normalize()
            new_particles = []
            for _ in range(self.numParticles):
                new_particles.append(weights.sample())
            self.particles = new_particles
        

    def predict(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        newParticles = []
        for oldParticle in self.particles:
            newParticle = list(oldParticle)  # A list of ghost positions

            # now loop through and update each entry in newParticle...
            "*** YOUR CODE HERE ***"
            for i in range(self.numGhosts):
                # Get the distribution over new positions for the i-th ghost,
                # given the old particle's positions of all ghosts
                newPosDist = self.getPositionDistribution(gameState, newParticle, i, self.ghostAgents[i])

                # Sample a new position for the i-th ghost
                newParticle[i] = newPosDist.sample()

            """*** END YOUR CODE HERE ***"""
            newParticles.append(tuple(newParticle))
        self.particles = newParticles


# One JointInference module is shared globally across instances of MarginalInference
jointInference = JointParticleFilter()


class MarginalInference(InferenceModule):
    """
    A wrapper around the JointInference module that returns marginal beliefs
    about ghosts.
    """
    def initializeUniformly(self, gameState):
        """
        Set the belief state to an initial, prior value.
        """
        if self.index == 1:
            jointInference.initialize(gameState, self.legalPositions)
        jointInference.addGhostAgent(self.ghostAgent)

    def observe(self, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        if self.index == 1:
            jointInference.observe(gameState)

    def predict(self, gameState):
        """
        Predict beliefs for a time step elapsing from a gameState.
        """
        if self.index == 1:
            jointInference.predict(gameState)

    def getBeliefDistribution(self):
        """
        Return the marginal belief over a particular ghost by summing out the
        others.
        """
        jointDistribution = jointInference.getBeliefDistribution()
        dist = DiscreteDistribution()
        for t, prob in jointDistribution.items():
            dist[t[self.index - 1]] += prob
        return dist
