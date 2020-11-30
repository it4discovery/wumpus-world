
from copy import deepcopy
import random
from enum import Enum
#from wumpusWorld.environment.Agent import *
from wumpusWorld.environment.Orientation import East, North, South, West

class Action(Enum):
    Forward = "Forward"
    TurnLeft = "TurnLeft"
    TurnRight = "TurnRight"
    Shoot = "Shoot"
    Grab = "Grab"
    Climb = "Climb"


class Coords:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def adjacentCells(self, gridWidth, gridHeight):
        toLeft = Coords(self._x - 1, self._y ) if (self._x > 0) else False
        toRight =  Coords(self._x + 1, self._y ) if (self._x < gridWidth - 1) else False
        below = Coords(self._x, self._y  - 1) if (self._y  > 0) else False
        above = Coords(self._x, self._y  + 1) if (self._y  < gridHeight - 1) else False

        return [toLeft, toRight, below, above]
    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y


class Percept:
    def __init__(self, stench, breeze, glitter, bump, scream, isTerminated, reward):
        self.stench = stench
        self.breeze = breeze
        self.glitter = glitter
        self.bump = bump
        self.scream = scream
        self.isTerminated = isTerminated
        self.reward = reward

    def show(self):
        return "stench:%s breeze:%s glitter:%s bump:%s scream:%s isTerminated:%s reward:%s" % (self.stench, self.breeze, self.glitter, self.bump, self.scream, self.isTerminated, self.reward)


class Environment:
    def __init__(self,
                 gridWidth,
                 gridHeight,
                 pitProb,
                 allowClimbWithoutGold,
                 agent,
                 pitLocations,
                 isTerminated,
                 wumpusLocation,
                 wumpusAlive,
                 goldLocation):
        self.gridWidth = gridWidth
        self.gridHeight = gridHeight
        self.pitProb = pitProb
        self.allowClimbWithoutGold = allowClimbWithoutGold
        self.agent = agent
        self.pitLocations = pitLocations
        self.isTerminated = isTerminated
        self.wumpusLocation = wumpusLocation
        self.wumpusAlive = wumpusAlive
        self.goldLocation = goldLocation

    def isPitAt(self, coords):
        for item in self.pitLocations:
            if(item.x == coords.x and item.y == coords.y):
                return True

        return False

    def isWumpusAt(self, coords):
        if(self.wumpusLocation is None):
            return False
        elif(self.wumpusLocation.x == coords.x and self.wumpusLocation.y == coords.y):
            return True
        else:
            return False

    def isAgentAt(self, coords):
        if(self.agent.location.x == coords.x and self.agent.location.y == coords.y):
            return True
        else:
            return False

    def isGlitter(self):
        if(self.goldLocation is None):
            return False
        elif(self.agent.location.x == self.goldLocation.x and self.agent.location.y == self.goldLocation.y):
            return True
        else:
            return False           

    def isGoldAt(self, coords):
        if(self.goldLocation is None):
            return False
        elif(coords.x == self.goldLocation.x and coords.y == self.goldLocation.y):
            return True
        else:
            return False         

    def killAttemptSuccessful(self):
        if(self.wumpusLocation is None):
            return True

        wumpusInLineOfFire = False
        
        if self.agent.orientation == West:
            if (self.agent.location.x > self.wumpusLocation.x and self.agent.location.y == self.wumpusLocation.y):
                wumpusInLineOfFire = True
        elif self.agent.orientation == East:
            if (self.agent.location.x < self.wumpusLocation.x and self.agent.location.y == self.wumpusLocation.y):
                wumpusInLineOfFire = True
        elif self.agent.orientation == South:
            if (self.agent.location.x == self.wumpusLocation.x and self.agent.location.y > self.wumpusLocation.y):
                wumpusInLineOfFire = True
        elif self.agent.orientation == North:
            if (self.agent.location.x == self.wumpusLocation.x and self.agent.location.y < self.wumpusLocation.y):
                wumpusInLineOfFire = True
        
        return self.agent.hasArrow and self.wumpusAlive and wumpusInLineOfFire

    def adjacentCells(self, coords):
        toLeft = Coords(coords.x - 1, coords.y) if (coords.x > 0) else False
        toRight =  Coords(coords.x + 1, coords.y) if (coords.x < self.gridWidth - 1) else False
        below = Coords(coords.x, coords.y - 1) if (coords.y > 0) else False
        above = Coords(coords.x, coords.y + 1) if (coords.y < self.gridHeight - 1) else False

        return [toLeft, toRight, below, above]

    def isPitAdjacent(self, coords):
        adjacent = self.adjacentCells(coords)
        for item in adjacent:
            if(item != False):
                if(self.isPitAt(item)):
                    return True
                    
        return False

    def isWumpusAdjacent(self, coords):
        adjacent = self.adjacentCells(coords)

        for item in adjacent:
            if(item != False):
                if(self.isWumpusAt(item)):
                    return True

        return False

    def isBreeze(self):
        return self.isPitAdjacent(self.agent.location)

    def isStench(self):
        return self.isWumpusAdjacent(self.agent.location) or self.isWumpusAt(self.agent.location)

    def applyAction(self, action):
        if (self.isTerminated):
            return self, Percept(False, False, False, False, False, True, 0)
        else:
            if action.name is Action.Forward.name:                
                movedAgent = self.agent.forward(
                    self.gridWidth, self.gridHeight)

                death = (self.isWumpusAt(movedAgent.location)
                         and self.wumpusAlive) or self.isPitAt(movedAgent.location)
                
                newAgent = deepcopy(movedAgent)
                newAgent.isAlive = False if death else True

                Bump = False
                if(newAgent.location.x == self.agent.location.x and newAgent.location.y == self.agent.location.y):
                    Bump = True
                
                newEnv = Environment(self.gridWidth, self.gridHeight, self.pitProb, self.allowClimbWithoutGold, newAgent, self.pitLocations,
                                     death, self.wumpusLocation, self.wumpusAlive, newAgent.location if self.agent.hasGold else self.goldLocation)
                return newEnv, Percept(newEnv.isStench(), newEnv.isBreeze(), newEnv.isGlitter(), Bump, False, False if newAgent.isAlive else True,  -1 if newAgent.isAlive else -1001)
            elif action.name is Action.TurnLeft.name:
                return Environment(self.gridWidth, self.gridHeight, self.pitProb, self.allowClimbWithoutGold, self.agent.turnLeft(), self.pitLocations, self.isTerminated,  self.wumpusLocation,  self.wumpusAlive,  self.goldLocation), Percept(self.isStench(), self.isBreeze(), self.isGlitter(), False, False,  False, -1)
            elif action.name is Action.TurnRight.name:
                return Environment(self.gridWidth, self.gridHeight, self.pitProb, self.allowClimbWithoutGold, self.agent.turnRight(), self.pitLocations, self.isTerminated, self.wumpusLocation, self.wumpusAlive, self.goldLocation), Percept(self.isStench(), self.isBreeze(), self.isGlitter(), False, False,  False, -1)
            elif action.name is Action.Grab.name:
                newAgent = deepcopy(self.agent)
                newAgent.hasGold = self.isGlitter()

                return Environment(self.gridWidth, self.gridHeight, self.pitProb, self.allowClimbWithoutGold, newAgent, self. pitLocations, self.isTerminated, self.wumpusLocation, self.wumpusAlive, self.agent.location if newAgent.hasGold else self.goldLocation), Percept(self.isStench(), self.isBreeze(), self.isGlitter(), False, False,  False, -1)
            elif action.name is Action.Climb.name:
                inStartLocation = False
                if(self.agent.location.x == Coords(0, 0).x and self.agent.location.y == Coords(0, 0).y):
                    inStartLocation = True

                success = self.agent.hasGold and inStartLocation
                isTerminated = success or self.allowClimbWithoutGold

                return Environment(self.gridWidth, self.gridHeight, self.pitProb, self.allowClimbWithoutGold, self.agent, self.pitLocations, isTerminated, self.wumpusLocation, self.wumpusAlive, self.goldLocation), Percept(False, False, self.agent.hasGold, False, False, isTerminated, 999 if success else -1)
            elif action.name is Action.Shoot.name:
                hadArrow = self.agent.hasArrow
                wumpusKilled = self.killAttemptSuccessful()
                newAgent = deepcopy(self.agent)
                newAgent.hasArrow = False
                return Environment(self.gridWidth, self.gridHeight, self.pitProb, self.allowClimbWithoutGold, newAgent, self.pitLocations, self.isTerminated, self.wumpusLocation, self.wumpusAlive and not wumpusKilled, self.goldLocation), Percept(self.isStench(), self.isBreeze(), self.isGlitter(), False, wumpusKilled, False, -11 if hadArrow else -1)

    def visualize(self):
        wumpusSymbol =  "W" if (self.wumpusAlive == True) else "w"         

        Rows = []
        for y in range(self.gridHeight):
            Cells = []
            for x in range(self.gridWidth):
                A = "A" if (self.isAgentAt(Coords(x, y))) else " "
                P = "P" if (self.isPitAt(Coords(x, y))) else " "
                G = "G" if (self.isGoldAt(Coords(x, y))) else " "
                W = wumpusSymbol if (self.isWumpusAt(Coords(x, y))) else " "
                
                Cells.append("%s%s%s%s" % (A, P, G, W))

            Rows.append('|'.join(Cells))

        return '\n'.join(Rows)

class WumpusWorldEnvironment:
    def apply(self, gridWidth, gridHeight, pitProb, allowClimbWithoutGold):
        self.gridWidth = gridWidth
        self.gridHeight = gridHeight
        self.pitProb = pitProb
        self.allowClimbWithoutGold = allowClimbWithoutGold

        cellIndexes = []
        pitLocations = []

        for x in range(gridWidth):
            for y in range(gridHeight):
                cellIndexes.append(Coords(x, y))

        cellIndexes.pop(0)
        pitCount = 0
        for item in cellIndexes:
            if random.uniform(0, 1) < pitProb:
                if(pitCount == 3):
                    break
                pitLocations.append(item)
                pitCount = pitCount + 1

        env = Environment(
            gridWidth,
            gridHeight,
            pitProb,
            allowClimbWithoutGold,
            Agent(),
            pitLocations,
            False,
            self.randomLocationExceptOrigin(),
            True,
            self.randomLocationExceptOrigin()
        )

        return env, Percept(env.isStench(), env.isBreeze(), False, False, False, False,  0.0)

    def randomLocationExceptOrigin(self):
        x = random.randrange(self.gridWidth)
        y = random.randrange(self.gridHeight)

        if x == 0 and y == 0:
            self.randomLocationExceptOrigin()
        else:
            return Coords(x, y)
