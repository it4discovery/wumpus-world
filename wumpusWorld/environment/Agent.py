from copy import deepcopy
from wumpusWorld.environment.Environment import Coords, Action
from wumpusWorld.environment.Orientation import East, North, South, West

class Agent:
    def __init__(self, location = Coords(0,0), orientation = East, hasGold = False, hasArrow = True, isAlive = True):
        self.location =  location
        self.orientation = orientation
        self.hasGold = hasGold
        self.hasArrow = hasArrow
        self.isAlive = isAlive

    def turnLeft(self):
        ret = deepcopy(self)
        ret.orientation = self.orientation.turnLeft(self)

        return ret

    def turnRight(self):
        ret = deepcopy(self)
        ret.orientation  = self.orientation.turnRight(self)

        return ret

    def useArrow(self):
        ret = deepcopy(self)
        ret.hasArrow = False
    
        return ret

    def forward(self, gridWidth, gridHeight):
        ret = deepcopy(self)
        
        newAgentLocation = False

        if self.orientation == West:
            newAgentLocation = Coords(max(0, self.location.x - 1), self.location.y)
        elif self.orientation == East:
            newAgentLocation = Coords(min(gridWidth - 1, self.location.x + 1), self.location.y)
        elif self.orientation == South:
            newAgentLocation = Coords(self.location.x, max(0, self.location.y - 1))
        elif self.orientation == North:
            newAgentLocation = Coords(self.location.x, min(gridHeight - 1, self.location.y + 1))


        ret.location = newAgentLocation
        
        return ret

    def applyMoveAction(self, action, gridWidth, gridHeight):
        if action == Action.Forward:
            return self.forward(gridWidth, gridHeight)
        elif action == Action.TurnRight:
            return self.turnRight()
        elif action == Action.TurnLeft:
            return self.turnLeft()
