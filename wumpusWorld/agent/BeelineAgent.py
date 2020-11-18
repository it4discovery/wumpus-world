from wumpusWorld.environment.Agent import Agent
from wumpusWorld.environment.Environment import Coords, Action, Percept
from wumpusWorld.environment.Orientation import East, North, South, West
from networkx import nx
from copy import deepcopy
from random import randrange
import sys
sys.path.append(".")


class BeelineAgent():
    def __init__(self, gridHeight=4, gridWidth=4, agentState=Agent, safeLocations=[], beelineActionList=[]):
        self.gridHeight = gridHeight
        self.gridWidth = gridWidth
        self.agentState = agentState()
        self.safeLocations = safeLocations
        self.beelineActionList = beelineActionList

    def buildEscapeRoute(self, safeLocations):
        G = nx.grid_2d_graph(self.gridWidth, self.gridHeight)

        for location in safeLocations:
            nx.add_path(G, (location.x, location.y))

        escapeRoute = list(nx.shortest_path(
            G, (safeLocations[-1].x, safeLocations[-1].y), (0, 0)))

        return escapeRoute

    def shouldBeFacing(self, currentItem, nextItem):
        currX = currentItem[0]
        currY = currentItem[1]
        nextX = nextItem[0]
        nextY = nextItem[1]   

        if(nextY < currY):
            return South      
        elif(nextX < currX):
            return West

    def determineNextAction(self, agentOrientation, escapeRouteActions):
        currentEscapeAction = Action.Forward

        if(agentOrientation != self.shouldBeFacing(escapeRouteActions[0], escapeRouteActions[1])):
            currentEscapeAction = Action.TurnRight
        else:
            escapeRouteActions = escapeRouteActions[1:]

        return currentEscapeAction, escapeRouteActions

    def nextAction(self, percept):
        randGen = randrange(4)
        ret = deepcopy(self)

        if ret.agentState.hasGold == True:
            if(self.agentState.location.x == Coords(0, 0).x and self.agentState.location.y == Coords(0, 0).y):
                return ret, Action.Climb
            else:
                escapeRouteActions = self.buildEscapeRoute(self.safeLocations) if len(self.beelineActionList) == 0 else self.beelineActionList 
                
                currentEscapeAction, escapeRouteActions =  self.determineNextAction(ret.agentState.orientation, escapeRouteActions)

                ret.agentState = ret.agentState.applyMoveAction(currentEscapeAction, self.gridWidth, self.gridHeight)
                ret.beelineActionList = escapeRouteActions

                return ret, currentEscapeAction
        elif percept.glitter == True:
            ret.agentState.hasGold = True
            return ret, Action.Grab
        elif randGen == 0:
            ret.agentState = ret.agentState.forward(
                self.gridWidth, self.gridHeight)
            ret.safeLocations.append(ret.agentState.location)
            return ret, Action.Forward
        elif randGen == 1:
            ret.agentState = ret.agentState.turnLeft()
            return ret, Action.TurnLeft
        elif randGen == 2:
            ret.agentState = ret.agentState.turnRight()
            return ret, Action.TurnRight
        elif randGen == 3:
            ret.agentState = ret.agentState.useArrow()
            return ret, Action.Shoot
