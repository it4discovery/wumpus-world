from wumpusWorld.environment.Agent import Agent
from wumpusWorld.environment.Environment import Coords, Action, Percept
from wumpusWorld.environment.Orientation import East, North, South, West
from copy import deepcopy
from random import randrange
import sys
import numpy as np
sys.path.append(".")


class DeepQAgent():
    def __init__(self, gridHeight=4, gridWidth=4, agentState=Agent, safeLocations=[Coords(0,0)], stenchLocations=[], breezeLocations=[], agentLocationGrid = [], safeLocationGrid = [], stenchLocationGrid = [], breezeLocationGrid = [], agentHasGold = False, agentSensesGold = False, agentHasArrow = False, agentHeardScream = False, agentOrientationSet= False):
        self.gridHeight = gridHeight
        self.gridWidth = gridWidth
        self.agentState = agentState()
        self.safeLocations = safeLocations
        self.stenchLocations = stenchLocations
        self.breezeLocations = breezeLocations
        self.agentLocationGrid = agentLocationGrid
        self.safeLocationGrid = safeLocationGrid
        self.stenchLocationGrid = stenchLocationGrid
        self.breezeLocationGrid = breezeLocationGrid
        self.agentHasGold = agentHasGold
        self.agentSensesGold = agentSensesGold
        self.agentHasArrow = agentHasArrow
        self.agentHeardScream = agentHeardScream
        self.agentOrientationSet = agentOrientationSet

    def printTable(self, grid):
        rows = []
        for i in range(self.gridHeight):
            cells = []
            for j in range(self.gridWidth):            
                cells.append("%s" % (grid[i][j]))

            rows.append('|'.join(cells))
        return '\n'.join(rows)

    def printBeliefState(self, percept):
        print("-- Percept --")
        print(percept.show())

        print('-- Current Agent Location --')
        print(self.printTable(self.agentLocationGrid))

        print('-- Agent Orientation Set --')
        print(self.agentOrientationSet)        

        print('-- Safe Locations --')
        print(self.printTable(self.safeLocationGrid))

        print('-- Stench Locations --')
        print(self.printTable(self.stenchLocationGrid))

        print('-- Breeze Locations --')
        print(self.printTable(self.breezeLocationGrid))    

        print('-- Agent Has Gold --')
        print(self.agentHasGold)  

        print('-- Agent Senses Gold --')
        print(self.agentSensesGold)  

        print('-- Agent Has Arrow --')
        print(self.agentHasArrow)  

        print('-- Agent Heard Scream --')
        print(self.agentHeardScream)          

    def getAgentBeliefState(self):       
        #flattening and concatenating all the arrays to one array of length 72
        return np.concatenate((np.array(self.agentLocationGrid).flatten(), np.array(self.safeLocationGrid).flatten(),np.array(self.stenchLocationGrid).flatten(), np.array(self.breezeLocationGrid).flatten(), np.array(self.agentOrientationSet).flatten(), np.array(self.agentHasGold).flatten(), np.array(self.agentSensesGold).flatten(), np.array(self.agentHasArrow).flatten(), np.array(self.agentHeardScream).flatten()))


    def buildAgentOrientatioSet(self, orientation):
        #Set is represented as a list of 1s and 0s
        #I don't think the order matters, but we'll say:
        #  [1, 0, 0, 0] = North
        #  [0, 1, 0, 0] = South
        #  [0, 0, 1, 0] = East
        #  [0, 0, 0, 1] = West
        orientationSet = []

        if orientation == North:
            orientationSet.append([1, 0, 0, 0])
        elif orientation == South:
            orientationSet.append([0, 1, 0, 0])
        elif orientation == East:
            orientationSet.append([0, 0, 1, 0])
        elif orientation == West:
            orientationSet.append([0, 0, 0, 1])

        return orientationSet

    def buildAgentLocationGrid(self, coords):
        rows = []

        for i in range(self.gridHeight):
            cols = []
            for j in range(self.gridWidth):
                if(i == coords.y and j == coords.x):
                    cols.append(1)
                else:
                    cols.append(0)   

            rows.append(cols)

        return rows

    def buildSafeLocationGrid(self, visited):
        rows = []

        for i in range(self.gridHeight):
            cols = []
            for j in range(self.gridWidth):
                if any(d.y == i and d.x == j for d in visited):
                    cols.append(1)
                else:
                    cols.append(0)   

            rows.append(cols)

        return rows            

    def buildStenchLocationGrid(self, stenches):
        rows = []

        for i in range(self.gridHeight):
            cols = []
            for j in range(self.gridWidth):
                if any(d.y == i and d.x == j for d in stenches):
                    cols.append(1)
                else:
                    cols.append(0)   

            rows.append(cols)

        return rows 

    def buildBreezeLocationGrid(self, breezes):
        rows = []

        for i in range(self.gridHeight):
            cols = []
            for j in range(self.gridWidth):
                if any(d.y == i and d.x == j for d in breezes):
                    cols.append(1)
                else:
                    cols.append(0)   

            rows.append(cols)

        return rows

    def nextAction(self, percept, action):
        ret = deepcopy(self)

        if(percept.stench == True):
            ret.stenchLocations.append(ret.agentState.location)
        if(percept.breeze == True):
            ret.breezeLocations.append(ret.agentState.location) 

        ret.agentLocationGrid = self.buildAgentLocationGrid(ret.agentState.location)
        ret.safeLocationGrid = self.buildSafeLocationGrid(ret.safeLocations)  
        ret.stenchLocationGrid = self.buildStenchLocationGrid(ret.stenchLocations)       
        ret.breezeLocationGrid = self.buildBreezeLocationGrid(ret.breezeLocations)
        ret.agentHasGold =  1 if ret.agentState.hasGold == True else 0
        ret.agentSensesGold = 1 if percept.glitter == True else 0
        ret.agentHasArrow = 1 if ret.agentState.hasArrow == True else 0
        ret.agentHeardScream = 1 if percept.scream == True else 0
        ret.agentOrientationSet = self.buildAgentOrientatioSet(ret.agentState.orientation)

        if action == 0:
            ret.agentState = ret.agentState.forward(
                self.gridWidth, self.gridHeight)
            ret.safeLocations.append(ret.agentState.location)

            return ret, Action.Forward
        elif action == 1:
            ret.agentState = ret.agentState.turnLeft()
            return ret, Action.TurnLeft
        elif action == 2:
            ret.agentState = ret.agentState.turnRight()
            return ret, Action.TurnRight
        elif action == 3:
            ret.agentState = ret.agentState.useArrow()
            return ret, Action.Shoot
        if action == 4:
            if percept.glitter == True:
                ret.agentState.hasGold = True

            return ret, Action.Grab
        if action == 5:
            ret.agentState.isTerminated = True
            return ret, Action.Climb                            