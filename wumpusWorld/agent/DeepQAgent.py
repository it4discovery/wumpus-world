from copy import deepcopy
from random import randrange
import sys
import numpy as np
sys.path.append(".")
from wumpusWorld.environment.Agent import Agent
from wumpusWorld.environment.Environment import Coords, Action, Percept
from wumpusWorld.environment.Orientation import East, North, South, West



class DeepQAgent():
    def __init__(self, gridHeight=4, gridWidth=4, agentState=Agent, safeLocations=[Coords(0,0)], stenchLocations=[], breezeLocations=[], agentLocationGrid = [], safeLocationGrid = [], stenchLocationGrid = [], breezeLocationGrid = [], agentHasGold = False, agentSensesGold = False, agentHasArrow = False, agentHeardScream = False, agentOrientationSet= False, previousAction = [], previousLocation = [], sameMovesSet = [], sameLocationSet = []):
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
        self.previousAction = previousAction
        self.previousLocation = previousLocation
        self.sameMovesSet = sameMovesSet
        self.sameLocationSet = sameLocationSet

    #Helper method to print grids nicely
    def printTable(self, grid):
        rows = []
        for i in range(self.gridHeight):
            cells = []
            for j in range(self.gridWidth):            
                cells.append("%s" % (grid[i][j]))

            rows.append('|'.join(cells))
        return '\n'.join(rows)

    #Helper method to print all the different belief state values
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

    #flattening and concatenating all the arrays to one array of length 78
    def getAgentBeliefState(self):          
        return np.concatenate((np.array(self.sameLocationSet).flatten(), np.array(self.sameMovesSet).flatten(), np.array(self.agentLocationGrid).flatten(), np.array(self.safeLocationGrid).flatten(),np.array(self.stenchLocationGrid).flatten(), np.array(self.breezeLocationGrid).flatten(), np.array(self.agentOrientationSet).flatten(), np.array(self.agentHasGold).flatten(), np.array(self.agentSensesGold).flatten(), np.array(self.agentHasArrow).flatten(), np.array(self.agentHeardScream).flatten()))

    #Set is represented as a list of 1s and 0s
    #I don't think the order matters, but we'll say:
    #  [1, 0, 0, 0] = North
    #  [0, 1, 0, 0] = South
    #  [0, 0, 1, 0] = East
    #  [0, 0, 0, 1] = West
    def buildAgentOrientatioSet(self, orientation):
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

    #Build counters for last number of moves and if they are the same
    def buildSameMoves(self, moves):
      sequence = 1
      reversed_moves = moves[::-1]
      fiveSame = 0         
      tenSame = 0         
      fiftySame = 0         
      for index, move in enumerate(reversed_moves):
          if index + 1 == len(reversed_moves):
              break
          
          if(move == reversed_moves[index + 1]):
              sequence += 1
          else:
              break
      if sequence >= 5:
        fiveSame = 1

      if sequence >= 10:
        tenSame = 1

      if sequence >= 50:
        fiftySame = 1  

      return [fiveSame, tenSame, fiftySame] 

    #Build counters for last number of locations and if they are the same
    def buildSameLocations(self, locations):
      sequence = 1
      reversed_locations = locations[::-1]
      fourSame = 0         
      tenSame = 0         
      fiftySame = 0         
      for index, location in enumerate(reversed_locations):
          if index + 1 == len(reversed_locations):
              break
          
          if(location.x == reversed_locations[index + 1].x and location.y == reversed_locations[index + 1].y):
              sequence += 1
          else:
              break
      if sequence >= 4:
        fourSame = 1

      if sequence >= 10:
        tenSame = 1

      if sequence >= 50:
        fiftySame = 1  

      return [fourSame, tenSame, fiftySame]                      

    #Build agent location grid
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
    
    #Build safe location grid
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

    #Build stench location grid
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
    
    #Build breeze location grid
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

    #apply next action. Action value is passed in from training(0,1,2,4,5)
    def nextAction(self, percept, action):
        ret = deepcopy(self)
        
        ret.previousAction.append(action)
        ret.previousLocation.append(ret.agentState.location)
        
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
        ret.sameMovesSet = self.buildSameMoves(ret.previousAction)
        ret.sameLocationSet = self.buildSameLocations(ret.previousLocation)

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
            if(ret.agentState.location.x == Coords(0,0).x and ret.agentState.location.y == Coords(0,0).y):
                ret.agentState.isTerminated = True
                
            return ret, Action.Climb                            