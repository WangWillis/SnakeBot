# Snake code by sanchitgangwar
# Genetic Learning Alg code by Willis Wang
# SNAKES GAME
# Use ARROW KEYS to play, SPACE BAR for pausing/resuming and Esc Key for exiting

import copy
import math
import numpy as np
from numpy import exp, array, random, dot
import curses
from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN
from random import randint

def sigmoid(x):
    return 1/(1+np.exp(-x))

class NeuralNetwork():
    def __init__(self, inp, hidden, output):
        #initialize sizes
        self.inp = inp+1
        self.hidden = hidden
        #self.mHidden = hidden/4
        self.output = output
    
        #initialize the arrays for the layers
        self.il = [1.0] * self.inp
        self.hl = [1.0] * self.hidden
        #self.hhl = [1.0] * self.mHidden
        self.ol = [1.0] * self.output

        #randomize weights
        self.wi = np.random.randn(self.inp, self.hidden)
        #self.wh = np.random.randn(self.hidden, self.mHidden)
        self.wo = np.random.randn(self.hidden, self.output)

    def __getitem__(self):
        return self

    def think(self, inp):
        if len(inp) != self.inp-1:
            return ValueError("BAD INPUT")
        softMax = 0
        #copy data to input
        for i in range(self.inp-1):
            self.il[i] = inp[i]
    
        #propigate through hidden layers
        for j in range(self.hidden):
            tot = 0.0 
            #loop through hidden layer to change values
            for i in range(self.inp):
                tot += self.il[i]*self.wi[i][j]
            #set hidden layer values
            self.hl[j] = sigmoid(tot)

        # #propigate through hidden layers
        # for j in range(self.mHidden):
        #     tot = 0.0 
        #     #loop through hidden layer to change values
        #     for i in range(self.hidden):
        #         tot += self.hl[i]*self.wh[i][j]
        #     #set hidden layer values
        #     self.hhl[j] = sigmoid(tot)

        #propigate activations
        for j in range(self.output):
            tot = 0.0 
            #loop through hidden layer to change values
            for i in range(self.hidden):
                tot += self.hl[i]*self.wo[i][j]
            #set hidden layer values
            sig = sigmoid(tot)
            self.ol[j] = sig
            softMax += exp(sig)

        #calculate prob using softMax
        for i in range(self.output):
            self.ol[i] = exp(self.ol[i])/softMax

        return self.ol

screenX = 20
screenY = 60
inputSize =7
numHiddenLayers = 300
numOutputs = 4
key = 0
KEYS = [KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN, KEY_RIGHT]

def runGame(nn):
    curses.initscr()
    curses.noecho()
    curses.curs_set(0)
    win = curses.newwin(screenX, screenY, 0, 0)
    win.keypad(1)
    win.border(0)
    win.nodelay(1)
    global key
    key = KEY_RIGHT                                                # Initializing values
    score = 0
    timeAlive = 0
    timeLastScore = 0 
    avgDist = 0
    results = []
    
    grid = [[0 for x in range(screenY)] for y in range(screenX)]
    snake = [[4,10], [4,9], [4,8]]                                     # Initial snake co-ordinates
    grid[4][10] = 1
    grid[4][9] = 1
    grid[4][8] = 1
    food = []

    leftH = 0
    rightH = 0
    frontH = 0

    while food == []:
        food = [randint(1, 18), randint(1, 58)]                 # Calculating next food's coordinates
        if food in snake: food = []
    win.addch(food[0], food[1], '*')

    while key != 27:
        timeAlive += 0.01                                            # While Esc key is not pressed
        win.border(0)
        win.addstr(0, 2, 'Score: ' + str(score) + ' ')                # Printing 'Score' and
        win.addstr(0, 15, 'Gen: ' + str(gen) + ' ')
        win.addstr(19, 30, 'Sec: ' + str(timeAlive) + ' ')
        win.addstr(0, 27, 'fitness: ' + str(fitness([score, timeAlive, avgDist/(timeAlive*100)])))                # 'SNAKE' strings
        #win.timeout(5)
        
        
        xDis = snake[0][0]-food[0]
        yDis = snake[0][1]-food[1]
        avgDist += math.sqrt(xDis*xDis+yDis*yDis)
        cull = getBadDis(snake, grid, key)
        prevKey = key
        KEYS[4] = prevKey
        event = KEYS[getBiggestIndex(nn.think([math.pow(xDis, 3), xDis, math.pow(yDis, 3), yDis, cull[0], cull[1], cull[2]]))]
        done = win.getch()
        key = done if done == 27 else event 
        
        if ((key == KEY_LEFT and prevKey == KEY_RIGHT) or (key == KEY_RIGHT and prevKey == KEY_LEFT) or (key == KEY_UP and prevKey == KEY_DOWN) or (key == KEY_DOWN and prevKey == KEY_UP)):
            key = prevKey

        snake.insert(0, [snake[0][0] + (key == KEY_DOWN and 1) + (key == KEY_UP and -1), snake[0][1] + (key == KEY_LEFT and -1) + (key == KEY_RIGHT and 1)])
        
        if snake[0][0] <= 0 or snake[0][0] >= screenX-1 or snake[0][1] <= 0 or snake[0][1] >= screenY-1: break

        if snake[0] in snake[1:]: break
        
        if snake[0] == food:                                            # When snake eats the food
            food = []
            score += 1
            timeLastScore = timeAlive

            while food == []:
                food = [randint(1, 18), randint(1, 58)]                 # Calculating next food's coordinates
                if food in snake: food = []
            win.addch(food[0], food[1], '*')
        else:    
            last = snake.pop()                                          # [1] If it does not eat the food, length decreases
            win.addch(last[0], last[1], ' ')
            grid[last[0]][last[1]] = 0
        win.addch(snake[0][0], snake[0][1], '#')
        grid[snake[0][0]][snake[0][1]] = 1

        if (timeAlive-timeLastScore) > 1.5:
            break
        key = win.getch()
        
    results.append(score)
    results.append(timeAlive)
    results.append(avgDist/(timeAlive*100))
    curses.endwin()
    return results