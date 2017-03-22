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
        self.output = output
    
        #initialize the arrays for the layers
        self.il = [1.0] * self.inp
        self.hl = [1.0] * self.hidden
        self.ol = [1.0] * self.output

        #randomize weights
        self.wi = np.random.randn(self.inp, self.hidden)
        self.wo = np.random.randn(self.hidden, self.output)

    def think(self, inp):
        if len(inp) != self.inp-1:
            return ValueError("BAD INPUT")
    
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

        #propigate activations
        for j in range(self.output):
            tot = 0.0 
            #loop through hidden layer to change values
            for i in range(self.hidden):
                tot += self.hl[i]*self.wo[i][j]
            #set hidden layer values
            self.ol[j] = sigmoid(tot)

        return self.ol

screenX = 20
screenY = 60
initialSnakeSize = 3
MAX_FOOD = screenX*screenY-initialSnakeSize
inputSize = 5
numHiddenLayers = 350
numOutputs = 4
numChrome = 8
gen = 0
key = 0
KEYS = [KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN, KEY_RIGHT]

fitnesses = [0] * numChrome
population = [0] * numChrome
bestFit = 1
bestNN = 0
bestScore = 0

def createScreen(snake, food):
    screen = [0]*inputSize

    for i in range(0, len(snake)):
        screen[snake[i][0]*screenY+snake[i][1]] = 5*(snake[i][0]+snake[i][1])
    screen[snake[0][0]*screenY+snake[0][1]] = 20*(snake[0][0]+snake[0][1])
    screen[food[0]*screenY+food[1]] = 50*(snake[0][0]+snake[0][1])
 
    
    return screen

def getSnakeVal(snake):
    badTouch = 0
    for i in range(1, len(snake)):
        badTouch += i*(snake[0][0]-snake[i][0])
        badTouch += (len(snake)-i)*(snake[0][1]-snake[i][1])
    return badTouch

def getBiggestIndex(arr):
    index = 0
    for i in range(1, len(arr)):
        if arr[i] > arr[index]:
            index = i
    return index

def fitness(res):
    return 1/((exp(res[0])/(15*(res[0]+1)))+((math.log(res[1]*100+1)/100))+1)

def runGame(nn):
    curses.initscr()
    curses.noecho()
    curses.curs_set(0)
    win = curses.newwin(screenX, screenY, 0, 0)
    win.keypad(1)
    win.border(0)
    win.nodelay(1)
    global key
    key = KEY_RIGHT   
    keyVal = 0                                                 # Initializing values
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
        num1 = randint(1,18)
        num2 = randint(1,58)
        food = [num1, num2]
        grid[num1][num2] = 2                 # Calculating next food's coordinates
        if food in snake: food = []
    win.addch(food[0], food[1], '*')

    while key != 27:       
        timeAlive += 0.01                                            # While Esc key is not pressed
        win.border(0)
        win.addstr(0, 2, 'Score: ' + str(score) + ' ')                # Printing 'Score' and
        win.addstr(0, 15, 'Gen: ' + str(gen) + ' ')
        win.addstr(19, 30, 'Sec: ' + str(timeAlive) + ' ')
        win.addstr(0, 27, 'fitness: ' + str(fitness([score, timeAlive])))                # 'SNAKE' strings
        #win.timeout(5)
        cullX = snake[0][0]
        cullY = snake[0][1]
        colPosX = 0
        colPosY = 0
        if key == KEY_LEFT:
        	leftH = 5
    		rightH = 5
    		frontH = 5
        	if snake[0][1]-1 >= 0:
        		frontH = grid[snake[0][0]][snake[0][1]-1]*5
        	if snake[0][0]+1 < screenX:
        		leftH = grid[snake[0][0]+1][snake[0][1]]*5
        	if snake[0][0]-1 >= 0:
        		rightH = grid[snake[0][0]-1][snake[0][1]]*5
            # keyVal = 0
            # colPosY = -1
        elif key == KEY_RIGHT:
        	leftH = 5
    		rightH = 5
    		frontH = 5
    		if snake[0][1]+1 < screenY:
        		frontH = grid[snake[0][0]][snake[0][1]+1]*5
        	if snake[0][0]-1 >= 0:
        		leftH = grid[snake[0][0]-1][snake[0][1]]*5
        	if snake[0][0]+1 < screenX:
        		rightH = grid[snake[0][0]+1][snake[0][1]]*5
            # keyVal = 1
            # colPosY = 1
        elif key == KEY_UP:
        	leftH = 5
    		rightH = 5
    		frontH = 5
    		if snake[0][0]-1 >= 0:
        		frontH = grid[snake[0][0]-1][snake[0][1]]*5
        	if snake[0][1]-1 >= 0:
        		leftH = grid[snake[0][0]][snake[0][1]-1]*5
        	if snake[0][1]+1 < screenY:
        		rightH = grid[snake[0][0]][snake[0][1]+1]*5
            # keyVal = 2
            # colPosX = -1
        else:
        	leftH = 5
    		rightH = 5
    		frontH = 5
    		if snake[0][0]+1 < screenX:
        		frontH = grid[snake[0][0]+1][snake[0][1]]*5
        	if snake[0][1]+1 < screenY:
        		leftH = grid[snake[0][0]][snake[0][1]+1]*5
        	if snake[0][1]-1 >= 0:
        		rightH = grid[snake[0][0]][snake[0][1]-1]*5
            # keyVal = 3
            # colPosX = 1

        # while cullX > 0 and cullX < screenX or cullY > 0 and cullY < screenY:
        #     if grid[cullX][cullY] == 1:
        #         break
        #     cullX += colPosX
        #     cullY += colPosY

        cullX = abs(cullX-snake[0][0])
        if cullX == 0:
        	cullX = abs(cullY-snake[0][1])
        
        xDis = snake[0][0]-food[0]
        yDis = snake[0][1]-food[1]
        avgDist += math.sqrt(xDis*xDis+yDis*yDis)
        prevKey = key
        KEYS[4] = prevKey
        event = KEYS[getBiggestIndex(nn.think([(xDis), (yDis), frontH, leftH, rightH]))]
        done = win.getch()
        key = done if done == 27 else event 

        if key == ord(' '):                                            # If SPACE BAR is pressed, wait for another
            key = -1                                                   # one (Pause/Resume)
            while key != ord(' '):
                key = win.getch()
            key = prevKey
            continue
        
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
                num1 = randint(1,18)
    	        num2 = randint(1,58)
                food = [num1, num2]
                grid[num1][num2] = 2 
                if food in snake: food = []
            win.addch(food[0], food[1], '*')
        else:    
            last = snake.pop()                                          # [1] If it does not eat the food, length decreases
            win.addch(last[0], last[1], ' ')
            grid[last[0]][last[1]] = 0
        win.addch(snake[0][0], snake[0][1], '#')
        grid[snake[0][0]][snake[0][1]] = 1
        if (timeAlive-timeLastScore) > 5:
            break
        
    results.append(score)
    results.append(timeAlive)
    results.append(avgDist/(timeAlive*100))
    curses.endwin()
    return results

def breed(nn, bestNN, fit):
    prob = 0.02+fit*0.785
    nnNew = copy.deepcopy(nn)

    size = nnNew.wi.shape 
    for i in range(0, size[0]):
        for j in range(0, size[1]):
            if random.random() < prob:
                nnNew.wi[i][j] = bestNN.wi[i][j]
    
    size = nnNew.wo.shape 
    for i in range(0, size[0]):
        for j in range(0, size[1]):
            if random.random() < prob:
                nnNew.wo[i][j] = bestNN.wo[i][j]
    return nnNew

def mutate(nn, fit):
    random.seed()
    nnNew = copy.deepcopy(nn)
    prob = 0.1+fit*0.5
    size = nnNew.wi.shape
    for i in range(0, size[0]):
        for j in range(0, size[1]):
            if random.random() < prob:
                nnNew.wi[i][j] += np.random.randn()

    size = nnNew.wo.shape
    for i in range(0, size[0]):
        for j in range(0, size[1]):
            if random.random() < prob:
                nnNew.wo[i][j] += np.random.randn()
    return nnNew

def getMostFit(fitnesses):
    random.seed()
    mostFit = [0, 0.0]
    for i in range(len(fitnesses)):
        mostFit[1] += fitnesses[i]
        if fitnesses[i] < fitnesses[mostFit[0]]:
            mostFit[0] = i
        if fitnesses[i] == fitnesses[mostFit[0]] and random.random() < 0.5:
            mostFit[0] = i

    mostFit[1] /= len(fitnesses)
    return mostFit



#init starting off neural networks
for i in range(len(population)):
    population[i] = NeuralNetwork(inputSize, numHiddenLayers, numOutputs)

random.seed()
welp = 0
#loop for 1000 generations
while key != 27:
    gen += 1
    #play game for whole population
    for i in range(len(population)):
    	#run each nn 4 times and average its fitness
    	for j in range(0,4):
	        res = runGame(population[i])
	        fitnesses[i] += fitness(res)
	        if key == 27:
	            break

        if key == 27:
	    	break
        #print data
        print "Gen: " + str(gen) + " NN: " + str(i)
        print "Score: " + str(res[0]) + " Avg Food Dist: " + str(res[1])
        print "Fitness: " + str(fitnesses[i])

        fitnesses[i] /= 4
        if fitnesses[i] < bestFit:
            bestNN = population[i]
            bestFit = fitnesses[i]
            bestScore = res[0]
            welp = 0
    
    if key == 27:
        break

    #go to next generation
    mostFit = getMostFit(fitnesses) 
    print "Gen: " + str(gen) + " Avg Fitness: " + str(mostFit[1])
    print "Creating next gen"
    
    fitInd = mostFit[0]
    mostFitNum = fitnesses[fitInd]
    mostFit[0] = population[fitInd]

    #for if it diverged too badly
    if welp == 20:
    	mostFit[0] = bestNN
    	mostFitNum = bestFit
    welp += 1

    #swap the best nn to front
    fitnesses[fitInd] = fitnesses[0]
    fitnesses[0] = mostFitNum
    population[fitInd] = population[0]
    population[0] = mostFit[0]

    #make next generation
    for i in range(1, len(population)):
    	#use average fitness to determine if should change
    	if random.random() < (0.1+0.9*mostFit[1]):
    		#determine breeding or mutating
            if random.random() < 0.8:
            	population[i] = breed(population[i], population[0], fitnesses[i])
            elif random.random() < 0.2:
            	population[i] = mutate(population[i], fitnesses[i])
            else:
            	population[i] = mutate(population[0], fitnesses[0])
        
runGame(bestNN)
print "BestNN"
print bestNN.wi
print bestNN.wo
print bestScore
