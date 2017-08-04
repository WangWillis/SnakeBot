# Snake code by sanchitgangwar
# Genetic Learning Alg code by Willis Wang
# SNAKES GAME
# Use ARROW KEYS to play, SPACE BAR for pausing/resuming and Esc Key for exiting

import copy
import tensorflow as tf
import numpy as np
import curses
from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN
from random import randint

key = 0

screenX = 20
screenY = 60

KEYS = [KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN, KEY_RIGHT]
eps = 0.1

convF1Size = 32
convF2Size = 64
connW1Size = 256
numOutputs = 4
def policy_network():
    with tf.variable_scope("policy"):
        gridIn = tf.placeholder("float32", [None, screenX, screenY, 1]) # used to hold grid input
        action = tf.placeholder("float32", [None, 4])
        advantages = tf.placeholder("float32", [None, 1])

        # set up conv nn
        convF1 = tf.get_variable("convF1", [3, 3, 1, convF1Size])
        convF2 = tf.get_variable("convF2", [3, 3, convF1Size, convF2Size])
        connW1 = tf.get_variable("connW1", [5*5*convF2Size, connW1Size])
        outLayer = tf.get_variable("outLayer", [connW1Size, numOutputs])

        # model nn conv layer
        convL1 = tf.nn.conv2d(gridIn, convF1, strides=[1, 1, 1, 1], padding="SAME")
        convL1 = tf.nn.max_pool(convL1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        convL2 = tf.nn.conv2d(convL1, convF2, strides=[1, 1, 1, 1], padding="SAME")
        convL2 = tf.nn.max_pool(convL2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        #fully connected layer
        connIn = tf.reshape(convL2, [-1, 5*5*convF2Size])
        print(gridIn.get_shape(), connIn.get_shape())
        connL1 = tf.matmul(connIn, connW1)
        connL1 = tf.nn.relu(connL1)
        poli_out = tf.matmul(connL1, outLayer)
        poli_out = tf.nn.softmax(poli_out)

        prob = tf.multiply(tf.log(poli_out), action)
        loss = -(tf.reduce_sum(prob*advantages))
        optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)

        return poli_out, gridIn, action, advantages, optimizer

inL1Size = 30
hL1Size = 40
outSize = 1
def value_network():
    with tf.variable_scope("value"):
        state = tf.placeholder("float32", [None, screenX*screenY])

        newvals = tf.placeholder("float",[None, 1])

        w1 = tf.get_variable("w1", [screenX*screenY, inL1Size])
        b1 = tf.get_variable("b1", [inL1Size])
        h1 = tf.nn.relu(tf.matmul(state, w1) + b1)
        w2 = tf.get_variable("w2", [inL1Size, hL1Size])
        b2 = tf.get_variable("b2", [hL1Size])
        calculated = tf.nn.relu(tf.matmul(h1,w2) + b2)

        outW = tf.get_variable("outW", [hL1Size, outSize])
        outB = tf.get_variable("outB", [outSize])
        val_out = tf.matmul(calculated, outW) + outB

        diffs = val_out - newvals
        loss = tf.nn.l2_loss(diffs)
        optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)
        return calculated, state, newvals, optimizer, loss


def runGame(poliOut, gridInput):
    curses.initscr()
    curses.noecho()
    curses.curs_set(0)
    win = curses.newwin(screenX, screenY, 0, 0)
    win.keypad(1)
    win.border(0)
    win.nodelay(1)
    #win.timeout(1)
    global key
    key = KEY_RIGHT                                                # Initializing values
    
    score = 0
    timeAlive = 0
    timeLastScore = 0

    #store the states and return for the learning
    states = []
    actions = []
    transitions = []

    results = []
    
    grid = [[0 for x in range(screenY)] for y in range(screenX)]
    snake = [[4,10], [4,9], [4,8]]                                     # Initial snake co-ordinates
    grid[4][10] = 1
    grid[4][9] = 1
    grid[4][8] = 1

    food = []

    while food == []:
        foodX = randint(1, screenX-2)
        foodY = randint(1, screenY-2)
        if(grid[foodX][foodY] != 0):
            continue
        food = [foodX, foodY]
        grid[foodX][foodY] = 2

    win.addch(food[0], food[1], '*')

    while key != 27:
        timeAlive += 0.01                                            # While Esc key is not pressed
        win.border(0)
        win.addstr(0, 2, 'Score: ' + str(score) + ' ')                # Printing 'Score' and

        # controls stuff
        action = [0, 0, 0, 0] #one hot vector representing which key pressed
        nextKey = 0
        if(np.random.uniform(0,1) < eps):
            nextKey = randint(0, 3)
        else:
            gridIn = np.reshape(grid, [-1, screenX, screenY, 1])
            qValArr = sess.run(poliOut, feed_dict={gridInput: gridIn})[0]
            currMin = 0
            randNum = np.random.uniform(0,1)
            for i in range(len(qValArr)):
                if(randNum >= currMin and randNum < qValArr[i]):
                    nextKey = i
                    break
                currMin += qValArr[i]
        action[nextKey] = 1
        event = KEYS[nextKey]

        done = win.getch()
        key = done if done == 27 else event 

        transition = [grid, action, 0]

        snake.insert(0, [snake[0][0] + (key == KEY_DOWN and 1) + (key == KEY_UP and -1), snake[0][1] + (key == KEY_LEFT and -1) + (key == KEY_RIGHT and 1)])
        
        if snake[0] == food:                                            # When snake eats the food
            food = []
            score += 1
            transition[2] = 5
            timeLastScore = timeAlive

            while food == []:
                foodX = randint(1, screenX-2)
                foodY = randint(1, screenY-2)
                if(grid[foodX][foodY] != 0):
                    continue
                food = [foodX, foodY]
                grid[foodX][foodY] = 2

            win.addch(food[0], food[1], '*')
        else:    
            last = snake.pop()                                          # [1] If it does not eat the food, length decreases
            win.addch(last[0], last[1], ' ')
            grid[last[0]][last[1]] = 0
        win.addch(snake[0][0], snake[0][1], '#')
        grid[snake[0][0]][snake[0][1]] = 1

        states.append(transition[0])
        actions.append(transition[1])
        transitions.append(transition)

        if(key == 27):
            break

        if snake[0][0] <= 0 or snake[0][0] >= screenX-1 or snake[0][1] <= 0 or snake[0][1] >= screenY-1: break

        if snake[0] in snake[1:]: break

        if (timeAlive-timeLastScore) > 1.5:
            break
        key = win.getch()
    #curses.endwin()

    return transitions, states, actions

policy_prob, grid_in, act, advantage, opt = policy_network()
val_value, val_grid, val_correct, val_opt, val_loss = value_network()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

while key != 27:
    transitions, states, actions = runGame(policy_prob, grid_in)

    advantages = []
    update_vals = []
    i = 0
    for transition in transitions:
        grid, action, reward = transition
        grid = np.reshape(grid, [-1, screenX*screenY])
        # calculate discounted monte-carlo return
        futReward = 0
        nextTranstions = len(transitions)-i
        gam = 1
        for j in range(nextTranstions):
            futReward += transitions[i+j][2] * gam
            gam = gam * 0.97

        val = sess.run(val_value, feed_dict={val_grid: grid})[0][0]

        # advantage: how much better was this action than normal
        advantages.append(futReward - val)

        # update the value function towards new return
        update_vals.append(futReward)
        i += 1
    print(len(states), len(actions))
    # update value function
    states_reshape = np.reshape(states, [-1, screenX*screenY])
    update_vals_vector = np.expand_dims(update_vals, axis=1)
    sess.run(val_opt, feed_dict={val_grid: states_reshape, val_correct: update_vals_vector})
    # real_vl_loss = sess.run(vl_loss, feed_dict={vl_state: states, vl_newvals: update_vals_vector})

    states = np.reshape(states_reshape, [len(states), screenX, screenY, 1])
    advantages_vector = np.reshape(advantages, [len(advantages), 1])
    actions = np.reshape(actions, [len(actions), 4])
    print(len(states), len(advantages), len(actions))
    sess.run(opt, feed_dict={grid_in: states, advantage: advantages_vector, act: actions})