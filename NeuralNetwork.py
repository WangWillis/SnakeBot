import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class NeuralNetwork():
    def init(self, inp, hidden, output):
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
