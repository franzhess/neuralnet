import numpy as np
import matplotlib.pyplot as plt


class Sigmoid:
    def execute(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return x * (1 - x)


class TanH:
    def execute(self, x):
        return (2 / (1 + np.exp(-x))) - 1

    def derivative(self, x):
        return 1 - x * x

class NeuralNet:
    def __init__(self, inputNodes, hiddenNodes, outputNodes, activationFunction=Sigmoid()):
        self.hiddenLayer = self.initLayer(inputNodes + 1, hiddenNodes)
        self.outputLayer = self.initLayer(hiddenNodes + 1, outputNodes)
        self.activationFunction = activationFunction

    def initLayer(self, rows, width):
        return 2 * np.random.random((rows, width)) - 1

    def addBias(self, vector):
        return np.hstack([vector, np.ones((vector.shape[0], 1))])

    def forward(self, input):
        self.input = self.addBias(input);
        self.hiddenResult = self.addBias(self.activationFunction.execute(np.dot(self.input, self.hiddenLayer)))
        self.output = self.activationFunction.execute(np.dot(self.hiddenResult, self.outputLayer))
        return self.output

    def backward(self, result):
        outputError = result - self.output
        outputDelta = outputError * self.activationFunction.derivative(self.output)

        # ignore the last row of the output layer since it's for the bias
        hiddenError = outputDelta.dot(self.outputLayer[:-1, :].T)
        hiddenDelta = hiddenError * self.activationFunction.derivative(self.hiddenResult[:, :-1])

        self.outputLayer += self.hiddenResult.T.dot(outputDelta)
        self.hiddenLayer += self.input.T.dot(hiddenDelta)

    def train(self, input, result, epochs):
        for epoch in range(0, epochs):
            self.forward(input)
            self.backward(result)

    def trainsingle(self, input, result, epochs, subsetsize=1):
        for epoch in range(0, epochs):
            for i in range(0, (input.shape[0] - 1) // subsetsize):
                self.forward(np.array(input[i * subsetsize:i * subsetsize + subsetsize, :]))
                self.backward(np.array(result[i * subsetsize:i * subsetsize + subsetsize, :]))

    def print(self):
        print('hidden')
        print(self.hiddenLayer)
        print('output')
        print(self.outputLayer)


def printresult(result, referencResult):
    print('result')
    print(result)
    print('error', np.mean(np.abs(referencResult - result)))


def printAbsAvgErrorOverEpoch(network, trainingInput, trainingResult, epochs, sample=1000):
    index = []
    error = []

    for i in range(1, epochs // sample):
        network.trainSingle(trainingInput, trainingResult, sample, 8)
        result = network.forward(trainingInput);
        index.append(i * epochs)
        error.append(np.mean(np.abs(trainingResult - result)))

    plt.plot(index, error)
    plt.xlabel('epochs')
    plt.ylabel('avg abs error')
    plt.show()


def compareNetworks(network1, network2, trainingInput, trainingResult, epochs, sample=1000):
    index = []
    error1 = []
    error2 = []

    for i in range(1, epochs // sample):
        index.append(i * epochs)

        network1.train(trainingInput, trainingResult, sample)
        result = network1.forward(trainingInput);
        error1.append(np.mean(np.abs(trainingResult - result)))

        network2.train(trainingInput, trainingResult, sample)
        result = network2.forward(trainingInput);
        error2.append(np.mean(np.abs(trainingResult - result)))

    plt.plot(index, error1)
    plt.plot(index, error2)
    plt.xlabel('epochs')
    plt.ylabel('avg abs error')
    plt.legend(['network1', 'network2'])
    plt.show()
