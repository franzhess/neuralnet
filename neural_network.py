from timeit import default_timer
import numpy as np
import ann

XORinput = np.array([[0, 0],
                     [1, 0],
                     [0, 1],
                     [1, 1]])

XORresult = np.array([[0],
                      [1],
                      [1],
                      [0]])

# works with tanh as activation function
XORinput2 = np.array([[-1, -1],
                      [1, -1],
                      [-1, 1],
                      [1, 1]])

XORresult2 = np.array([[-1],
                       [1],
                       [1],
                       [-1]])

INCinput = np.array([[0, 0, 0],
                     [0, 0, 1],
                     [0, 1, 0],
                     [0, 1, 1],
                     [1, 0, 0],
                     [1, 0, 1],
                     [1, 1, 0],
                     [1, 1, 1]])

INCoutput = np.array([[0, 0, 1],
                      [0, 1, 0],
                      [0, 1, 1],
                      [1, 0, 0],
                      [1, 0, 1],
                      [1, 1, 0],
                      [1, 1, 1],
                      [0, 0, 0]])


def intToBit(x):
    arr = np.array([1 if digit == '1' else 0 for digit in bin(x)[2:]])
    return np.pad(arr, (8 - arr.shape[0], 0), 'constant')


def shifting(bitlist):
    out = 0
    for bit in bitlist:
        out = (out << 1) | bit
    return out


def createInputs():
    inputs = [];
    for i in range(0, 256):
        inputs.append(intToBit(i))
    return np.array(inputs)


def createOutputs():
    outputs = [];
    for i in range(0, 256):
        outputs.append(intToBit(i // 2))
    return np.array(outputs)


def test():
    inputs = createInputs()
    outputs = createOutputs()
    network = ann.NeuralNet(8, 8, 8)
    start = default_timer()
    network.trainSingle(inputs, outputs, 100, 8)
    end = default_timer()
    result = network.forward(inputs)
    ann.printResult(np.round(result), outputs)
    print('training took', end - start, 'seconds')


network = ann.NeuralNet(2, 2, 1, ann.TanH())
start = default_timer()
network.train(XORinput2, XORresult2, 20000)
end = default_timer()
result = network.forward(XORinput2)
ann.printResult(np.round(result), XORresult2)
print('training took', end - start, 'seconds')

network = ann.NeuralNet(2, 2, 1)
start = default_timer()
network.train(XORinput, XORresult, 20000)
end = default_timer()
result = network.forward(XORinput)
ann.printResult(np.round(result), XORresult)
print('training took', end - start, 'seconds')
