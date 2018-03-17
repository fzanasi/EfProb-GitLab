#
# Experiments with Neural networks in EfProb
#
# Copyright: Bart Jacobs; 
# Radboud University Nijmegen
# efprob.cs.ru.nl
#
# Date: 2018-02-01
#
from efprob_dc import *

def sigmoid(x):
    """The sigmoid function."""
    return 1.0/(1.0 + np.exp(-x))

# def sigmoid_inv(y):
#     """The sigmoid function."""
#     return np.log(y/(1.0-y))

def unit_vectors(n):
    """ return all n unit vectors (1,0..), (0,1,..) etc of length n """
    output = []
    for i in range(n):
        ls = np.zeros(n)
        ls[i] = 1
        output.append(ls)
    return output

class Layer:
    """ Values for a nodes of a layer in a Neural Network """
    def __init__(self, values):
        self.dom = len(values)
        self.arr = np.array(values)

    def __repr__(self):
        return str(self.arr)

    def __mul__(self, other):
        """ Pointwise multiplication """
        return Layer(self.arr * other.arr)

    def __sub__(self, other):
        """ Pointwise difference """
        return Layer(self.arr - other.arr)

    def __invert__(self):
        """ Pointwise orthosupplement, sending entry x to 1-x """
        return Layer(1.0 - self.arr)

    def inc(self):
        """ extend the array with a value 1 at the end """
        return np.array(list(self.arr) + [1])

    def dec(self):
        """ remove the last entry from the array """
        return Layer(self.arr[0:self.dom-1])



class NeuralChannel:
    """ Neural channels """
    def __init__(self, matrix, eta=0.5):
        """ nxm matrix gives EfProb channel n -> m; bias is in last row """
        if len(matrix.shape) != 2:
            raise Exception('A 2-dimensional matrix is required in a neural channel')
        self.mat = matrix
        self.dom = matrix.shape[0] - 1
        self.cod = matrix.shape[1]
        self.eta = eta

    def __repr__(self):
        return "Neural channel from {} nodes to {} nodes".format(self.dom, 
                                                                 self.cod)

    def __rshift__(self, layer):
        """ forward propagation; an entry 1 is added to handle the bias """
        arr = np.dot(self.mat, layer.inc())
        # apply sigmoid elementwise, and return as layer
        return Layer(sigmoid(arr))

    def __lshift__(self, layer):
        """ backward propagation """
        arr = np.dot(layer.arr, self.mat)
        # apply sigmoid elementwise, and return as layer
        return Layer(arr)

    def biasless_mat(self):
        """ matrix without bias """
        ls = []
        for i in range(self.mat.shape[0]):
            ls.append(self.mat[i][0:self.mat.shape[1]-1])
        return np.array(ls)

    def update_outer(self, input, target):
        """ Update the channel when it is used as output """
        output = self >> input
        out = (output - target) * output * ~output
        input_inc = input.inc()
        #print("outer")
        #print( np.outer(out.arr, input_inc) )
        self.mat -= self.eta * np.outer(out.arr, input_inc)
        print( self.mat )

    def update_hidden(self, input, target, next_nchan):
        """ Update the channel when it is used in hidden form """
        # next state
        output = self >> input
        print("Intermediate", output * ~output )
        next_output = next_nchan >> output
        next_diff = (next_output - target) * next_output * ~next_output
        tmp = np.dot(np.dot(np.dot((next_output - target).arr,
                                   np.diag((next_output * ~next_output).arr)),
                            next_nchan.biasless_mat()),
                     np.diag((output * ~output).arr))
        print("\nMatrix form:\n", np.outer(tmp, input.inc()) )
        error = (next_nchan << next_diff).dec()
        out = error * output * ~output
        print("\nAd hoc form\n", np.outer(out.arr, input.inc()) )
        self.mat -= self.eta * np.outer(out.arr, input.inc())
        print( self.mat )




def random_nchan(n, m):
    """ random neural channel from n nodes to m nodes """
    mat = np.random.randn(m, n+1)
    return NeuralChannel(mat)



class NeuralNetwork:
    """ Neural Network """
    def __init__(self, node_list, eta=0.5):
        """ nxm matrix gives EfProb channel n -> m; bias is in last row """
        if len(node_list) < 2:
            raise Exception('A neural network requires at least 2 layers')
        self.node_list = node_list
        channel_list = []
        for i in range(len(node_list) - 1):
            channel_list.append(random_nchan(node_list[i], node_list[i+1]))
        self.channels = channel_list
        self.eta = eta

    def __repr__(self):
        return "networks with nodes " + str(self.node_list)

    def __rshift__(self, state):
        """ forward pass of a state through the network """
        for i in range(len(self.channels)):
            state = self.channels[i] >> state
        return state

    def update(self, state, target):
        #
        # Apply forward computations iteratively, giving a list of
        # random variables (without biases)
        #
        forwards = [state]
        for i in range(len(self.channels)):
            fv = forwards[len(forwards) - 1]
            forwards.append(self.channels[i] >> fv)
        print("Length of forwards:", len(forwards) )
        #
        # next compute backwards the iterated delta's, 
        #
        last_state = forwards[len(forwards) - 1]
        backwards = [ ((last_state - target) * last_state * ~last_state).arr ]
        for i in range(len(self.channels)-1, -1, -1):
            loss = backwards[0]
            backwards[0] = np.outer(loss, forwards[i].inc())
            backwards = [ np.dot(np.diag((forwards[i] * ~forwards[i]).arr),
                                 np.dot(self.channels[i].biasless_mat(),
                                        loss)) ] + backwards
        #     backwards = [ rv * forwards[i] * ~forwards[i] ] + backwards
        #
        # finally update the channels
        #
        for i in range(len(self.channels)):
            self.channels[i].mat \
                -= self.eta * np.outer(backwards[i+1].arr, forwards[i].inc())
                                       



def main():
    # https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    #
    # Given matrices, with (constant) bias in the last row
    #
    mat1 = np.array([[0.15, 0.20, 0.35],
                     [0.25, 0.30, 0.35]])
    mat2 = np.array([[0.40, 0.45, 0.60],
                     [0.50, 0.55, 0.60]])
    c1 = NeuralChannel(mat1)
    c2 = NeuralChannel(mat2)
    print( c1 )
    l1 = Layer([0.05, 0.1])
    l2 = c1 >> l1
    #print( random_nchan(2,4) >> l1 )
    print("\nInitially, after one step ", l2)
    print( c2 >> l2 )
    print("\nUpdate")
    target = Layer([0.01, 0.99])
    # Outcomes of the outer update below; the first two rows are as
    # described on the webpage; but there, the last bias row is ignored.
    # [[ 0.35891648 0.51130127] 
    #  [ 0.40866619 0.56137012] 
    #  [ 0.53075072 0.61904912]]
    print( c2 << target )
    c2.update_outer(l2, target)
    # Outcomes of the hidden update below; the first two rows differ
    # slightly from the values on the webpage.
    # [[ 0.14980475  0.24977376]
    #  [ 0.1996095   0.29954752]
    #  [ 0.34609502  0.3454752 ]]
    c1.update_hidden(l1, target, c2)    


    print("\nNetwork, input")
    net = NeuralNetwork([2,6,4])
    print( net >> l1 )
    print( net.channels[1] >> (net.channels[0] >> l1) )
    print("\nNetwork, update")
    for i in range(1000):
        net.update(l1, Layer([0, 1, 0.25, 0.75]))
    print( net >> l1 )


if __name__ == "__main__":
    main()

