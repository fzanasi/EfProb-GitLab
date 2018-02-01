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

def sigmoid_inv(y):
    """The sigmoid function."""
    return np.log(y/(1.0-y))

def randvar_extension(rv):
    """ add a value 1 at the end of the array of a random variable """
    return RandVar(np.array(list(rv.array) + [1]),
                   Dom(range(len(rv.dom[0]) + 1)))

class NeuralChannel:
    """ Neural channels """
    def __init__(self, matrix, eta=0.5):
        """ nxm matrix gives EfProb channel n -> m; bias is in last row """
        if len(matrix.shape) > 2:
            raise Exception('A 2x2 matrix is required in a neural channel')
        n = matrix.shape[1]
        m = matrix.shape[0]
        self.chan = Channel(matrix, Dom(range(n)), Dom(range(m)))
        #
        # Note: "neural" domain and codomain are used below, which are
        # the codomain and the domain-1 of the channel
        #
        self.ndom = m-1
        self.ncod = n
        self.eta = eta

    def __repr__(self):
        return "Neural channel from {} nodes to {} nodes".format(self.ndom, 
                                                                 self.ncod)

    def __rshift__(self, randvar):
        """ forward pass via backward randvar transformation """
        # Add an entry 1 at the end in order to handle the bias
        rv = self.chan << randvar_extension(randvar)
        # apply sigmoid elementwise, and return as randvar
        return RandVar(sigmoid(rv.array), rv.dom)

    def __lt__(self, randvar):
        """backward transition of random variable, without bias/sigmoid;
           basically this is EfProb state transformation, with the
           last (bias) row removed."""
        return RandVar(np.dot(self.chan.array, randvar.array)[:self.ndom], 
                       Dom(range(self.ndom)))

    def update_outer(self, input_rv, target_rv):
        """ Update the channel when it is used as output """
        output_rv = self >> input_rv
        out = (output_rv - target_rv) & output_rv & ~output_rv
        input_rv = randvar_extension(input_rv)
        #print( np.outer(input_rv.array, out.array) )
        self.chan.array -= self.eta * np.outer(input_rv.array, out.array)
        #print( self.chan.array )

    def update_hidden(self, input_rv, target_rv, next_nchan):
        """ Update the channel when it is used in hidden form """
        output = self >> input_rv
        next_output = next_nchan >> output
        next_diff = (next_output - target_rv) & next_output & ~next_output
        error = next_nchan < next_diff
        out = error & output & ~output
        input_rv = randvar_extension(input_rv)
        #print( np.outer(input_rv.array, out.array) )
        self.chan.array -= self.eta * np.outer(input_rv.array, out.array)
        #print( self.chan.array )




def random_nchan(n, m):
    """ random neural channel from n nodes to m nodes """
    mat = np.random.randn(n+1, m)
    print( mat )
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

    def __rshift__(self, randvar):
        """ forward pass of a random variable through the network """
        for i in range(len(self.channels)):
            randvar = self.channels[i] >> randvar
        return randvar

    def update(self, in_rv, target):
        #
        # Apply forward computations iteratively, giving a list of
        # random variables (without biases)
        #
        forwards = [in_rv]
        for i in range(len(self.channels)):
            rv = forwards[len(forwards) - 1]
            forwards.append(self.channels[i] >> rv)
        #
        # next compute backwards the iterated "errors"
        #
        current_rv = forwards[len(forwards) - 1]
        backwards = [ (current_rv - target) & current_rv & ~current_rv ]
        for i in range(len(self.channels)-1, -1, -1):
            rv = self.channels[i] < backwards[0]
            backwards = [ rv & forwards[i] & ~forwards[i] ] + backwards
        #
        # finally update the channels
        #
        for i in range(len(self.channels)):
            self.channels[i].chan.array \
                -= self.eta * np.outer(randvar_extension(forwards[i]).array,
                                       backwards[i+1].array)



def main():
    # https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    #
    # Given matrices, with (constant) bias in the last row
    #
    mat1 = np.array([[0.15, 0.25],
                     [0.20, 0.30],
                     [0.35, 0.35]])
    mat2 = np.array([[0.40, 0.50],
                     [0.45, 0.55],
                     [0.60, 0.60]])
    c1 = NeuralChannel(mat1)
    c2 = NeuralChannel(mat2)
    print( c1 )
    inp1 = randvar_fromlist([0.05, 0.1])
    print( random_nchan(2,4) >> inp1 )
    inp2 = c1 >> inp1
    print("\nInitially, after one step ", inp2)
    print( c2 >> inp2 )
    print("\nUpdate")
    target = randvar_fromlist([0.01, 0.99])
    # Outcomes of the outer update below; the first two rows are as
    # described on the webpage; but there, the last bias row is ignored.
    # [[ 0.35891648 0.51130127] 
    #  [ 0.40866619 0.56137012] 
    #  [ 0.53075072 0.61904912]]
    c2.update_outer(inp2, target)
    # Outcomes of the hidden update below; the first two rows differ
    # slightly from the values on the webpage.
    # [[ 0.14980475  0.24977376]
    #  [ 0.1996095   0.29954752]
    #  [ 0.34609502  0.3454752 ]]
    c1.update_hidden(inp1, target, c2)    


    print("\nNetwork, input")
    net = NeuralNetwork([2,6,4])
    print( net >> inp1 )
    print( net.channels[1] >> (net.channels[0] >> inp1) )
    print("\nNetwork, update")
    for i in range(100):
        net.update(inp1, randvar_fromlist([0,1,0.25,0.75]))
    print( net >> inp1 )


if __name__ == "__main__":
    main()

