from numpy import exp, array, random, dot

class NeuralNetwork:
    
    def __init__(self):
        random.seed(1)
        self.synaptic_weights=random.rand(3,1)*2-1
    
    def sigmoid(self,x):
        return 1/(1+exp(-x))
    
    def sigmoid_derivative(self,x):
        return x*(1-x)
    
    def train(self,training_set_inputs,training_set_outputs,no_of_iterations):
        
        for iterations in range(no_of_iterations):
            
            output=self.sigmoid(dot(training_set_inputs,self.synaptic_weights))
            
            error=(training_set_outputs-output)
            
            adjustment=dot(training_set_inputs.T,error*self.sigmoid_derivative(output))
            
            self.synaptic_weights+=adjustment


if __name__ == "__main__":

    #Intialise a single neuron neural network.
    neural_network = NeuralNetwork()

    print ("Random starting synaptic weights: ")
    print( neural_network.synaptic_weights)

    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs,10000)

    print ("New synaptic weights after training: ")
    print (neural_network.synaptic_weights)

    # Test the neural network with a new situation.
  
