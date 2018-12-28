import numpy as np 


class NeuralNetwork(object):

	def sigmoid (self,x):
		return 1./(1.+np.exp(-x))

	def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
		
		#Set number of nodes at each layer
		self.input_nodes = input_nodes
		self.hidden_nodes = hidden_nodes
		self.output_nodes = output_nodes

		#Set learning rate
		self.lr = learning_rate

		#Set weights
		#Layer 1 - Weights from input to hidden
		self.W1 = np.random.normal(0.0, self.hidden_nodes**(-0.5), (self.input_nodes, self.hidden_nodes)).T
		#Layer 2 - Weights from hidden to output
		self.W2 = np.random.normal(0.0, self.output_nodes**(-0.5), (self.hidden_nodes, self.output_nodes)).T
		

		#Set bias
		#Bias layer 1
		self.B1 = np.zeros((self.hidden_nodes,1))
		#Bias layer 2
		self.B2 = np.zeros((self.output_nodes,1))

		#Set activation function
		self.activation_function = self.sigmoid

	def train (self, input_data, target_data):

		#Convert input list to 2d array
		inputs = np.array(input_data, ndmin=2).T 
		targets = np.array(target_data, ndmin=2).T 

		#Forward pass

		#Layer 1
		hidden_logits = self.W1.dot(inputs)+self.B1
		hidden_activations = self.activation_function(hidden_logits) 

		#Layer 2
		output_logits = self.W2.dot(hidden_activations)+self.B2
		output_activations = output_logits

		#Backward pass

		#Layer 2
		output_error = targets - output_activations
		output_delta = 1 * output_error

		#Layer 1
		hidden_error = self.W2.T.dot(output_delta)
		hidden_delta = hidden_error * (hidden_activations*(1-hidden_activations))

		#Update weights
		self.W1 += self.lr * hidden_delta.dot(inputs.T)
		self.W2 += self.lr * output_delta.dot(hidden_activations.T)
		self.B1 += self.lr * hidden_delta
		self.B2 += self.lr * output_delta

	def test (self, input_data):

		#Convert input list to 2d array
		inputs = np.array(input_data, ndmin=2).T

		#Forward pass
		hidden_logits = self.W1.dot(inputs)+self.B1
		hidden_activations = self.activation_function(hidden_logits)

		output_logits = self.W2.dot(hidden_activations)+self.B2
		output_activations = output_logits

		return output_activations






