import numpy as np
from gate import MultiplyGate, AddGate
from activation import Tanh, Sigmoid
from output import Softmax

class Model:

	#Layers_dim is related to the layers dimensions, i.e. [2, 3, 2] which stands for a 2d input layer, 3 hidden neurons and a 2d output (2 neurons)
	def __init__(self, layers_dim): 
		self.b = []
		self.W = []

		self.mulGate = MultiplyGate()
		self.addGate = AddGate()
		self.activation = Tanh()#Sigmoid()
		self.output = Softmax()

		for i in range(len(layers_dim)-1):
			self.W.append(np.random.randn(layers_dim[i],layers_dim[i+1])/np.sqrt(layers_dim[i]))
			self.b.append(np.random.randn(layers_dim[i+1]).reshape(1,layers_dim[i+1]))


	def compute_loss(self, X, y):

		input_data = X

		for i in range(len(self.W)):
			logits_mul = self.mulGate.forward(input_data, self.W[i])
			logits_add = self.addGate.forward(logits_mul, self.b[i])
			activ = self.activation.forward(logits_add)
			input_data = activ #the output of the layer is the input of the next layer

		data_loss = self.output.loss(input_data,y)
		return data_loss



	def predict(self,X):

		input_data = X

		for i in range(len(self.W)):
			logits_mul = self.mulGate.forward(input_data, self.W[i])
			logits_add = self.addGate.forward(logits_mul, self.b[i])
			activ = self.activation.forward(logits_add)
			input_data = activ

		probs = self.output.predict(input_data)
		return np.argmax(probs, axis=1)

	def train(self, X, y, epochs=20000, lr=0.01, reg_lambda=0.01, print_loss=False):

		for epoch in range(epochs):
		# Forward propagation
			input = X
			forward = [(None, None, input)]
			for i in range(len(self.W)):
				#print(i)
				mul = self.mulGate.forward(input, self.W[i])
				add = self.addGate.forward(mul, self.b[i])
				#print(add)
				input = self.activation.forward(add)
				forward.append((mul, add, input))

			#for i in range(len(forward)):
				#print(forward[i])
				#print('\n')

			# Back propagation
			#print('bakc')
			#print(forward)
			dtanh = self.output.diff(forward[len(forward)-1][2], y)
			#print(dtanh)
			#print(forward[len(forward)-1][2])
			#print(y)
			#print('ooss')
			for i in range(len(forward)-1, 0, -1):
				dadd = self.activation.backward(forward[i][1], dtanh)
				db, dmul = self.addGate.backward(forward[i][0], self.b[i-1], dadd)
				dW, dtanh = self.mulGate.backward(forward[i-1][2], self.W[i-1], dmul)
				# Add regularization terms (b1 and b2 don't have regularization terms)
				dW += reg_lambda * self.W[i-1]
				# Gradient descent parameter update
				self.b[i-1] += -lr * db
				self.W[i-1] += -lr * dW

			if print_loss and epoch % 1000 == 0:
				print("Loss after iteration %i: %f" %(epoch, self.compute_loss(X, y)))








