import numpy as np 


class Sigmoid:
    def forward(self, X):
        return 1.0 / (1.0 + np.exp(-X))

    def backward(self, X, err):
        output = self.forward(X)
        return (1.0 - output) * output * err





class Tanh:

	def forward(self, X):
		return np.tanh(X)

	def backward(self, X, err):
		output = self.forward(X)
		return (1.0-np.square(output))*err
