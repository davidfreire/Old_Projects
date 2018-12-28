import numpy as np 


class MultiplyGate:

	def forward(self, X, W):
		return X.dot(W)

	def backward(self, X, W, delta):
		v = np.atleast_2d(X) #if X is only one sample, it switchs from (2,) to (2,1)
		dW = np.dot(np.transpose(v), delta)
		dX = np.dot(delta, np.transpose(W))
		return dW, dX






class AddGate:

	def forward(self, X, b):
		return X+b

	def backward (self, X, b, delta):
		dX = delta * np.ones_like(X)
		db = np.ones((1,delta.shape[0]), dtype=np.float64).dot(delta)
		return (db, dX)
