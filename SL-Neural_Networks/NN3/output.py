import numpy as np 

class Softmax:

	def predict(self, X):
		exp_scores = np.exp(X)
		return exp_scores/np.sum(exp_scores, axis=1, keepdims=True)

	def loss (self, X, y):
		num_samples = X.shape[0]
		probs = self.predict(X)
		correct_prob_loss = -np.log(probs[range(num_samples),y])
		data_loss = np.sum(correct_prob_loss)
		return (1./num_samples) * data_loss

	def diff (self, X, y):
		num_samples = X.shape[0]
		probs = self.predict(X)
		probs[range(num_samples), y] -= 1
		return probs

