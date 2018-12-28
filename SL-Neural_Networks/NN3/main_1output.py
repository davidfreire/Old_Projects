import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import model_one
from plot_utils import plot_decision_boundary

# Generate a dataset and plot it
np.random.seed(0)
#X, y = sklearn.datasets.make_moons(200, noise=0.20)
#plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
#plt.show()
X = np.array([[0,0,1],
	[0,1,1],
	[1,0,1],
	[1,1,1]])

y = np.array([[0,1,1,0]])


layers_dim = [X.shape[1], 4, 1]

#model = model_one.Model_one_output_layer(layers_dim)
#model.train(X, y, epochs=60000, lr=1, reg_lambda=0.0001, print_loss=True)

# Plot the decision boundary
#plot_decision_boundary(lambda x: model.predict(x), X, y)
#plt.title("Decision Boundary for hidden layer size 3")
#plt.show()