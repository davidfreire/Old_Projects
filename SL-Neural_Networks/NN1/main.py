import numpy as np


def sigmoid(x, deriv = False):
	if (deriv == True):
		return x*(1-x)
	return 1/(1+np.exp(-x))


def run():
	#Input data
	x = np.array([[0,0,1],
				  [0,1,1],
				  [1,0,1],
				  [1,1,1]])

	y = np.array([[0],
				  [1],
				  [1],
				  [0]])
	
	np.random.seed(1)

	#Build model
	num_epochs = 60000

	#initialize weights
	w0 = 2* np.random.random((3,4)) - 1 # values between 1 and -1
	w1 = 2* np.random.random((4,1)) - 1 # values between 1 and -1
	#Then, 4 hidden units and one output unit
	print(w0.shape)
	print(w1.shape)


	for i in range(0,num_epochs):

		#feedforward
		L0 = x


		L1 = sigmoid(L0.dot(w0)) #Hidden layer output
		L2 = sigmoid(L1.dot(w1)) # NN output



		#Backprop
		L2_error = y - L2
		
		if (i%10000) == 0:
			print ("Error:" + str(np.mean(np.abs(L2_error))))


		L2_delta = L2_error * sigmoid(L2, deriv=True)

		L1_error = L2_delta.dot(w1.T)

		L1_delta = L1_error * sigmoid(L1, deriv=True)

		w1 += L1.T.dot(L2_delta)
		w0 += L0.T.dot(L1_delta)






if __name__ == "__main__":
	run()
