
#Code based on https://github.com/llSourcell/How-to-Use-Tensorflow-for-Time-Series-Live-
import numpy as np 
import tensorflow as tf 


def generateData(total_series_length, echo_step, batch_size):
	#0,1 50K samples
	x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5])) #2 classes, length of the series, and the probabilitie of each number generated is 0 or 1 (50-50)
	#shift 3 step to the left (goes from [100011] to [011100], three elements (100) goes from the front to the end of the array)
	y = np.roll(x,echo_step)
	#Padd the beginning with zeros (goes from [011100] to [000100])
	y[0:echo_step] = 0
	#Reshape the data for our machine into rows of size batch_size
	x = x.reshape((batch_size, -1))
	y = y.reshape((batch_size, -1))
	
	return (x,y)








def run():

	#define hyperparameters
	epochs = 100
	total_series_length = 50000
	truncated_backprop_length = 15 #Backprop is truncated so gradient dont evanish o explode.
	state_size = 4 #number of neurons in the hidden layer
	num_classes = 2 
	echo_step = 3
	batch_size = 5 
	learning_rate = 0.3

	num_batches = total_series_length // batch_size #Double // generates integer results
	num_batches //= truncated_backprop_length

	data = generateData(total_series_length, echo_step, batch_size)


	#Build the model

	#datatype, shape (5, 15) 2D array or matrix, batch size shape for later
	batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
	batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])
	#and one for the RNN state, (5,4) 
	init_state = tf.placeholder(tf.float32, [batch_size, state_size]) 
	#Weights and biases from input to hidden layer
	W_1 = tf.Variable(np.random.rand(state_size+1, state_size), dtype = tf.float32)
	b_1 = tf.Variable(np.zeros((1, state_size)), dtype=tf.float32)
	#Weights and biases from hidden to output layer
	W_2 = tf.Variable(np.random.rand(state_size, num_classes), dtype = tf.float32)
	b_2 = tf.Variable(np.zeros((1, num_classes)), dtype = tf.float32)

	#unpack matrix into 1D array
	inputs_series = tf.unstack(batchX_placeholder, axis=1)
	labels_series = tf.unstack(batchY_placeholder, axis=1)

	#Forward pass
	#state placeholder
	current_state = init_state
	#series of states through time
	state_series = []

	#for each set of inputs
	#forward pass through the network to get new state value
	#store all states in memory
	for current_input in inputs_series:
		#format input
		current_input = tf.reshape(current_input, [batch_size, 1]) 
		#mix both state and input data 
		input_and_state_concatenated = tf.concat([current_input, current_state], 1) # Increasing number of columns
		#perform matrix multiplication between weights and input, add bias
		#squash with a nonlinearity, for probability value
		next_state = tf.tanh(tf.add(tf.matmul(input_and_state_concatenated, W_1),b_1))
		#store the state in memory
		state_series.append(next_state)
		#set current state to next one
		current_state = next_state 

	#Calculate loss and minimize it
	#for state in states_series:
	logits_series = [tf.add(tf.matmul(state,W_2), b_2) for state in state_series]
	prediction_series = [tf.nn.softmax(logits) for logits in logits_series]

	losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series,labels_series)]
	total_loss = tf.reduce_mean(losses)


	train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)



	init = tf.global_variables_initializer()
	with tf.Session() as sess:

		sess.run(init)

		loss_list = []
		for epoch in range(epochs):
			x,y = generateData(total_series_length, echo_step, batch_size)

			#init hidden state
			_current_state = np.zeros((batch_size, state_size))

			print("New data, epoch ", epoch)

			#each batch
			for batch_idx in range(num_batches):
				#starting and ending point per batch
				#since weights reoccuer at every layer through time
				#These layers will not be unrolled to the beginning of time,
				#that would be too computationally expensive, and are therefore truncated 
				#at a limited number of time-steps
				start_idx = batch_idx * truncated_backprop_length
				end_idx = start_idx + truncated_backprop_length

				batchX = x[:,start_idx:end_idx]
				batchY = y[:,start_idx:end_idx]

				#run the computation graph, give it the values
				#we calculated earlier
				_total_loss, _train_step, _current_state, _predictions_series = sess.run(
	            	[total_loss, train_step, current_state, prediction_series],
	            	feed_dict={
	            	batchX_placeholder:batchX,
	            	batchY_placeholder:batchY,
	            	init_state:_current_state
	            	})

				loss_list.append(_total_loss)

				if batch_idx%100 == 0:
					print("Step",batch_idx, "Loss", _total_loss)














if __name__ == '__main__':
	run()