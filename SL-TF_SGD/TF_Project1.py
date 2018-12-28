from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import math

epochs = 80
learning_rate = 0.001
n_input = 784  # MNIST data input (img shape: 28*28 = 784)
n_classes = 10  # MNIST total classes (0-9 digits)


def print_epoch_stats(cost, features, labels, accuracy, valid_features, valid_labels, epoch_i, sess, last_features, last_labels):
    """
    Print cost and validation accuracy of an epoch
    """
    current_cost = sess.run(
        cost,
        feed_dict={features: last_features, labels: last_labels})
    valid_accuracy = sess.run(
        accuracy,
        feed_dict={features: valid_features, labels: valid_labels})
    print('Epoch: {:<4} - Cost: {:<8.3} Valid Accuracy: {:<5.3}'.format(
        epoch_i,
        current_cost,
        valid_accuracy))

def batches(batch_size, features, labels):
    """
    Creates batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Features, Labels)
    """
    assert len(features) == len(labels)
    output_batches = []
    
    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch = [features[start_i:end_i], labels[start_i:end_i]]
        output_batches.append(batch)
        
    return output_batches

def run():
	# Import MNIST data
	mnist = input_data.read_data_sets('mnist/', one_hot=True)

	# The features are already scaled and the data is shuffled
	train_features = mnist.train.images
	valid_features = mnist.validation.images
	test_features = mnist.test.images

	train_labels = mnist.train.labels.astype(np.float32)
	valid_labels = mnist.validation.labels.astype(np.float32)
	test_labels = mnist.test.labels.astype(np.float32)

	# Features and Labels
	# Continuing the example, if each sample had n_input = 784 features and n_classes = 10 
	#possible labels, the dimensions for features would be [None, n_input] and labels would be [None, n_classes].
	features = tf.placeholder(tf.float32, [None, n_input])
	labels = tf.placeholder(tf.float32, [None, n_classes])
	#The None dimension is a placeholder for the batch size. At runtime, TensorFlow will accept 
	#any batch size greater than 0.



	# Weights & bias
	weights = tf.Variable(tf.random_normal([n_input, n_classes]))
	bias = tf.Variable(tf.random_normal([n_classes]))

	# Logits - xW + b
	logits = tf.add(tf.matmul(features, weights), bias)

	# Define loss and optimizer
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

	# Calculate accuracy
	correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


	#Set batch size
	batch_size = 128
	assert batch_size is not None, 'You must set the batch size'

	input_batches = batches(batch_size, train_features, train_labels)

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
	    sess.run(init)
	    		    

	    #Training cycle
	    for epoch_i in range(epochs):
		    #Train optimizer on all batches
	    	for batch_features, batch_labels in input_batches:
	        	sess.run(optimizer, feed_dict={features: batch_features, labels: batch_labels})

	    	# Calculate accuracy for test dataset
	    	test_accuracy = sess.run(
	        	accuracy,
	        	feed_dict={features: test_features, labels: test_labels})
	    	print_epoch_stats(cost, features, labels, accuracy, valid_features, valid_labels, epoch_i, sess, batch_features, batch_labels)
	
	print('Test Accuracy: {}'.format(test_accuracy))





if __name__ == '__main__':
	run()

