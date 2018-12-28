# Inspired from the https://github.com/golbin tutorial series.
# Learn the basic usage of RNN, which is widely used in natural language processing and speech processing.
# Create a model that predicts and counts numbers sequentially from 1 to 0.

import tensorflow as tf
import numpy as np


num_arr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
# Create an associative array to use one-hot encoding.
# {'1': 0, '2': 1, '3': 2, ..., '9': 9, '0', 10}
num_dic = {n: i for i, n in enumerate(num_arr)}
dic_len = len(num_dic)

# The following array will be used as input and output as follows.
# 123 -> X, 4 -> Y
# 234 -> X, 5 -> Y
seq_data = ['1234', '2345', '3456', '4567', '5678', '6789', '7890']


# In the above data, X and Y values are extracted to make one-hot encoding and batch data.
def one_hot_seq(seq_data):
    x_batch = []
    y_batch = []
    for seq in seq_data:
        # The x_data and y_data are index numbers in a 
        # numeric list, not actual numbers.
        # [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5] ...
        x_data = [num_dic[n] for n in seq[:-1]]
        # 3, 4, 5, 6...10
        y_data = num_dic[seq[-1]]
        # one-hot codification
        # if x_data is [0, 1, 2]:
        # [[ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
        #  [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
        #  [ 0.  0.  1.  0.  0.  0.  0.  0.  0.  0.]]
        x_batch.append(np.eye(dic_len)[x_data])
        # if 3: [ 0.  0.  0.  1.  0.  0.  0.  0.  0.  0.]
        y_batch.append(np.eye(dic_len)[y_data])

    return x_batch, y_batch


####################
# Hyperparameters #
###################
# Input value size. Because it is one-hot encoding for 10 numbers, it will be 10.
# If 3 => [0 0 1 0 0 0 0 0 0 0 0]
n_input = 10
# n_steps: [1 2 3] => 3
# Is the number of sequences that make up the RNN.
n_steps = 3
# The output value is also divided into 10 numbers like the input value.
n_classes = 10
# Number of characteristic values of hidden layer
n_hidden = 128


########################
# Neural network model # 
########################
X = tf.placeholder(tf.float32, [None, n_steps, n_input])
Y = tf.placeholder(tf.float32, [None, n_classes])

W = tf.Variable(tf.random_normal([n_hidden, n_classes]))
b = tf.Variable(tf.random_normal([n_classes]))

# Change the dimension configuration to use tf.nn.rnn, a function for RNN learning.
# [batch_size, n_steps, n_input]
#    -> Tensor[n_steps, batch_size, n_input]
X_t = tf.transpose(X, [1, 0, 2])
#    -> Tensor[n_steps*batch_size, n_input]
X_t = tf.reshape(X_t, [-1, n_input])
#    -> [n_steps, Tensor[batch_size, n_input]]
X_t = tf.split(X_t, n_steps, 0)

# Create an RNN cell.
# The following functions make it easy to change cells to other structures
# BasicRNNCell,BasicLSTMCell,GRUCell
cell = tf.contrib.rnn.BasicRNNCell(n_hidden)

# tf.nn.rnn Function to create a recurrent neural network.
outputs, states =  tf.contrib.rnn.static_rnn(cell, X_t, dtype=tf.float32)


# To create the loss function, reconstruct the output value into a dimension of the same type as Y
logits = tf.matmul(outputs[-1], W) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    

train_op = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(cost)


#####################
# Training Process #
####################
sess = tf.Session()
sess.run(tf.global_variables_initializer())

x_batch, y_batch = one_hot_seq(seq_data)

for epoch in range(10):
    _, loss = sess.run([train_op, cost], feed_dict={X: x_batch, Y: y_batch})

    # Try to output the change and the predicted value during learning.
    print (sess.run(tf.argmax(logits, 1), feed_dict={X: x_batch, Y: y_batch}))
    print (sess.run(tf.argmax(Y, 1), feed_dict={X: x_batch, Y: y_batch}))

    print ('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

print ('Optimized!')


################
# Check result #
################
prediction = tf.argmax(logits, 1)
prediction_check = tf.equal(prediction, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

seq_data = ['1234', '3456', '6789', '7890']
x_batch, y_batch = one_hot_seq(seq_data)

real, predict, accuracy_val = sess.run([tf.argmax(Y, 1), prediction, accuracy],
                                       feed_dict={X: x_batch, Y: y_batch})

print ("\n=== Results ===")
print ('Sequential column:', seq_data)
print ('Actual value:', [num_arr[i] for i in real])
print ('Predicted value:', [num_arr[i] for i in predict])
print ('Accuracy:', accuracy_val)

