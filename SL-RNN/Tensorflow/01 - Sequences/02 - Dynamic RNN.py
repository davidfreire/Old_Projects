# Inspired from the https://github.com/golbin tutorial series.
# Multi-layer RNNs and dynamic RNNs from the tensor flow for more 
# efficient RNN learning.

import tensorflow as tf
import numpy as np


num_arr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
num_dic = {n: i for i, n in enumerate(num_arr)}
dic_len = len(num_dic)

# Try to learn sequential columns of different sizes.
# 123 -> X, 4 -> Y
# 12 -> X, 3 -> Y
seq_data = ['1234', '2345', '3456', '4567', '5678', '6789', '7890']
seq_data2 = ['123', '234', '345', '456', '567', '678', '789', '890']


def one_hot_seq(seq_data):
    x_batch = []
    y_batch = []
    for seq in seq_data:
        x_data = [num_dic[n] for n in seq[:-1]]
        y_data = num_dic[seq[-1]]
        x_batch.append(np.eye(dic_len)[x_data])
        # The loss function sparse_softmax_cross_entropy_with_logits used in this example
        # Do not use one-hot encoding, just pass the index.
        y_batch.append([y_data])


    return x_batch, y_batch


####################
# Hyperparameters #
###################
n_input = 10
n_classes = 10
n_hidden = 128
# RNN cells in multiple layers.
n_layers = 3


########################
# Neural network model # 
########################
# To handle sequences of varying lengths, set the time steps size to None.
# [batch size, time steps, input size]
X = tf.placeholder(tf.float32, [None, None, n_input])
# Since sparse_softmax_cross_entropy_with_logits is used in the cost function, 
#the original value type for calculation with the output value is as follows.
# [batch size, time steps]
Y = tf.placeholder(tf.int32, [None, 1])

W = tf.Variable(tf.random_normal([n_hidden, n_classes]))
b = tf.Variable(tf.random_normal([n_classes]))

# In the tf.nn.dynamic_rnn option, if you set the time_major value to True, 
# you can transform the input value to a smaller value.
# [batch_size, n_steps, n_input] -> Tensor[n_steps, batch_size, n_input]
X_t = tf.transpose(X, [1, 0, 2])

# Create an RNN cell. 
# Use multi-layer and dropout techniques to avoid overfitting.
cell = tf.contrib.rnn.BasicRNNCell(n_hidden)
cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.5)
# Multi-layer configuration
cell = tf.contrib.rnn.MultiRNNCell([cell] * n_layers)

# Create a recurrent neural network using the tf.nn.dynamic_rnn function.
outputs, states = tf.nn.dynamic_rnn(cell, X_t, dtype=tf.float32, time_major=True)

# Logits uses one-hot encoding.
logits = tf.matmul(outputs[-1], W) + b
# The labels of the sparse_softmax_cross_entropy_with_logits function do not use 
# one-hot encoding, so they are passed in a one-dimensional array. (Because the time step is 1)
# (Since the rank of logits is 2 [batch_size, n_classes])
labels = tf.reshape(Y, [-1])


cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
 
train_op = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(cost)


#####################
# Training Process #
####################
sess = tf.Session()
sess.run(tf.global_variables_initializer())

x_batch, y_batch = one_hot_seq(seq_data)
x_batch2, y_batch2 = one_hot_seq(seq_data2)

for epoch in range(30):
    _, loss4 = sess.run([train_op, cost], feed_dict={X: x_batch, Y: y_batch})
    _, loss3 = sess.run([train_op, cost], feed_dict={X: x_batch2, Y: y_batch2})

    print ('Epoch:', '%04d' % (epoch + 1), 'cost =', \
        'bucket[4] =', '{:.6f}'.format(loss4), \
        'bucket[3] =', '{:.6f}'.format(loss3))

print ('Optimized!')


################
# Check result #
################
# Function to take test data and predict a result
def prediction(seq_data):
    prediction = tf.cast(tf.argmax(logits, 1), tf.int32)
    prediction_check = tf.equal(prediction, labels)
    accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

    x_batch_t, y_batch_t = one_hot_seq(seq_data)
    real, predict, accuracy_val = sess.run([labels, prediction, accuracy],
                                           feed_dict={X: x_batch_t, Y: y_batch_t})

    print ("\n=== Results ===")
    print ('Sequential column:', seq_data)
    print ('Actual value:', [num_arr[i] for i in real])
    print ('Predicted value:', [num_arr[i] for i in predict])
    print ('Accuracy:', accuracy_val)


# Test with sequences included in the training set.
seq_data_test = ['123', '345', '789']
prediction(seq_data_test)

seq_data_test = ['1234', '2345', '7890']
prediction(seq_data_test)

# Test new sequences not included in the training set.
seq_data_test = ['23', '78', '90']
prediction(seq_data_test)

seq_data_test = ['12345', '34567', '67890']
prediction(seq_data_test)
