# Inspired from the https://github.com/golbin tutorial series.
# Implement Seq2Seq, a sequence learning / generation model used for 
#chatbot, translation, and image captioning.

import tensorflow as tf
import numpy as np


# S: Symbol indicating the start of decoding input
# E: Symbol indicating the end of decoding output
# P: A symbol that fills an empty sequence if the size of the current 
#    batch data is smaller than the time step size eg) If the maximum 
#    size of the current batch data is 4
#       1234 -> [1, 2, 3, 4]
#       12   -> [1, 2, P, P]
num_arr = ['S', 'E', 'P', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
num_dic = {n: i for i, n in enumerate(num_arr)}
dic_len = len(num_dic)

# Divide input value and output value to input dynamic value
# 12 -> X, 34 -> Y
# 123 -> X, 456 -> Y
seq_data = [['12', '34'], ['23', '45'], ['34', '56'], ['45', '67'], ['56', '78'], ['67', '89'], ['78', '90']]
seq_data2 = [['123', '456'], ['234', '567'], ['345', '678'], ['456', '789'], ['567', '890']]


def one_hot_seq(seq_data):
    x_batch = []
    y_batch = []
    target_batch = []
    for seq in seq_data:
        # Prefix P with the input and output values to equal the time step. (Not required)
        x_data = [num_dic[n] for n in ('P' + seq[0])]
        # The input value of the decoder cell. Prefix the S symbol to indicate start.
        y_data = [num_dic[n] for n in ('S' + seq[1])]
        # Output value to compare for learning. Attach E to the end to let it know that it is over.
        target_data = [num_dic[n] for n in (seq[1] + 'E')]

        x_batch.append(np.eye(dic_len)[x_data])
        y_batch.append(np.eye(dic_len)[y_data])
        # Output value is not one-hot encoding (sparse_softmax_cross_entropy_with_logits)
        target_batch.append(target_data)

    return x_batch, y_batch, target_batch


####################
# Hyperparameters #
###################
# The sizes of input and output are the same because they are the same in one-hot encoding.
n_classes = n_input = dic_len
n_hidden = 128
n_layers = 3


########################
# Neural network model # 
########################
# The Seq2Seq model has the same format as the input of the encoder and the input of the decoder.
# [batch size, time steps, input size]
enc_input = tf.placeholder(tf.float32, [None, None, n_input])
dec_input = tf.placeholder(tf.float32, [None, None, n_input])
# [batch size, time steps]
targets = tf.placeholder(tf.int64, [None, None])

W = tf.Variable(tf.ones([n_hidden, n_classes]))
b = tf.Variable(tf.zeros([n_classes]))

# Set the time_major value to True in the tf.nn.dynamic_rnn option
# [batch_size, n_steps, n_input] -> Tensor[n_steps, batch_size, n_input]
enc_input = tf.transpose(enc_input, [1, 0, 2])
dec_input = tf.transpose(dec_input, [1, 0, 2])

# Encode cell configuration
with tf.variable_scope('encode'):
    enc_cell = tf.contrib.rnn.BasicRNNCell(n_hidden)
    enc_cell = tf.contrib.rnn.DropoutWrapper(enc_cell, output_keep_prob=0.5)
    enc_cell = tf.contrib.rnn.MultiRNNCell([enc_cell] * n_layers)

    outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_input,
                                            dtype=tf.float32)

# Decode cell configuration
with tf.variable_scope('decode'):
    dec_cell = tf.contrib.rnn.BasicRNNCell(n_hidden)
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, output_keep_prob=0.5)
    dec_cell = tf.contrib.rnn.MultiRNNCell([dec_cell] * n_layers)

    # In the Seq2Seq model, it is important to put the final state value 
    # of the encoder cell as the initial state value of the decoder cell.
    outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_input,
                                            initial_state=enc_states,
                                            dtype=tf.float32)


# sparse_softmax_cross_entropy_with_logits. In order to use the function, 
# the dimensions of each tensor are calculated by transforming as follows.
#    -> [batch size, time steps, hidden layers]
time_steps = tf.shape(outputs)[1]
#    -> [batch size * time steps, hidden layers]
outputs_trans = tf.reshape(outputs, [-1, n_hidden])
#    -> [batch size * time steps, class numbers]
logits = tf.matmul(outputs_trans, W) + b
#    -> [batch size, time steps, class numbers]
logits = tf.reshape(logits, [-1, time_steps, n_classes])


cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets))
train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)


########################
# Neural network model # 
########################
sess = tf.Session()
sess.run(tf.global_variables_initializer())

x_batch, y_batch, target_batch = one_hot_seq(seq_data)
x_batch2, y_batch2, target_batch2 = one_hot_seq(seq_data2)

for epoch in range(100):
    _, loss4 = sess.run([train_op, cost],
                        feed_dict={enc_input: x_batch, dec_input: y_batch, targets: target_batch})
    _, loss3 = sess.run([train_op, cost],
                        feed_dict={enc_input: x_batch2, dec_input: y_batch2, targets: target_batch2})

    print ('Epoch:', '%04d' % (epoch + 1), 'cost =', \
        'bucket[4] =', '{:.6f}'.format(loss4), \
        'bucket[3] =', '{:.6f}'.format(loss3))

print ('Optimized!')


################
# Check result #
################
# Function to take test data and predict a result
def prediction_test(seq_data):
    prediction = tf.argmax(logits, 2)
    prediction_check = tf.equal(prediction, targets)
    accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

    x_batch_t, y_batch_t, target_batch_t = one_hot_seq(seq_data)
    real, predict, accuracy_val = sess.run([targets, prediction, accuracy],
                                           feed_dict={enc_input: x_batch_t,
                                                      dec_input: y_batch_t,
                                                      targets: target_batch_t})

    print ("\n=== Results ===")
    print ('Sequential column:', seq_data)
    print ('Actual value:', [[num_arr[j] for j in dec] for dec in real])
    print ('Predicted value:', [[num_arr[i] for i in dec] for dec in predict])
    print ('Accuracy:', accuracy_val)



# Test with sequences included in the training set.
prediction_test(seq_data)
prediction_test(seq_data2)

seq_data_test = [['12', '34'], ['23', '45'], ['78', '90']]
prediction_test(seq_data_test)

seq_data_test = [['123', '456'], ['345', '678'], ['567', '890']]
prediction_test(seq_data_test)


#########
# Let's predict the next sequence by input.
######
# Function that take sequence data and predict and decode the following results
def decode(seq_data):
    prediction = tf.argmax(logits, 2)
    x_batch_t, y_batch_t, target_batch_t = one_hot_seq([seq_data])

    result = sess.run(prediction,
                      feed_dict={enc_input: x_batch_t,
                                 dec_input: y_batch_t,
                                 targets: target_batch_t})

    decode_seq = [[num_arr[i] for i in dec] for dec in result][0]

    return decode_seq


# A function that takes the sequence data and predicts the next one, 
# and makes progressive prediction until the end symbol E is reached.
def decode_loop(seq_data):
    decode_seq = ''
    current_seq = ''

    while current_seq != 'E':
        decode_seq = decode(seq_data)
        seq_data = [seq_data[0], ''.join(decode_seq)]
        current_seq = decode_seq[-1]

    return decode_seq


print ("\n=== Predict sequential progression by one character ===")

seq_data = ['123', '']
print ("123 ->", decode_loop(seq_data))

seq_data = ['67', '']
print ("67 ->", decode_loop(seq_data))

seq_data = ['3456', '']
print ("3456 ->", decode_loop(seq_data))

print ("\n=== Predict entire sequences at once ===")

seq_data = ['123', 'PPP']
print ("123 ->", decode(seq_data))

seq_data = ['67', 'PP']
print ("67 ->", decode(seq_data))

seq_data = ['3456', 'PPPP']
print ("3456 ->", decode(seq_data))
