import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os
import random
import numpy as np
import random
import requests
from tensorflow.python.framework import ops

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# sample = pd.read_csv('sample.csv', header=None)

train.head()

y_train = np.array(train['y'])


train['year'] = train['datetime'].apply(lambda x: x.split('-')[0])
train['month'] = train['datetime'].apply(lambda x: x.split('-')[1])
cols_of_interest = ['year','month','soldout','kcal','temperature']
x_train = np.array(train[cols_of_interest])


# reset the graph for new run
ops.reset_default_graph()

# Create graph session
sess = tf.Session()

# set batch size for training
batch_size = 100

# make results reproducible
seed = 3
np.random.seed(seed)
tf.set_random_seed(seed)


# Normalize by column (min-max norm to be between 0 and 1)
def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min) / (col_max - col_min)

x_train = np.nan_to_num(normalize_cols(x_train))
normalize_cols(np.nan_to_num(x_train))
np.nan_to_num(x_train)
# x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

print('ii')

# # Define Variable Functions (weights and bias)
# def init_weight(shape, st_dev):
#     weight = tf.Variable(tf.random_normal(shape, stddev=st_dev))
#     return(weight)
#
#
# def init_bias(shape, st_dev):
#     bias = tf.Variable(tf.random_normal(shape, stddev=st_dev))
#     return(bias)
#
#
# # Create Placeholders
# x_data = tf.placeholder(shape=[None, 395], dtype=tf.float32)
# y_target = tf.placeholder(shape=[None, 395], dtype=tf.float32)
#
#
# # Create a fully connected layer:
# def fully_connected(input_layer, weights, biases):
#     layer = tf.add(tf.matmul(input_layer, weights), biases)
#     return(tf.nn.relu(layer))
#
#
# #--------Create the first layer (50 hidden nodes)--------
# weight_1 = init_weight(shape=[7, 25], st_dev=10.0)
# bias_1 = init_bias(shape=[25], st_dev=10.0)
# layer_1 = fully_connected(x_data, weight_1, bias_1)
#
# #--------Create second layer (25 hidden nodes)--------
# weight_2 = init_weight(shape=[25, 10], st_dev=10.0)
# bias_2 = init_bias(shape=[10], st_dev=10.0)
# layer_2 = fully_connected(layer_1, weight_2, bias_2)
#
#
# #--------Create third layer (5 hidden nodes)--------
# weight_3 = init_weight(shape=[10, 3], st_dev=10.0)
# bias_3 = init_bias(shape=[3], st_dev=10.0)
# layer_3 = fully_connected(layer_2, weight_3, bias_3)
#
#
# #--------Create output layer (1 output value)--------
# weight_4 = init_weight(shape=[3, 1], st_dev=10.0)
# bias_4 = init_bias(shape=[1], st_dev=10.0)
# final_output = fully_connected(layer_3, weight_4, bias_4)
#
# # Declare loss function (L1)
# loss = tf.reduce_mean(tf.abs(y_target - final_output))
#
# # Declare optimizer
# my_opt = tf.train.AdamOptimizer(0.05)
# train_step = my_opt.minimize(loss)
#
# # Initialize Variables
# init = tf.global_variables_initializer()
# sess.run(init)
#
# # Training loop
# loss_vec = []
# test_loss = []
# for i in range(200):
#     rand_index = np.random.choice(len(x_train), size=batch_size)
#     rand_x = x_train[rand_index]
#     rand_y = np.transpose([y_train[rand_index]])
#     sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
#
#     temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
#     loss_vec.append(temp_loss)
#
#     # test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
#     # test_loss.append(test_temp_loss)
#     if (i+1) % 25 == 0:
#         print('Generation: ' + str(i+1) + '. Loss = ' + str(temp_loss))
#
#
# # # Plot loss (MSE) over time
# # plt.plot(loss_vec, 'k-', label='Train Loss')
# # plt.plot(test_loss, 'r--', label='Test Loss')
# # plt.title('Loss (MSE) per Generation')
# # plt.legend(loc='upper right')
# # plt.xlabel('Generation')
# # plt.ylabel('Loss')
# # plt.show()
