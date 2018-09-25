# import matplotlib.pyplot as plt
# import numpy as np
# import tensorflow as tf
# from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
# from tensorflow.python.framework import ops
# ops.reset_default_graph()
#
# # Start a graph session
# sess = tf.Session()
#
# # Load data
# data_dir = 'temp'
# mnist = read_data_sets(data_dir)
#
# # 画像を28*28配列に変換
# train_xdata = np.array([np.reshape(x, (28,28)) for x in mnist.train.images])
# test_xdata = np.array([np.reshape(x, (28,28)) for x in mnist.test.images])
#
# # ラベルをone-hotエンコーディング表現のベクトルに変換
# train_labels = mnist.train.labels
# test_labels = mnist.test.labels
#
# # パラメータのセット
# batch_size = 100
# learning_rate = 0.005
# evaluation_size = 500
# image_width = train_xdata[0].shape[0]
# image_height = train_xdata[0].shape[1]
# target_size = max(train_labels) + 1
# num_channels = 1 # greyscale = 1 channel
# generations = 500
# eval_every = 5
# conv1_features = 25
# conv2_features = 50
# max_pool_size1 = 2 # NxN window for 1st max pool layer
# max_pool_size2 = 2 # NxN window for 2nd max pool layer
# fully_connected_size1 = 100
#
# # Declare model placeholders
# x_input_shape = (batch_size, image_width, image_height, num_channels)
# x_input = tf.placeholder(tf.float32, shape=x_input_shape)
# y_target = tf.placeholder(tf.int32, shape=(batch_size))
# eval_input_shape = (evaluation_size, image_width, image_height, num_channels)
# eval_input = tf.placeholder(tf.float32, shape=eval_input_shape)
# eval_target = tf.placeholder(tf.int32, shape=(evaluation_size))
#
# # 畳み込み層の変数
# conv1_weight = tf.Variable(tf.truncated_normal([4, 4, num_channels, conv1_features],stddev=0.1, dtype=tf.float32))
# conv1_bias = tf.Variable(tf.zeros([conv1_features], dtype=tf.float32))
#
# conv2_weight = tf.Variable(tf.truncated_normal([4, 4, conv1_features, conv2_features],stddev=0.1, dtype=tf.float32))
# conv2_bias = tf.Variable(tf.zeros([conv2_features], dtype=tf.float32))
#
# # 全結合相の変数
# resulting_width = image_width // (max_pool_size1 * max_pool_size2)
# resulting_height = image_height // (max_pool_size1 * max_pool_size2)
# full1_input_size = resulting_width * resulting_height * conv2_features
# full1_weight = tf.Variable(tf.truncated_normal([full1_input_size, fully_connected_size1],stddev=0.1, dtype=tf.float32))
# full1_bias = tf.Variable(tf.truncated_normal([fully_connected_size1], stddev=0.1, dtype=tf.float32))
# full2_weight = tf.Variable(tf.truncated_normal([fully_connected_size1, target_size],stddev=0.1, dtype=tf.float32))
# full2_bias = tf.Variable(tf.truncated_normal([target_size], stddev=0.1, dtype=tf.float32))
#
#
# # Initialize Model Operations
# def my_conv_net(input_data):
#     # First Conv-ReLU-MaxPool Layer
#     conv1 = tf.nn.conv2d(input_data, conv1_weight, strides=[1, 1, 1, 1], padding='SAME')
#     relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
#     max_pool1 = tf.nn.max_pool(relu1, ksize=[1, max_pool_size1, max_pool_size1, 1],
#                                strides=[1, max_pool_size1, max_pool_size1, 1], padding='SAME')
#
#     # Second Conv-ReLU-MaxPool Layer
#     conv2 = tf.nn.conv2d(max_pool1, conv2_weight, strides=[1, 1, 1, 1], padding='SAME')
#     relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))
#     max_pool2 = tf.nn.max_pool(relu2, ksize=[1, max_pool_size2, max_pool_size2, 1],
#                                strides=[1, max_pool_size2, max_pool_size2, 1], padding='SAME')
#
#     # Transform Output into a 1xN layer for next fully connected layer
#     final_conv_shape = max_pool2.get_shape().as_list()
#     final_shape = final_conv_shape[1] * final_conv_shape[2] * final_conv_shape[3]
#     flat_output = tf.reshape(max_pool2, [final_conv_shape[0], final_shape])
#
#     # First Fully Connected Layer
#     fully_connected1 = tf.nn.relu(tf.add(tf.matmul(flat_output, full1_weight), full1_bias))
#
#     # Second Fully Connected Layer
#     final_model_output = tf.add(tf.matmul(fully_connected1, full2_weight), full2_bias)
#
#     return(final_model_output)
#
# model_output = my_conv_net(x_input)
# test_model_output = my_conv_net(eval_input)
#
# # Declare Loss Function (softmax cross entropy)
# loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_output, labels=y_target))
#
# # Create a prediction function
# prediction = tf.nn.softmax(model_output)
# test_prediction = tf.nn.softmax(test_model_output)
#
# # Create accuracy function
# def get_accuracy(logits, targets):
#     batch_predictions = np.argmax(logits, axis=1)
#     num_correct = np.sum(np.equal(batch_predictions, targets))
#     return(100. * num_correct/batch_predictions.shape[0])
#
# # Create an optimizer
# my_optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
# train_step = my_optimizer.minimize(loss)
#
# # Initialize Variables
# init = tf.global_variables_initializer()
# sess.run(init)
#
# # Start training loop
# train_loss = []
# train_acc = []
# test_acc = []
# for i in range(generations):
#     rand_index = np.random.choice(len(train_xdata), size=batch_size)
#     rand_x = train_xdata[rand_index]
#     rand_x = np.expand_dims(rand_x, 3)
#     rand_y = train_labels[rand_index]
#     train_dict = {x_input: rand_x, y_target: rand_y}
#
#     sess.run(train_step, feed_dict=train_dict)
#     temp_train_loss, temp_train_preds = sess.run([loss, prediction], feed_dict=train_dict)
#     temp_train_acc = get_accuracy(temp_train_preds, rand_y)
#
#     if (i+1) % eval_every == 0:
#         eval_index = np.random.choice(len(test_xdata), size=evaluation_size)
#         eval_x = test_xdata[eval_index]
#         eval_x = np.expand_dims(eval_x, 3)
#         eval_y = test_labels[eval_index]
#         test_dict = {eval_input: eval_x, eval_target: eval_y}
#         test_preds = sess.run(test_prediction, feed_dict=test_dict)
#         temp_test_acc = get_accuracy(test_preds, eval_y)
#
#         # Record and print results
#         train_loss.append(temp_train_loss)
#         train_acc.append(temp_train_acc)
#         test_acc.append(temp_test_acc)
#         acc_and_loss = [(i+1), temp_train_loss, temp_train_acc, temp_test_acc]
#         acc_and_loss = [np.round(x,2) for x in acc_and_loss]
#         print('Generation # {}. Train Loss: {:.2f}. Train Acc (Test Acc): {:.2f} ({:.2f})'.format(*acc_and_loss))
#
#
# # Matlotlib code to plot the loss and accuracies
# eval_indices = range(0, generations, eval_every)
# # Plot loss over time
# plt.plot(eval_indices, train_loss, 'k-')
# plt.title('Softmax Loss per Generation')
# plt.xlabel('Generation')
# plt.ylabel('Softmax Loss')
# plt.show()
#
# # Plot train and test accuracy
# plt.plot(eval_indices, train_acc, 'k-', label='Train Set Accuracy')
# plt.plot(eval_indices, test_acc, 'r--', label='Test Set Accuracy')
# plt.title('Train and Test Accuracy')
# plt.xlabel('Generation')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
# plt.show()
#
# # Plot some samples
# # Plot the 6 of the last batch results:
# actuals = rand_y[0:6]
# predictions = np.argmax(temp_train_preds,axis=1)[0:6]
# images = np.squeeze(rand_x[0:6])
#
# Nrows = 2
# Ncols = 3
# for i in range(6):
#     plt.subplot(Nrows, Ncols, i+1)
#     plt.imshow(np.reshape(images[i], [28,28]), cmap='Greys_r')
#     plt.title('Actual: ' + str(actuals[i]) + ' Pred: ' + str(predictions[i]),
#                                fontsize=10)
#     frame = plt.gca()
#     frame.axes.get_xaxis().set_visible(False)
#     frame.axes.get_yaxis().set_visible(False)



# # More Advanced CNN Model: CIFAR-10
# #---------------------------------------
# #
# # In this example, we will download the CIFAR-10 images
# # and build a CNN model with dropout and regularization
# #
# # CIFAR is composed ot 50k train and 10k test
# # images that are 32x32.
#
# import os
# import sys
# import tarfile
# import matplotlib.pyplot as plt
# import numpy as np
# import tensorflow as tf
# from six.moves import urllib
# from tensorflow.python.framework import ops
# ops.reset_default_graph()
#
# # Change Directory
# abspath = os.path.abspath(__file__)
# dname = os.path.dirname(abspath)
# os.chdir(dname)
#
# # Start a graph session
# sess = tf.Session()
#
# # Set model parameters
# batch_size = 128
# data_dir = 'temp'
# output_every = 50
# generations = 20000
# eval_every = 500
# image_height = 32
# image_width = 32
# crop_height = 24
# crop_width = 24
# num_channels = 3
# num_targets = 10
# extract_folder = 'cifar-10-batches-bin'
#
# # Exponential Learning Rate Decay Params
# learning_rate = 0.1
# lr_decay = 0.1
# num_gens_to_wait = 250.
#
# # Extract model parameters
# image_vec_length = image_height * image_width * num_channels
# record_length = 1 + image_vec_length # ( + 1 for the 0-9 label)
#
#
# # Load data
# data_dir = 'temp'
# if not os.path.exists(data_dir):
#     os.makedirs(data_dir)
# cifar10_url = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
#
# # Check if file exists, otherwise download it
# data_file = os.path.join(data_dir, 'cifar-10-binary.tar.gz')
# if os.path.isfile(data_file):
#     pass
# else:
#     # Download file
#     def progress(block_num, block_size, total_size):
#         progress_info = [cifar10_url, float(block_num * block_size) / float(total_size) * 100.0]
#         print('\r Downloading {} - {:.2f}%'.format(*progress_info), end="")
#     filepath, _ = urllib.request.urlretrieve(cifar10_url, data_file, progress)
#     # Extract file
#     tarfile.open(filepath, 'r:gz').extractall(data_dir)
#
#
# # Define CIFAR reader
# def read_cifar_files(filename_queue, distort_images = True):
#     reader = tf.FixedLengthRecordReader(record_bytes=record_length)
#     key, record_string = reader.read(filename_queue)
#     record_bytes = tf.decode_raw(record_string, tf.uint8)
#     image_label = tf.cast(tf.slice(record_bytes, [0], [1]), tf.int32)
#
#     # Extract image
#     image_extracted = tf.reshape(tf.slice(record_bytes, [1], [image_vec_length]),
#                                  [num_channels, image_height, image_width])
#
#     # Reshape image
#     image_uint8image = tf.transpose(image_extracted, [1, 2, 0])
#     reshaped_image = tf.cast(image_uint8image, tf.float32)
#     # Randomly Crop image
#     final_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, crop_width, crop_height)
#
#     if distort_images:
#         # Randomly flip the image horizontally, change the brightness and contrast
#         final_image = tf.image.random_flip_left_right(final_image)
#         final_image = tf.image.random_brightness(final_image,max_delta=63)
#         final_image = tf.image.random_contrast(final_image,lower=0.2, upper=1.8)
#
#     # Normalize whitening
#     final_image = tf.image.per_image_standardization(final_image)
#     return(final_image, image_label)
#
#
# # Create a CIFAR image pipeline from reader
# def input_pipeline(batch_size, train_logical=True):
#     if train_logical:
#         files = [os.path.join(data_dir, extract_folder, 'data_batch_{}.bin'.format(i)) for i in range(1,6)]
#     else:
#         files = [os.path.join(data_dir, extract_folder, 'test_batch.bin')]
#     filename_queue = tf.train.string_input_producer(files)
#     image, label = read_cifar_files(filename_queue)
#
#     # min_after_dequeue defines how big a buffer we will randomly sample
#     #   from -- bigger means better shuffling but slower start up and more
#     #   memory used.
#     # capacity must be larger than min_after_dequeue and the amount larger
#     #   determines the maximum we will prefetch.  Recommendation:
#     #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
#     min_after_dequeue = 5000
#     capacity = min_after_dequeue + 3 * batch_size
#     example_batch, label_batch = tf.train.shuffle_batch([image, label],
#                                                         batch_size=batch_size,
#                                                         capacity=capacity,
#                                                         min_after_dequeue=min_after_dequeue)
#
#     return(example_batch, label_batch)
#
#
# # Define the model architecture, this will return logits from images
# def cifar_cnn_model(input_images, batch_size, train_logical=True):
#     def truncated_normal_var(name, shape, dtype):
#         return(tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.truncated_normal_initializer(stddev=0.05)))
#     def zero_var(name, shape, dtype):
#         return(tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.constant_initializer(0.0)))
#
#     # First Convolutional Layer
#     with tf.variable_scope('conv1') as scope:
#         # Conv_kernel is 5x5 for all 3 colors and we will create 64 features
#         conv1_kernel = truncated_normal_var(name='conv_kernel1', shape=[5, 5, 3, 64], dtype=tf.float32)
#         # We convolve across the image with a stride size of 1
#         conv1 = tf.nn.conv2d(input_images, conv1_kernel, [1, 1, 1, 1], padding='SAME')
#         # Initialize and add the bias term
#         conv1_bias = zero_var(name='conv_bias1', shape=[64], dtype=tf.float32)
#         conv1_add_bias = tf.nn.bias_add(conv1, conv1_bias)
#         # ReLU element wise
#         relu_conv1 = tf.nn.relu(conv1_add_bias)
#
#     # Max Pooling
#     pool1 = tf.nn.max_pool(relu_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool_layer1')
#
#     # Local Response Normalization (parameters from paper)
#     # paper: http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
#     norm1 = tf.nn.lrn(pool1, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75, name='norm1')
#
#     # Second Convolutional Layer
#     with tf.variable_scope('conv2') as scope:
#         # Conv kernel is 5x5, across all prior 64 features and we create 64 more features
#         conv2_kernel = truncated_normal_var(name='conv_kernel2', shape=[5, 5, 64, 64], dtype=tf.float32)
#         # Convolve filter across prior output with stride size of 1
#         conv2 = tf.nn.conv2d(norm1, conv2_kernel, [1, 1, 1, 1], padding='SAME')
#         # Initialize and add the bias
#         conv2_bias = zero_var(name='conv_bias2', shape=[64], dtype=tf.float32)
#         conv2_add_bias = tf.nn.bias_add(conv2, conv2_bias)
#         # ReLU element wise
#         relu_conv2 = tf.nn.relu(conv2_add_bias)
#
#     # Max Pooling
#     pool2 = tf.nn.max_pool(relu_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_layer2')
#
#      # Local Response Normalization (parameters from paper)
#     norm2 = tf.nn.lrn(pool2, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75, name='norm2')
#
#     # Reshape output into a single matrix for multiplication for the fully connected layers
#     reshaped_output = tf.reshape(norm2, [batch_size, -1])
#     reshaped_dim = reshaped_output.get_shape()[1].value
#
#     # First Fully Connected Layer
#     with tf.variable_scope('full1') as scope:
#         # Fully connected layer will have 384 outputs.
#         full_weight1 = truncated_normal_var(name='full_mult1', shape=[reshaped_dim, 384], dtype=tf.float32)
#         full_bias1 = zero_var(name='full_bias1', shape=[384], dtype=tf.float32)
#         full_layer1 = tf.nn.relu(tf.add(tf.matmul(reshaped_output, full_weight1), full_bias1))
#
#     # Second Fully Connected Layer
#     with tf.variable_scope('full2') as scope:
#         # Second fully connected layer has 192 outputs.
#         full_weight2 = truncated_normal_var(name='full_mult2', shape=[384, 192], dtype=tf.float32)
#         full_bias2 = zero_var(name='full_bias2', shape=[192], dtype=tf.float32)
#         full_layer2 = tf.nn.relu(tf.add(tf.matmul(full_layer1, full_weight2), full_bias2))
#
#     # Final Fully Connected Layer -> 10 categories for output (num_targets)
#     with tf.variable_scope('full3') as scope:
#         # Final fully connected layer has 10 (num_targets) outputs.
#         full_weight3 = truncated_normal_var(name='full_mult3', shape=[192, num_targets], dtype=tf.float32)
#         full_bias3 =  zero_var(name='full_bias3', shape=[num_targets], dtype=tf.float32)
#         final_output = tf.add(tf.matmul(full_layer2, full_weight3), full_bias3)
#
#     return(final_output)
#
#
# # Loss function
# def cifar_loss(logits, targets):
#     # Get rid of extra dimensions and cast targets into integers
#     targets = tf.squeeze(tf.cast(targets, tf.int32))
#     # Calculate cross entropy from logits and targets
#     cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
#     # Take the average loss across batch size
#     cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
#     return(cross_entropy_mean)
#
#
# # Train step
# def train_step(loss_value, generation_num):
#     # Our learning rate is an exponential decay after we wait a fair number of generations
#     model_learning_rate = tf.train.exponential_decay(learning_rate, generation_num,
#                                                      num_gens_to_wait, lr_decay, staircase=True)
#     # Create optimizer
#     my_optimizer = tf.train.GradientDescentOptimizer(model_learning_rate)
#     # Initialize train step
#     train_step = my_optimizer.minimize(loss_value)
#     return(train_step)
#
#
# # Accuracy function
# def accuracy_of_batch(logits, targets):
#     # Make sure targets are integers and drop extra dimensions
#     targets = tf.squeeze(tf.cast(targets, tf.int32))
#     # Get predicted values by finding which logit is the greatest
#     batch_predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
#     # Check if they are equal across the batch
#     predicted_correctly = tf.equal(batch_predictions, targets)
#     # Average the 1's and 0's (True's and False's) across the batch size
#     accuracy = tf.reduce_mean(tf.cast(predicted_correctly, tf.float32))
#     return(accuracy)
#
# # Get data
# print('Getting/Transforming Data.')
# # Initialize the data pipeline
# images, targets = input_pipeline(batch_size, train_logical=True)
# # Get batch test images and targets from pipline
# test_images, test_targets = input_pipeline(batch_size, train_logical=False)
#
# # Declare Model
# print('Creating the CIFAR10 Model.')
# with tf.variable_scope('model_definition') as scope:
#     # Declare the training network model
#     model_output = cifar_cnn_model(images, batch_size)
#     # This is very important!!!  We must set the scope to REUSE the variables,
#     #  otherwise, when we set the test network model, it will create new random
#     #  variables.  Otherwise we get random evaluations on the test batches.
#     scope.reuse_variables()
#     test_output = cifar_cnn_model(test_images, batch_size)
#
# # Declare loss function
# print('Declare Loss Function.')
# loss = cifar_loss(model_output, targets)
#
# # Create accuracy function
# accuracy = accuracy_of_batch(test_output, test_targets)
#
# # Create training operations
# print('Creating the Training Operation.')
# generation_num = tf.Variable(0, trainable=False)
# train_op = train_step(loss, generation_num)
#
# # Initialize Variables
# print('Initializing the Variables.')
# init = tf.global_variables_initializer()
# sess.run(init)
#
# # Initialize queue (This queue will feed into the model, so no placeholders necessary)
# tf.train.start_queue_runners(sess=sess)
#
# # Train CIFAR Model
# print('Starting Training')
# train_loss = []
# test_accuracy = []
# for i in range(generations):
#     _, loss_value = sess.run([train_op, loss])
#
#     if (i+1) % output_every == 0:
#         train_loss.append(loss_value)
#         output = 'Generation {}: Loss = {:.5f}'.format((i+1), loss_value)
#         print(output)
#
#     if (i+1) % eval_every == 0:
#         [temp_accuracy] = sess.run([accuracy])
#         test_accuracy.append(temp_accuracy)
#         acc_output = ' --- Test Accuracy = {:.2f}%.'.format(100.*temp_accuracy)
#         print(acc_output)
#
# # Print loss and accuracy
# # Matlotlib code to plot the loss and accuracies
# eval_indices = range(0, generations, eval_every)
# output_indices = range(0, generations, output_every)
#
# # Plot loss over time
# plt.plot(output_indices, train_loss, 'k-')
# plt.title('Softmax Loss per Generation')
# plt.xlabel('Generation')
# plt.ylabel('Softmax Loss')
# plt.show()
#
# # Plot accuracy over time
# plt.plot(eval_indices, test_accuracy, 'k-')
# plt.title('Test Accuracy')
# plt.xlabel('Generation')
# plt.ylabel('Accuracy')
# plt.show()
