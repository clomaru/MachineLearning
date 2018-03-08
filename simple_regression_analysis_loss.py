# TensoFlow
# simple regression analysis
# lossを用いる
#----------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
# %matplotlib inline

# リセット
ops.reset_default_graph()
sess = tf.Session()


###
# データ読み込み
###

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# sample = pd.read_csv('sample.csv', header=None)


###
# データを作成
###

trainX = train['temperature']
trainY = train['y']


###
# ハイパーパラメータ
###

learning_rate = 0.01 # 大きすぎると発散する
batch_size = 200
epoch = 100000


###
# モデルの作成
###

# プレースホルダを初期化
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# 変数を作成
a = tf.Variable(tf.random_normal(shape=[1,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

# モデルの方程式を設定
model_output = tf.add(tf.matmul(x_data, a), b)

# L2損失関数を指定(RMSE)
loss = tf.sqrt(tf.reduce_mean(tf.square(y_target - model_output)))

# 変数を初期化
init = tf.global_variables_initializer()
sess.run(init)

# 最適化関数を指定
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


###
# 学習
###

loss_vec = []
for i in range(epoch):
    rand_index = np.random.choice(len(trainX), size=batch_size)

    rand_x = np.transpose([trainX[rand_index]])
    rand_y = np.transpose([trainY[rand_index]])

    sess.run(train_op, feed_dict={x_data: rand_x, y_target: rand_y})
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})

    loss_vec.append(temp_loss)

    if (i+1)%1000 == 0:
        print('Step #{}: y = {} * x + {}'.format(i+1, sess.run(a), sess.run(b)))
        print('Loss = '+ str(temp_loss))

[coef] = sess.run(a)
[intercept] = sess.run(b)


###
# 可視化
###

best_fit = []
for i in trainX:
    best_fit.append(coef * i + intercept)

plt.plot(trainX, trainY, 'o', label="data points")
plt.plot(trainX, best_fit, 'r-', label="best fit line", linewidth=3)
plt.legend(loc='upper left')
plt.title('sepal length vs pedal width')
plt.xlabel('pedal width')
plt.ylabel('sepal length')
plt.show()

plt.plot(loss_vec, 'k-')
plt.title('l2 loss per generation')
plt.xlabel('generation')
plt.ylabel('l2 loss')
plt.show()

testX = test['temperature']
pred = []
for i in testX:
    pred.append(coef[0] * i + intercept[0])

# sample[1] = pred
# sample.to_csv('sral4.csv', header=None, index=None)
