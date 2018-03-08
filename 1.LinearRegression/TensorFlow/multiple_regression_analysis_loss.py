'''
tensorflow: multiple_regression_analysis
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops
from sklearn.preprocessing import StandardScaler
# %matplotlib inline

# リセット
ops.reset_default_graph()
sess = tf.Session()

# データ読み込み
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample = pd.read_csv('sample.csv', header=None)

# データの加工
train['year'] = train['datetime'].apply(lambda x:  x.split('-')[0])
train['month'] = train['datetime'].apply(lambda x:  x.split('-')[1])
test['year'] = test['datetime'].apply(lambda x:  x.split('-')[0])
test['month'] = test['datetime'].apply(lambda x:  x.split('-')[1])

train['year'] = train['year'].astype(np.int)
train['month'] = train['month'].astype(np.int)
test['year'] = test['year'].astype(np.int)
test['month'] = test['month'].astype(np.int)

def isMenu(x):
    if x == 'お楽しみメニュー':
        return 1
    else:
        return 0

train['fun'] = train['remarks'].apply(lambda x : isMenu(x))
test['fun'] = test['remarks'].apply(lambda x : isMenu(x))

# データを作成
trainX1 = np.transpose(np.matrix(train['temperature']))
trainX2 = np.transpose(np.matrix(train['year']))
trainX3 = np.transpose(np.matrix(train['month']))
trainX4 = np.transpose(np.matrix(train['fun']))
x_columns = np.concatenate((trainX1,trainX2,trainX3,trainX4), axis=1)
one_columns = np.transpose(np.matrix(np.repeat(1,len(x_columns))))
trainX = np.concatenate((one_columns, x_columns), axis=1)

trainY = np.transpose(np.matrix(train['y']))

# 標準化
ss = StandardScaler()
ss.fit(trainX)
trainX = ss.transform(trainX)

# ハイパーパラメータ
learning_rate = 0.001 # 大きすぎると発散する
batch_size = 50
epoch = 10000

# プレースホルダを初期化
x_data = tf.placeholder(shape=[None, 5], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# 変数を作成
w = tf.Variable(tf.random_normal(shape=[5,1]))

# モデルの方程式を設定
model_output = tf.matmul(x_data, w)

# L2損失関数を指定(MSE)
loss = tf.sqrt(tf.reduce_mean(tf.square(y_target - model_output)))

# 変数を初期化
init = tf.global_variables_initializer()
sess.run(init)

# 最適化関数を指定
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# 学習
loss_vec = []
for i in range(epoch):
    rand_index = np.random.choice(len(trainX), size=batch_size)

    rand_x = trainX[rand_index]
    rand_y = trainY[rand_index]

    sess.run(train_op, feed_dict={x_data: rand_x, y_target: rand_y})
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})

    loss_vec.append(temp_loss)

    if (i+1)%1000 == 0:
        print('Step #' + str(i+1) + ' w = '+ str(sess.run(w)))
        print('Loss = '+ str(temp_loss))

# 評価

[intercept] = sess.run(w[0])
[coef1] = sess.run(w[1])
[coef2] = sess.run(w[2])
[coef3] = sess.run(w[3])
[coef4] = sess.run(w[3])

# テストデータ
testX1 = np.transpose(np.matrix(test['temperature']))
testX2 = np.transpose(np.matrix(test['year']))
testX3 = np.transpose(np.matrix(test['month']))
testX4 = np.transpose(np.matrix(test['fun']))
x_columns = np.concatenate((testX1,testX2,testX3,testX4), axis=1)
one_columns = np.transpose(np.matrix(np.repeat(1,len(x_columns))))
testX = np.concatenate((one_columns, x_columns), axis=1)

pred = []
for idx in range(len(testX)):
    pred.append( coef1 * testX1[idx,0] + coef2 * testX2[idx,0] + coef3 * testX3[idx,0] + coef4 * testX4[idx,0] + intercept)


sample[1] = pred
sample.to_csv('my_submit_multi.csv', header=None, index=None)
