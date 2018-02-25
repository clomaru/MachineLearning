'''
tensorflow: simple_regression_analysis
逆行列法を用いる
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
# %matplotlib inline

sess = tf.Session()

'''
データ読み込み
'''
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

'''
データを作成
'''
trainX = train['temperature']
trainY = train['y']

# 計画行列Xを作成
one_columns = np.transpose(np.matrix(np.repeat(1, len(trainX))))
x_columns = np.transpose(np.matrix(trainX))
X = np.concatenate((one_columns, x_columns), axis=1) # 水平方向に合体

# 行列yを作成(castも)
y = np.transpose(np.matrix(trainY)).astype(np.float64)

# テンソルを作成
X_tensor = tf.constant(X)
y_tensor = tf.constant(y)

# 逆行列法の計算グラフを作成する
XT_X = tf.matmul(tf.transpose(X_tensor), X_tensor)
product = tf.matmul(tf.matrix_inverse(XT_X), tf.transpose(X))
solution = tf.matmul(product, y_tensor)

'''
演算の実行
'''
solution_eval = sess.run(solution)

coef = solution_eval[1][0]
intercept = solution_eval[0][0]
print('coef: {}'.format(coef))
print('intercept: {}'.format(intercept))

'''
可視化
'''
best_fit = []
for i in trainX:
    best_fit.append(coef * i + intercept)

plt.plot(trainX, trainY, 'o')
plt.plot(trainX, best_fit, 'r-')
plt.show()
