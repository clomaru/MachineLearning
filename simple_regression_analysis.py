# sklearn
# simple regression analysis
#----------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from sklearn.linear_model import LinearRegression as LR
# %matplotlib inline


###
# データ読み込み
###

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 目的変数:y
# 説明変数:temperature
trainX = train['temperature']
trainY = train['y']
testX = test['temperature']


###
# データの加工
###

trainX = trainX.values.reshape(-1,1)
testX = testX.values.reshape(-1,1)


###
# 単回帰
###

model1 = LR()
model1.fit(trainX, trainY)
coef = model1.coef_
intercept = model1.intercept_

pred = model1.predict(testX)


###
# 可視化
###

# y = ax + b の式を作成
func = lambda x: x * coef + intercept
line = lines.Line2D([0,40],[func(0), func(40)], color='r')

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.add_line(line)
ax.scatter(x=trainX, y=trainY)
fig.show()
