import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression as LR
import statsmodels.formula.api as smf
import statsmodels.api as sm

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

drop_list = ['Utilities','Condition2','HouseStyle','RoofMatl','Exterior1st','Exterior2nd','Heating','Electrical','GarageQual','PoolQC','MiscFeature']
drop_list2 = ['Id','YearRemodAdd','BsmtFinSF1','BsmtUnfSF','LowQualFinSF','BsmtHalfBath','HalfBath','GarageYrBlt','GarageArea','OpenPorchSF','EnclosedPorch','PoolArea','MiscVal','MoSold','YrSold','Alley','LotShape','LotConfig','Neighborhood','Condition1','BldgType','RoofStyle','MasVnrType','ExterQual','Foundation','BsmtExposure','BsmtFinType1','BsmtFinType2','HeatingQC','CentralAir','Functional','FireplaceQu','GarageFinish','PavedDrive','Fence','SaleType','SaleCondition']
drop_list3 = ['BsmtFinSF2','1stFlrSF','2ndFlrSF','GrLivArea','3SsnPorch','MSZoning','Street','ExterCond','BsmtCond','GarageType']
drop_list4 = ['LandSlope','GarageCond','LotFrontage']





trainX = train.drop('SalePrice', axis=1)

trainX = trainX.drop(drop_list, axis=1)
trainX = trainX.drop(drop_list2, axis=1)
trainX = trainX.drop(drop_list3, axis=1)
trainX = trainX.drop(drop_list4, axis=1)
trainX = trainX.drop(drop_list5, axis=1)
trainX = pd.get_dummies(trainX, drop_first=True)
trainX = trainX.fillna(method="ffill")
trainX = sm.add_constant(trainX)

testX = test.drop(drop_list, axis=1)
testX = testX.drop(drop_list2, axis=1)
testX = testX.drop(drop_list3, axis=1)
testX = testX.drop(drop_list4, axis=1)
testX = testX.drop(drop_list5, axis=1)
testX = pd.get_dummies(testX, drop_first=True)
testX = testX.fillna(method="ffill")
testX = sm.add_constant(testX)

trainY = train['SalePrice']

# model = smf.OLS(trainY, trainX)
# result = model.fit()
# result.summary()

model = LR()
model.fit(trainX, trainY)

pred = model.predict(testX)
test['SalePrice'] = pred
# test[['Id','SalePrice']].to_csv('submission9.csv', index=False)

# matplotlibをインポート
import matplotlib.pyplot as plt
%matplotlib inline

trainY_pred = model.predict(trainX)


# 学習用、検証用それぞれで残差をプロット
plt.scatter(trainY_pred, trainY_pred - trainY, c = 'blue', marker = 'o', label = 'Train Data')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
# 凡例を左上に表示
plt.legend(loc = 'upper left')
# y = 0に直線を引く
plt.hlines(y = 0, xmin = -10, xmax = 50, lw = 2, color = 'red')
plt.xlim([10, 50])
plt.show()
