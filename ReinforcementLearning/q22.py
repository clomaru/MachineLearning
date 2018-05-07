import numpy as np
import matplotlib.pyplot as plt
import random
import collections


###
# ハイパーパラメータ
###

epoch = 500
lr = 0.05 # 学習率
reward = 10 # 報酬
gamma = 0.9 # 割引率
epcilon = 0.3 # ε-greedy


###
# mapの設定
###

statend = 7 # 状態の数
actionno = 2 # 行動の数
level = 2 # 枝分かれの深さ
goal = 6

up = 0
down = 1


###
# 関数定義
###

# Q値を更新する
def updateq(s, snext, a, qvalue):
    # 報酬が付与される場合
    if snext == goal:
        return qvalue[s][a] + lr * ( reward - qvalue[s][a] )
    else:
        return qvalue[s][a] + lr * ( gamma * qvalue[snext][set_a_by_q(snext, qvalue)] - qvalue[s][a] )

# 行動を選択する
selecta = lambda s, qvalue: rand_0or1 if random.random()<epcilon else set_a_by_q(s, qvalue)

# Q値最大値を選択
set_a_by_q = lambda s, qvalue: up if qvalue[s][up] > qvalue[s][down] else down

# 行動によって次の状態に遷移
nexts = lambda s,a: 2 * s + 1 + a

# 0または1を返す乱数関数
rand_0or1 = (lambda: 1 if random.random()>=0.5 else 0)()


###
# 学習
###

# Q値の初期化
qvalue = np.random.rand(statend, actionno)
print(qvalue)

qvalue_list = np.array([])
for i in range(epoch):

    s = 0 # 行動の初期状態
    for j in range(level):

        a = selecta(s, qvalue)
        print("s={} a={}".format(s, a))

        snext = nexts(s, a)
        qvalue[s][a] = updateq(s, snext, a, qvalue)

        s = snext

    qvalue_list = np.append(qvalue_list, qvalue)
    print(qvalue)

qvalue_list = np.reshape(qvalue_list, (-1,7,2))


###
# 描画
###

plt.plot(qvalue_list[:,0,0], color='b', marker='*', label='[0,0]')
plt.plot(qvalue_list[:,0,1], color='b', marker='.', label='[0,1]')
plt.plot(qvalue_list[:,1,0], color='g', marker='*', label='[1,0]')
plt.plot(qvalue_list[:,1,1], color='g', marker='.', label='[1,1]')
plt.plot(qvalue_list[:,2,0], color='r', marker='*', label='[2,0]')
plt.plot(qvalue_list[:,2,1], color='r', marker='.', label='[2,1]')
plt.plot(qvalue_list[:,3,0], color='c', marker='*', label='[3,0]')
plt.plot(qvalue_list[:,3,1], color='c', marker='.', label='[3,1]')
plt.plot(qvalue_list[:,4,0], color='m', marker='*', label='[4,0]')
plt.plot(qvalue_list[:,4,1], color='m', marker='.', label='[4,1]')
plt.plot(qvalue_list[:,5,0], color='y', marker='*', label='[5,0]')
plt.plot(qvalue_list[:,5,1], color='y', marker='.', label='[5,1]')
plt.plot(qvalue_list[:,6,0], color='k', marker='*', label='[6,0]')
plt.plot(qvalue_list[:,6,1], color='k', marker='.', label='[6,1]')
plt.xlabel('epoch')
plt.ylabel('Q-value')
plt.legend(loc='upper left')
plt.show()
