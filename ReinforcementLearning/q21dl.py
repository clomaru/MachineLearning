import numpy as np
import matplotlib.pyplot as plt
import random
import collections

'''
2.2式を教師にしているのは
2.2式は学習則であって関数ではない、
以前まではQTbaleというのでやっていたが、
それがNNを使うことで関数化するこができる。
勾配消失問題が解決できるようになったのでDeepNNができるようになった。
'''


###
# ハイパーパラメータ
###

epoch = 4000
lr = 0.1 # 学習率
reward = 1 # 報酬
gamma = 0.9 # 割引率
epcilon = 0.3 # ε-greedy



###
# mapの設定
###

state_no = 7 # 状態の数
action_no = 2 # 行動の数
level = 2 # 枝分かれの深さ
goal = 6 # 目的地

up = 0
down = 1



###
# ニューラルネットのモデル
###

input_no = 7 # 入力層のノード
hidden_no = 2 # 中間層のノード
output_no = 2 # 出力層のノード
nn_lr = 3 # 学習係数



###
# 関数定義
###

# Q値を更新する
def updateq(status, snext, action, w_hidden, w_output, o_hidden):
    '''
    :pram status: <int> 状態
    :pram snext: <int> 次の状態
    :pram action: <int> 行動
    :pram w_hidden: <int> 隠れ層の重み
    :pram w_output: <int> 出力層の重み
    :pram o_hidden: <int> 隠れ層の出力
    :return: 更新したQ値
    '''

    # 学習のデータセットを初期化
    e = np.zeros(input_no + 1)

    # 現在状態statusでのQ値を求める
    e[status] = 1
    qvalue_sa = forward(w_hidden, w_output[action], o_hidden, e)
    e[status] = 0

    # 次の状態snextでの最大Q値を求める
    e[snext] = 1
    qvalue_snexta = forward(w_hidden, w_output[set_a_by_q(snext, w_hidden, w_output, o_hidden)], o_hidden, e)

    # 報酬が付与される場合
    if snext == goal: return qvalue_sa + lr * ( reward - qvalue_sa )
    else: return qvalue_sa + lr * ( gamma *  qvalue_snexta - qvalue_sa )


# 行動を選択する
selecta = lambda status, w_hidden, w_output, o_hidden: rand_0or1 if random.random()<epcilon else set_a_by_q(status, w_hidden, w_output, o_hidden)


# Q値の最大値から次の行動を決定
def set_a_by_q(status, w_hidden, w_output, o_hidden):
    '''
    :pram status: <int> 状態
    :pram w_hidden: <int> 隠れ層の重み
    :pram w_output: <int> 出力層の重み
    :pram o_hidden: <int> 隠れ層の出力
    :return: 次の行動
    '''

    # 学習のデータセットを初期化
    e = np.zeros(input_no + 1)

    e[status] = 1
    up_qvalue = forward(w_hidden, w_output[up], o_hidden, e)
    down_qvalue = forward(w_hidden, w_output[down], o_hidden, e)

    if up_qvalue > down_qvalue:
        return up
    else:
        return down



# 行動によって次の状態に遷移
nexts = lambda status, action: 2 * status + 1 + action


# Q値を出力する
def printQvalue(w_hidden, w_output, o_hidden):

    print_list = []
    e = np.zeros(input_no + 1)

    for i in range(state_no):
        for j in range(action_no):
            e[i] = 1
            result = forward(w_hidden, w_output[j], o_hidden, e)
            print_list.append(result)
            # result_list = np.append(result_list, result)
            e[i] = 0

    return print_list


# 0または1を返す乱数関数
rand_0or1 = (lambda: 1 if random.random()>=0.5 else 0)()

# 順方向の計算
def forward(w_hidden, w_output, o_hidden, e):
    '''
    :pram w_hidden: <int> 隠れ層の重み
    :pram w_output: <int> 隠れ層の出力
    :pram o_hidden: <int> 隠れ層の出力
    :pram e: <int> 学習データセット
    :return: sigmoid
    '''

    # w_hiddenはhidden_no * (input_no + 1)の2次元行列。
    # +1の意味は各隠れ層のバイアス。

    hidden_no

    # o_hiddenの計算
    for i in range(hidden_no):
        u = 0

        for j in range(input_no):
            u += e[j] * w_hidden[i][j]

        u -= w_hidden[i][j] # 閾値を引く
        o_hidden[i] = sigmoid(u)

    # 出力outputの計算
    output = 0
    for i in range(hidden_no):
        output += o_hidden[i] * w_output[i]

    output -= w_output[i] # 閾値を引く

    return sigmoid(output)


# 出力層の重みの更新
def olearn(w_output, o_hidden, e, output, action):
    '''
    :pram w_output: <int> 出力層の重み
    :pram o_hidden: <int> 隠れ層の出力
    :pram e: <int> 学習データセット
    :pram output: <int> 出力
    :pram action: <int> 行動
    '''

    # 誤差の計算
    d = (e[input_no + action] - output) * output * ( 1 - output )

    # 重みの更新
    for i in range(hidden_no):
        w_output[i] += nn_lr * o_hidden[i] * d

    # 閾値の更新
    w_output[i] += nn_lr * (-1.0) * d



# 中間層の重みを更新
def hlearn(w_hidden, w_output, o_hidden, e, output, action):
    '''
    :pram w_hidden: <int> 隠れ層の重み
    :pram w_output: <int> 出力層の重み
    :pram o_hidden: <int> 隠れ層の出力
    :pram e: <int> 学習データセット
    :pram output: <int> 出力
    :pram action: <int> 行動
    '''

    # 中間層の各セルを対象
    for j in range(hidden_no):
        dj = o_hidden[j] * ( 1 - o_hidden[j]) * w_output[j] * (e[input_no+action] - output) * output * ( 1 - output )

        # i番目の重みを処理
        for i in range(input_no):
            w_hidden[j][i] += nn_lr * e[i] * dj

        # 閾値の更新
        w_hidden[j][i] += nn_lr * (-1.0) * dj



# シグモイド関数
sigmoid = lambda x: 1.0 / ( 1.0 + np.exp(-x) )


# 重みの初期化
w_hidden = np.random.rand(hidden_no, input_no + 1)
w_output = np.random.rand(output_no, hidden_no + 1)
print(w_hidden,w_output)



###
# 学習
###

result_list = []
for _ in range(epoch):

    status = 0 # 行動の初期状態
    o_hidden = np.zeros(hidden_no + 1)
    output = np.zeros(output_no)

    # Q値の更新
    for _ in range(level):

        # 行動選択
        action = selecta(status, w_hidden, w_output, o_hidden)
        print("status={} action={}".format(status, action))

        # 次の状態に遷移
        snext = nexts(status, action)

        e = np.zeros(input_no + output_no)
        e[status] = 1
        e[input_no + action] = updateq(status, snext, action, w_hidden, w_output, o_hidden)

        # 順方向の計算
        output[action] = forward(w_hidden, w_output[action], o_hidden, e)

        # 出力層の重みの更新
        olearn(w_output[action], o_hidden, e, output[action], action)

        # 中間層の重みの更新
        hlearn(w_hidden, w_output[action], o_hidden, e, output[action], action)

        status = snext

    result_list = np.append(result_list, printQvalue(w_hidden, w_output, o_hidden))

result_list = np.reshape(result_list, (epoch, state_no, action_no))



###
# 描画
###


plt.plot(result_list[:,0,0], color='b', marker='*', label='[0,0]')
plt.plot(result_list[:,0,1], color='b', marker='.', label='[0,1]')
plt.plot(result_list[:,1,0], color='g', marker='*', label='[1,0]')
plt.plot(result_list[:,1,1], color='g', marker='.', label='[1,1]')
plt.plot(result_list[:,2,0], color='r', marker='*', label='[2,0]')
plt.plot(result_list[:,2,1], color='r', marker='.', label='[2,1]')
plt.plot(result_list[:,3,0], color='c', marker='*', label='[3,0]')
plt.plot(result_list[:,3,1], color='c', marker='.', label='[3,1]')
plt.plot(result_list[:,4,0], color='m', marker='*', label='[4,0]')
plt.plot(result_list[:,4,1], color='m', marker='.', label='[4,1]')
plt.plot(result_list[:,5,0], color='y', marker='*', label='[5,0]')
plt.plot(result_list[:,5,1], color='y', marker='.', label='[5,1]')
plt.plot(result_list[:,6,0], color='k', marker='*', label='[6,0]')
plt.plot(result_list[:,6,1], color='k', marker='.', label='[6,1]')
plt.xlabel('epoch')
plt.ylabel('Q-value')
plt.legend(loc='upper left')
plt.show()
