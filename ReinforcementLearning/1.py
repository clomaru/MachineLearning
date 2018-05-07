# じゃんけんの強化学習
import random
import matplotlib.pyplot as plt
# import collections

# グー,チョキ,パーを出す比率
my_rate = [1,1,1]
o_rate = [4,3,1]

lr = 0.000001 # 学習率
epoch = 500000

def hand(rate):
    rate_list = [idx * random.random() for idx in rate]
    return rate_list.index(max(rate_list))

o_list = [hand(o_rate) for n in range(epoch)]
# collections.Counter(o_list)

pay_off_matrix = [[0, 1, -1],[-1, 0, 1],[1, -1, 0]]

gu = []
cho = []
pa = []
for ohand in o_list:
    myhand = hand(my_rate)
    gain = pay_off_matrix[myhand][ohand]

    my_rate[myhand] += gain * lr * my_rate[myhand]

    gu.append(my_rate[0])
    cho.append(my_rate[1])
    pa.append(my_rate[2])

    print("me:{} opponent:{} gain:{:2d} {:.5f} {:.5f} {:.5f}".format(myhand, ohand, gain, my_rate[0], my_rate[1], my_rate[2]))

plt.plot(gu, 'k--', label='gu')
plt.plot(cho, 'r--', label='choki')
plt.plot(pa, 'y--', label='pa')
plt.title('gu : cho : pa = '+str(o_rate[0])+' : '+str(o_rate[1])+' : '+str(o_rate[2]))
plt.xlabel('epoch')
plt.ylabel('rate')
plt.legend(loc='upper left')
plt.show()
