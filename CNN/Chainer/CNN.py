import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, datasets, iterators, optimizers
from chainer.training import extensions
import numpy as np

batch_size = 10
uses_device = 0 # GPU

class MNIST_Conv_NN(chainer.Chain):

    def __init__(self):
        super(MNIST_Conv_NN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 8, ksize=3) # フィルタサイズ:3,出力数:8
            self.linear1 = L.Linear(1352, 10) # 出力数:10

    # クラスのインスタンスを直接関数として実行した際に呼び出される関数
    def __call__(self, x, t=None, train=True):
        h1 = self.conv1(x) # 畳み込み層
        h2 = F.relu(h1) # 活性化関数
        h3 = F.max_pooling_2d(h2, 2) # プーリング層
        h4 = self.linear1(h3) # 全結合層

        # 損失か結果を返す
        return F.softmax_cross_entropy(h4, t) if train else F.softmax(h4)


model = MNIST_Conv_NN()

if uses_device >= 0:
    chainer.cuda.get_device_from_id(0).use()
    chainer.cuda.check_cuda_available()

    model.to_gpu()

train, test = chainer.datasets.get_mnist(ndim=3)

# 繰り返し条件を作成する
train_iter = iterators.SerialIterator(train, batch_size, shuffle=True)
test_iter = iterators.SerialIterator(test, batch_size, repeat=False, shuffle=False)

# 誤差逆伝播法アルゴリズムを選択する
optimizer = optimizers.Adam()
optimizer.setup(model)

# デバイスを選択してTrainerを作成する
updater = training.StandardUpdater(train_iter, optimizer, device=uses_device)
trainer = training.Trainer(updater, (5, 'epoch'), out="result")

# テストをTrainerに設定する
trainer.extend(extensions.Evaluator(test_iter, model, device=uses_device))

# 学習の進展を表示するようにする
trainer.extend(extensions.ProgressBar())

# 教師データとテスト用データの正解率を表示する
# trainer.extend(extensions.PrintReport(['main/accuracy', 'validation/main/accuracy']))

# 機械学習を実行する
trainer.run()

# 学習結果を保存する
chainer.serializers.save_hdf5('chapt02.hdf5', model)
