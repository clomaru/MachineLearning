import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist_daata")

def model_inputs(real_dim, z_dim):
    """!@brief 本物XとランダムノイズZのplaceholadrを作成
    @param real_dim :Xの次元
    @param z_dim    :ノイズの次元
    @return
    """
    # TODO: ノイズの次元の最適値はいくら？

    inputs_real = tf.placeholder(tf.float32, (None, real_dim), name='input_real')
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name="input_z")

    return inputs_real, inputs_z

def generator(z, out_dim, n_units=128, reuse=False, alpha=0.01):
    """!@brief データを生成するgeneratorの作成
    @param z        :入力
    @param out_dim  :出力の次元
    @param n_units  :中間層の次元
    @param reuse    :関数内の変数値の保存
    @param alpha    :LeakyReLUのゼロ以下の傾き
    """
    with tf.variable_scope('generator', reuse=reuse):
        h1 = tf.layers.dense(z, n_units, activation=None)
        h1 = tf.maximum(alpha * h1, h1) # Leaky ReLU

        logits = tf.layers.dense(h1, out_dim, activation=None)
        out = tf.tanh(logits) # tanh(-1~1).画像なので(?)

        return out

# discriminatorの作成
def discriminator(x, n_units=128, reuse=False, alpha=0.01):
    """!@brief コンストラクタ
    @param x        :入力
    @param n_units  :中間層の次元
    @param reuse    :関数内の変数値の保存
    @param alpha    :LeakyReLUのゼロ以下の傾き
    """
    with tf.variable_scope('discriminator', reuse=reuse):
        h1 = tf.layers.dense(x, n_units, activation=None)
        h1 = tf.maximum(alpha * h1, h1)
        logits = tf.layers.dense(h1, 1, activation=None)
        out = tf.sigmoid(logits)

        return out, logits


# ハイパーパラメータの初期化
input_size = 784
z_size = 100 # ランダムベクトルのサイズ
g_hidden_size = 128
d_hidden_size =128
alpha = 0.01
smooth = 0.1 # discriminatorの学習を円滑にする調整


# グラフの定義
tf.reset_default_graph()

input_real, input_z = model_inputs(input_size, z_size)

g_model = generator(input_z, input_size, n_units=g_hidden_size, alpha=alpha)
d_model_real, d_logits_real = discriminator(input_real, n_units=d_hidden_size, alpha=alpha)
d_model_fake, d_logits_fake = discriminator(g_model, reuse=True, n_units=d_hidden_size, alpha=alpha)



# 損失関数の定義

# '1'との誤差
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_logits_real)*(1-smooth)))
# '0'との誤差
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels = tf.zeros_like(d_logits_real)))
d_loss = d_loss_real + d_loss_fake

# 正解との誤差
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_logits_fake)))


# 最適化の定義
learning_rate = 0.002
t_vars = tf.trainable_variables()
g_vars = [var for var in t_vars if var.name.startswith('generator')]
d_vars = [var for var in t_vars if var.name.startswith('discriminator')]

d_trian_optimize = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
g_trian_optimize = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)

batch_size =100
batch = mnist.train.next_batch(batch_size)

# トレーニングの実行
epochs = 100
samples = []
losses = []

# 途中の経過を保存
saver = tf.train.Saver(var_list=g_vars)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):

        # ミニバッチ学習
        for i in range(mnist.train.num_examples//batch_size):
            batch = mnist.train.next_batch(batch_size)

            batch_images = batch[0].reshape((batch_size, 784))
            batch_images = batch_images * 2 - 1

            # generator
            batch_z = np.random.uniform(-1,1,size=(batch_size, z_size))

            # 最適化計算、パラメータ更新
            _ = sess.run(d_trian_optimize, feed_dict=({input_real: batch_images, input_z: batch_z}))
            _ = sess.run(g_trian_optimize, feed_dict=({input_z: batch_z}))

        # lossを計算
        train_loss_d = sess.run(d_loss, {input_z: batch_z, input_real: batch_images})
        train_loss_g = g_loss.eval({input_z: batch_z})

        print("epoxh {}/{}".format(e+1, epochs),
              "d loss: {:.4f}".format(train_loss_d),
              "g loss: {:.4f}".format(train_loss_g))

        losses.append((train_loss_d, train_loss_g))

        sample_z = np.random.uniform(-1, 1, size=(16, z_size))
        gen_samples = sess.run(generator(input_z, input_size, n_units=g_hidden_size, reuse=True, alpha=alpha),
                               feed_dict={input_z: sample_z})

        samples.append(gen_samples)
        saver.save(sess, './checkpoints/generator.ckpt')

with open('training_samples.pkl', 'wb') as f:
    pkl.dump(samples, f)


# 可視化
%matplotlib inline
flg, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='D')
plt.plot(losses.T[1], label='G')
plt.title('Train Loss')
plt.legend()

# イメージに変換して表示する
def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(7,7), nrows=4, ncols=4, sharey=True, sharex=True)
    for ax,img in zip(axes.flatten(), samples[epoch]):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((28,28)), cmap='Greys_r'

    return fig, axes

with open('training_samples.pkl', 'rb') as f:
    samples = pkl.load(f)

_ = view_samples(-1, samples)

rows, cols = 10, 6
fig, axes = plt.subplots(figsize=(7,12), nrows=rows, ncols=cols, sharex=True, sharey=True)

for sample, ax_row in zip(samples[::int(len(samples)/rows)], axes):
    for img, ax in zip(sample[::int(len(sample)/cols)], ax_row):
        ax.imshow(img.reshape((28,28)), cmap='Greys_r')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

# チェックポイントファイルから機械に画像を生成させる
saver = tf.train.Saver(var_list=g_vars)
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    sample_z = np.random.uniform(-1, 1, size=(16, z_size))
    gen_samples = sess.run(generator(input_z, input_size, n_units=g_hidden_size, reuse=True, alpha=alpha),
                          feed_dict={input_z: sample_z})

_ = view_samples(0, [gen_samples])
