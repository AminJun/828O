import numpy as np
import tensorflow as tf
import sys
from sklearn.preprocessing import scale
import pandas as pd


class Model:
    def __init__(self, n=2, in_size=3890, out_size=115181):
        self.x = tf.placeholder(tf.float32, shape=[None, in_size])
        self.y = tf.placeholder(tf.float32, shape=[None, out_size])

        self.layers = [None] * n
        self.w = [None] * (n - 1)
        self.layers[0] = self.x

        now = in_size
        for i in range(n - 2):
            self.w[i] = tf.Variable(tf.zeros([int(now), int((now + out_size) / 2)]))
            self.layers[i + 1] = tf.sigmoid(tf.matmul(self.layers[i], self.w[i]))
            now = int((now + out_size) / 2)
        self.w[n - 2] = tf.Variable(tf.zeros([now, out_size]))
        self.layers[n - 1] = tf.matmul(self.layers[n - 2], self.w[n - 2])
        self.prediction = self.layers[n - 1]
        self.loss = tf.losses.mean_squared_error(self.y, self.prediction)


class Trainer:
    def __init__(self, n=2, normalize=False, train_type='both'):
        self.x, self.y = self.read(normalize, train_type)
        self.mode = train_type
        self.model = Model(n)
        self.normalize = normalize

    def read(self, normalize=False, train_type='both'):
        x = pd.read_csv('NN-Data/{}'.format('Input_Layer.csv'), delimiter=',')[1:]
        y = pd.read_csv('NN-Data/{}'.format('Output_Layer.csv'), delimiter=',')[1:]
        index = x.values[:, 0]
        sel = [i for i in range(len(index))]
        if train_type is 'c':
            sel = [i for i in range(len(index)) if 'TCGA' in index[i]]
        if train_type is 'h':
            sel = [i for i in range(len(index)) if 'GTEX' in index[i]]
        x = np.array(x.values[sel, 1:])
        y = np.array(y.values[sel, 1:])
        if normalize:
            x = scale(x)
        return x, y

    def train(self, sess, epoch=2000, batch_size=30):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(self.model.loss)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for epoch_i in range(epoch):
            num_batch = int(len(self.x) / batch_size)
            loss = 0
            if epoch_i % 1000 is 0: 
                saver.save(sess, "./models/model_{}_{}_{}_{}/model.ckpt".format(epoch_i, len(self.model.layers), self.normalize, self.mode))
            for batch_i in range(num_batch):
                batch_start = batch_i * batch_size
                batch_x = self.x[batch_start:batch_start + batch_size]
                batch_y = self.y[batch_start:batch_start + batch_size]
                _, c = sess.run([optimizer, self.model.loss], feed_dict={self.model.x: batch_x, self.model.y: batch_y})
                loss += c
                #print("#{} / {}".format(batch_i, num_batch))
            print("#{} : {}".format(epoch_i, loss))
        saver.save(sess, "./models/model_{}_{}_{}_{}/model.ckpt".format(epoch, len(self.model.layers), self.normalize, self.mode))


if __name__ == '__main__':
    no_layers = int(sys.argv[1])
    do_normalize = sys.argv[2] == 'true'
    train_type = sys.argv[3]
    trainer = Trainer(no_layers, do_normalize, train_type)
    with tf.Session() as c_sess:
        trainer.train(c_sess)
