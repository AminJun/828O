import numpy as np
import tensorflow as tf
import sys
from sklearn.preprocessing import MinMaxScaler


class Model:
    def __init__(self, n=2, in_size=117000, out_size=3890):
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
    def __init__(self, n=2, normalize=False):
        self.x, self.y = self.read(normalize)
        self.model = Model(n)
        self.normalize = normalize

    def read(self, normalize=False):
        import pdb
        x = np.genfromtxt('NN-Data/{}'.format('Input_Layer.csv'), delimiter=',')[1:, 1:]
        y = np.genfromtxt('NN-Data/{}'.format('Output_Layer.csv'), delimiter=',')[1:, 1:]
        if normalize:
            x = MinMaxScaler().transform(x)
        return x, y

    def train(self, sess, epoch=1000, batch_size=2):
        optimizer = tf.train.AdamOptimizer().minimize(self.model.loss)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for epoch_i in range(epoch):
            num_batch = int(len(self.x) / batch_size)
            loss = 0
            for batch_i in range(num_batch):
                batch_start = batch_i * batch_size
                batch_x, batch_y = zip(self.x, self.y)[batch_start:batch_start + batch_size]
                import pdb
                pdb.set_trace()
                _, c = sess.run([optimizer, self.model.loss], feed_dict={self.model.x: batch_x, self.model.y: batch_y})
                loss += c
            print("#{} : {}".format(epoch_i, loss))
        saver.save(sess, "model{}{}".format(len(self.model.layers), self.normalize))


if __name__ == '__main__':
    no_layers = int(sys.argv[1])
    do_normalize = sys.argv[2] == 'true'
    trainer = Trainer(no_layers, do_normalize)
    with tf.Session() as c_sess:
        trainer.train(c_sess)
