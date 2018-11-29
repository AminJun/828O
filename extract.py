import numpy as np
import tensorflow as tf
import sys
from sklearn.preprocessing import scale
import pandas as pd
from tensorflow.python.tools import inspect_checkpoint as chkp



class Extractor:
    def extract(self, epoch=2000, n=2, normalize=True, mode='c'):
        tf.reset_default_graph()
        root="./models/f_model_{}_{}_{}_{}".format(epoch, n, normalize, mode)
        file_name="{}/{}".format(root, "model.ckpt")
        print(file_name)
        chkp.print_tensors_in_checkpoint_file(file_name, tensor_name='Variable', all_tensors=False)
        
        my_w = tf.get_variable("Variable", shape=[3890, 115181])
        saver = tf.train.Saver()
        sess = tf.Session()
       	saver.restore(sess, file_name)
        w = my_w.eval(session=sess)
        # np.savetxt("{}/{}".format(root, "w"), w, delimiter=',')
        np.save("{}/{}".format(root, "w.npy"), w)


if __name__ == '__main__':
    no_layers = int(sys.argv[1])
    do_normalize = sys.argv[2] == 'true'
    train_type = sys.argv[3]
    extractor = Extractor()
    extractor.extract( 500, no_layers, do_normalize, train_type)
