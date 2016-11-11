from __future__ import division
from __future__ import print_function

import random
import pprint
import tensorflow as tf
import sys

from attention import AttentionNN
# from tf_attention import TFAttentionNN

pp = pprint.PrettyPrinter()

flags = tf.app.flags
 
flags.DEFINE_integer("max_size", 30, "Maximum sentence length [30]")
flags.DEFINE_integer("batch_size", 128, "Number of examples in minibatch [128]")
flags.DEFINE_integer("random_seed", 123, "Value of random seed [123]")
flags.DEFINE_integer("epochs", 12, "Number of epochs to run [10]")
flags.DEFINE_integer("hidden_dim", 1000, "Size of hidden dimension [1000]")
flags.DEFINE_integer("embedding_dim", 256, "Size of hidden dimension [1000]")
flags.DEFINE_integer("num_layers", 4, "Number of recurrent layers [4]")
flags.DEFINE_float("init_learning_rate", 1., "initial learning rate [1]")
flags.DEFINE_float("grad_max_norm", 5., "gradient max norm [1]")
flags.DEFINE_boolean("use_attention", True, "Use attention [True]")
flags.DEFINE_float("dropout", 0.2, "Dropout [0.2]")
flags.DEFINE_boolean("show", False, "Print progress [False]")
flags.DEFINE_integer("mode", 0, "0 for training, 1 for testing, 2 for sampling [0]")
flags.DEFINE_boolean("validate", True, "True for cross validation, False otherwise [True]")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "Checkpoint directory [checkpoints]")
flags.DEFINE_integer("save_every", 1, "Save every [1] epochs")
flags.DEFINE_string("language", "viet", "Language for data and directory creation []")
flags.DEFINE_boolean("sample", False, "Sample data [False]")

FLAGS = flags.FLAGS

tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)


def main(_):

    pp.pprint(FLAGS.__flags)
    sys.stdout.flush()

    with tf.Session() as sess:
        attn = AttentionNN(FLAGS, sess)
        # attn = TFAttentionNN(FLAGS, sess)
        attn.build_model()
        attn.run()
        
        # if FLAGS.is_test:
        #     attn.run()
        # else:
        #     attn.run()



if __name__ == "__main__":
    tf.app.run()





