from __future__ import division
from __future__ import print_function

import random

import tensorflow as tf
import sys

from attention import AttentionNN

flags = tf.app.flags
 
flags.DEFINE_integer("max_size", 30, "Maximum sentence length [30]")
flags.DEFINE_integer("batch_size", 128, "Number of examples in minibatch [128]")
flags.DEFINE_integer("random_seed", 42, "Value of random seed [42]")
flags.DEFINE_integer("epochs", 30, "Number of epochs to run [30]")
flags.DEFINE_integer("hidden_dim", 1000, "Size of hidden dimension [1000]")
flags.DEFINE_integer("embedding_dim", 256, "Size of embedding dimension [256]")
flags.DEFINE_integer("num_layers", 2, "Number of recurrent layers [2]")
flags.DEFINE_float("init_learning_rate", 1., "initial learning rate [1]")
flags.DEFINE_float("grad_max_norm", 5., "gradient max norm [1]")
flags.DEFINE_boolean("use_attention", True, "Use attention [True]")
flags.DEFINE_float("dropout", 0.2, "Dropout [0.2]")
flags.DEFINE_integer("mode", 0, "0 for training, 1 for testing, 2 for sampling [0]")
flags.DEFINE_boolean("validate", True, "True for cross validation, False otherwise [True]")
flags.DEFINE_integer("save_every", 5, "Save every [5] epochs")
flags.DEFINE_boolean("sample", False, "Use sample data [False]")
flags.DEFINE_string("model_name", "out", "model name for prefix to checkpoint file [unnamed]")
flags.DEFINE_boolean("use_unilingual", False, "use unilingual french -> english for testing wordpiece / model [False]")
flags.DEFINE_string("optimizer", "SGD", "Gradient descent optimizer [SGD]")

FLAGS = flags.FLAGS

tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)


def main(_):

    sys.stdout.flush()

    with tf.Session() as sess:
        attn = AttentionNN(FLAGS, sess)
        attn.build_model()
        attn.run()


if __name__ == "__main__":
    tf.app.run()





