from __future__ import division
from __future__ import print_function

import random

import tensorflow as tf
import sys

from attention import AttentionNN

flags = tf.app.flags
 
flags.DEFINE_integer("max_size", 40, "Maximum sentence length [40]")
flags.DEFINE_integer("batch_size", 128, "Number of examples in minibatch [128]")
flags.DEFINE_integer("random_seed", 42, "Value of random seed [42]")
flags.DEFINE_integer("epochs", 120, "Number of epochs to run [120]")
flags.DEFINE_integer("hidden_dim", 512, "Size of hidden dimension [512]")
flags.DEFINE_integer("embedding_dim", 256, "Size of embedding dimension [256]")
flags.DEFINE_integer("num_layers", 2, "Number of recurrent layers [2]")
flags.DEFINE_float("init_learning_rate", 0.01, "initial learning rate [0.01]")
flags.DEFINE_float("grad_max_norm", 5., "gradient max norm [1]")
flags.DEFINE_boolean("attention", True, "Use attention [True]")
flags.DEFINE_float("dropout", 0.2, "Dropout [0.2]")
flags.DEFINE_integer("mode", 0, "0 for training, 1 for testing, 2 for sampling [0]")
flags.DEFINE_boolean("validate", True, "True for cross validation, False otherwise [True]")
flags.DEFINE_integer("save_every", 10, "Save every [10] epochs")
flags.DEFINE_boolean("sample", False, "Use sample data [False]")
flags.DEFINE_string("model_name", "out", "model name for prefix to checkpoint file [unnamed]")
flags.DEFINE_boolean("unilingual", False, "use unilingual french -> english for testing wordpiece / model [False]")
flags.DEFINE_string("optimizer", "Adam", "Gradient descent optimizer [Adam]")
flags.DEFINE_boolean("bidirectional", True, "Use bidirectional first layer in encoder [True]")
flags.DEFINE_boolean("keep_training", False, "Restore checkpoint and continye training [False]")

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





