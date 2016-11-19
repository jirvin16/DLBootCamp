from __future__ import division
from __future__ import print_function

import random

import tensorflow as tf
import sys

from autoencoder import VariationalAutoencoder

flags = tf.app.flags
 
flags.DEFINE_integer("batch_size", 100, "Number of examples in minibatch [100]")
flags.DEFINE_integer("random_seed", 123, "Value of random seed [123]")
flags.DEFINE_integer("hidden_dim", 500, "Size of hidden dimension [500]")
flags.DEFINE_integer("N_z", 2, "Dimension of latent space (must be 2 or 20) [2]")
flags.DEFINE_integer("num_layers", 3, "Number of recurrent layers [3]")
flags.DEFINE_float("init_learning_rate", 0.01, "initial learning rate [0.01]")
flags.DEFINE_integer("mode", 0, "0 for training, 1 for testing [0]")
flags.DEFINE_string("model_name", "out", "model name for prefix to checkpoint file [unnamed]")

FLAGS = flags.FLAGS

tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)


def main(_):

    with tf.Session() as sess:
        attn = VariationalAutoencoder(FLAGS, sess)
        attn.build_model()
        attn.run()


if __name__ == "__main__":
    tf.app.run()





