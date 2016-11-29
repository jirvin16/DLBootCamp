from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import h5py

import datetime
import os
import time
import sys
from pprint import pprint


class VariationalAutoencoder(object):
	def __init__(self, config, sess):

		self.sess = sess

		# Training network details
		self.batch_size            = config.batch_size
		self.current_learning_rate = config.init_learning_rate
		self.optimizer         	   = "Adagrad"
		self.loss                  = None
		self.optim 				   = None
		self.encode_mu 			   = None
		self.decode_mu 			   = None
		self.sample_mu 			   = None

		self.mnist 				   = input_data.read_data_sets('MNIST_data', one_hot=True)
		self.input_dim 			   = self.mnist.train.images.shape[1]
		self.N_z 				   = config.N_z
		self.epochs 			   = config.epochs
		self.is_test 			   = config.mode == 1 or config.mode == 2 or config.mode == 3
		self.manifold_visualize    = config.mode == 2
		self.scatter_visualize     = config.mode == 3
		if self.manifold_visualize:
			self.batch_size = 1
		elif self.scatter_visualize:
			self.batch_size = self.mnist.test.num_examples
		
		self.save_every 		   = config.save_every
		self.model_name 		   = config.model_name
		self.model_directory 	   = self.model_name
		self.checkpoint_directory  = os.path.join(self.model_directory, "checkpoints")
		self.log_directory 		   = os.path.join(self.model_directory, "logs")
		
		# Dimensions and initialization parameters
		self.std 				   = 0.01
		self.hidden_dim 	       = config.hidden_dim # size of hidden layers in encoder and decoder
		self.num_layers 	       = config.num_layers # number of hidden layers in encoder and decoder
		self.num_classes		   = 10

		# Model placeholders
		# self.batch 	       	  	   = tf.placeholder(tf.float32, shape=[self.batch_size, self.input_dim], name="batch")
		self.batch 	       	  	   = tf.placeholder(tf.float32, shape=[None, self.input_dim], name="batch")
		self.Z 					   = tf.placeholder(tf.float32, shape=[None, self.N_z], name="Z")

		if not os.path.isdir(self.model_directory):
			os.makedirs(self.model_directory)

		if not os.path.isdir(self.log_directory):
			os.makedirs(self.log_directory)

		if not os.path.isdir(self.checkpoint_directory):
			if self.is_test:
				raise Exception(" [!] Checkpoints directory %s not found" % self.checkpoint_directory)
			else:
				os.makedirs(self.checkpoint_directory)

		if self.is_test:
			self.outfile = os.path.join(self.model_directory, "test.out")
		else:
			self.outfile = os.path.join(self.model_directory, "train.out")

		with open(self.outfile, 'w') as outfile:
			pprint(config.__dict__['__flags'], stream=outfile)
			outfile.flush()
		


	def build_model(self):

		# Initialize weights
		W_initializer = tf.truncated_normal_initializer(stddev=self.std)
		b_initializer = tf.constant_initializer(0.1, dtype=tf.float32)

		self.weights = {}
		for weight_name in ["encoder", "decoder"]:
			input_dim = self.input_dim if weight_name == "encoder" else self.N_z
			output_dim = self.N_z if weight_name == "encoder" else self.input_dim
			self.weights["W_"+weight_name] = {
				"1"     	   : tf.get_variable("W_"+weight_name+"_1", shape=[input_dim, self.hidden_dim],
												 initializer=W_initializer),
				"mu" 		   : tf.get_variable("W_"+weight_name+"_mean", shape=[self.hidden_dim, output_dim],
												 initializer=W_initializer),
				"log_sigma_sq" : tf.get_variable("W_"+weight_name+"_sigma", shape=[self.hidden_dim, output_dim],
												 initializer=W_initializer)
			}
			for i in range(2, self.num_layers+1):
				self.weights["W_"+weight_name][str(i)] = tf.get_variable("W_"+weight_name+"_"+str(i), shape=[self.hidden_dim, self.hidden_dim],
												 	 				 	 initializer=W_initializer) 
			self.weights["b_"+weight_name] = {
				"mu" 		   : tf.get_variable("b_"+weight_name+"_mean", shape=[output_dim],
												 initializer=b_initializer),
				"log_sigma_sq" : tf.get_variable("b_"+weight_name+"_sigma", shape=[output_dim],
												 initializer=b_initializer)
			}
			for i in range(1, self.num_layers+1):
				self.weights["b_"+weight_name][str(i)] = tf.get_variable("b_"+weight_name+"_"+str(i), shape=[self.hidden_dim],
													 					 initializer=b_initializer)

		# Define encoder (recognition network) operations (gaussian MLP)
		encode_h = self.batch
		for i in range(1, self.num_layers+1):
			encode_h = tf.tanh(tf.matmul(encode_h, self.weights["W_encoder"][str(i)]) + self.weights["b_encoder"][str(i)])

		self.encode_mu = tf.matmul(encode_h, self.weights["W_encoder"]["mu"]) + self.weights["b_encoder"]["mu"]
		encode_log_sigma_sq = tf.matmul(encode_h, self.weights["W_encoder"]["log_sigma_sq"]) + self.weights["b_encoder"]["log_sigma_sq"]
		
		eps = tf.random_normal([self.batch_size, self.N_z])
		z = self.encode_mu + tf.mul(tf.sqrt(tf.exp(encode_log_sigma_sq)), eps)

		# Define decoder (generative network) operations (gaussian MLP)
		decode_h = z
		for i in range(1, self.num_layers+1):
			decode_h = tf.tanh(tf.matmul(decode_h, self.weights["W_decoder"][str(i)]) + self.weights["b_decoder"][str(i)])

		self.decode_mu = tf.sigmoid(tf.matmul(decode_h, self.weights["W_decoder"]["mu"]) + self.weights["b_decoder"]["mu"])

		# Kullbach Leibler Divergence
		KLD = -0.5 * tf.reduce_sum(1 + encode_log_sigma_sq - tf.square(self.encode_mu) - tf.exp(encode_log_sigma_sq), 1)

		# Reconstruction loss
		reconstruction_loss = -tf.reduce_sum(self.batch * tf.log(1e-10 + self.decode_mu) + \
											 (1-self.batch) * tf.log(1e-10 + 1 - self.decode_mu), 1)

		self.loss = tf.reduce_mean(KLD + reconstruction_loss)

		# only optimize if training
		if not self.is_test:
			self.optim = tf.contrib.layers.optimize_loss(self.loss, None, self.current_learning_rate, self.optimizer, 
														 	summaries=["learning_rate", "gradient_norm", "loss", "gradients"])
		else:
			sample_h = self.Z
			for i in range(1, self.num_layers+1):
				sample_h = tf.tanh(tf.matmul(sample_h, self.weights["W_decoder"][str(i)]) + self.weights["b_decoder"][str(i)])
			self.sample_mu = tf.sigmoid(tf.matmul(sample_h, self.weights["W_decoder"]["mu"]) + self.weights["b_decoder"]["mu"])
		
		self.sess.run(tf.initialize_all_variables())

		with open(self.outfile, "a") as outfile:
			for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
				print(var.name, file=outfile)
				print(var.get_shape(), file=outfile)	
				outfile.flush()

		self.saver = tf.train.Saver()

	def train(self):

		total_loss 			= 0.0

		merged_sum 			= tf.merge_all_summaries()
		t 	   	   			= datetime.datetime.now()
		writer     			= tf.train.SummaryWriter(os.path.join(self.log_directory, \
										    					  "{}-{}-{}-{}-{}-{}".format(t.year, t.month, t.day, t.hour % 12, t.minute, t.second)), \
										    					  self.sess.graph)

		current_epoch 	    = self.mnist.train.epochs_completed

		i 					= 0
		train_loss  		= 0.0
		num_batches  		= 0.0
		# previous_train_loss = float("inf")
		while current_epoch < self.epochs:

			batch, _ = self.mnist.train.next_batch(self.batch_size)

			feed     = {self.batch: batch}

			_, batch_loss, summary = self.sess.run([self.optim, self.loss, merged_sum], feed)

			train_loss += batch_loss
			
			if i % 10 == 0:
				writer.add_summary(summary, i)
			
			i += 1
			num_batches += 1.0

			if current_epoch < self.mnist.train.epochs_completed:

				state = {
					"train_loss" : train_loss / num_batches,
					"epoch" : current_epoch,
					"learning_rate" : self.current_learning_rate,
				}

				train_loss    = 0.0
				num_batches   = 0.0
				current_epoch = self.mnist.train.epochs_completed

				with open(self.outfile, 'a') as outfile:
					print(state, file=outfile)
					outfile.flush()

				# # Adaptive learning rate
				# if previous_train_loss <= train_loss + 1e-1:
				# 	self.current_learning_rate /= 2.

				# save model
				if (current_epoch % self.save_every == 0 or current_epoch == self.epochs - 1):
					self.saver.save(self.sess,
									os.path.join(self.checkpoint_directory, "MemN2N.model")
									)



	def test(self):

		# only load if in test mode (rather than cv)
		if self.is_test:
			import matplotlib.pyplot as plt
			self.load()

		if not self.manifold_visualize and not self.scatter_visualize:
			test_loss   = 0
			num_batches = 0.0
			while self.mnist.test.epochs_completed < 1:

				batch, _ 			  = self.mnist.test.next_batch(self.batch_size)	
				feed 				  = {self.batch: batch}
				loss, reconstructions = self.sess.run([self.loss, self.decode_mu], feed)
				test_loss 		     += loss

				# Reconstruction visualizations
				if self.is_test and num_batches == 0.0:
			
					num_samples = 6
					plt.figure(figsize=(12,18))
					for i in range(num_samples):
						plt.subplot(num_samples, 2, 2 * i + 1)
						plt.imshow(batch[i].reshape(28, 28), vmin=0, vmax=1)
						plt.title("Input")
						plt.colorbar()
						plt.subplot(num_samples, 2, 2 * i + 2)
						plt.imshow(reconstructions[i].reshape(28, 28), vmin=0, vmax=1)
						plt.title("Reconstruction")
						plt.colorbar()

					plt.tight_layout()
					plt.savefig(os.path.join(self.model_directory, "reconstructions.jpg"))
					plt.close()

				num_batches += 1.0

			state = {
				"test_loss" : test_loss / num_batches,
			}

			with open(self.outfile, 'a') as outfile:
				print(state, file=outfile)
				outfile.flush()

			return test_loss / num_batches
		
		if self.manifold_visualize and self.N_z == 2:
			# Manifold visualization
			x_dim 	 = 20
			y_dim 	 = 20
			x_values = np.linspace(-3, 3, 20)
			y_values = np.linspace(-3, 3, 20)
			canvas   = np.empty((28*y_dim, 28*x_dim))
			for i, x in enumerate(x_values):
				for j, y in enumerate(y_values):
					feed 	   = {self.Z: np.array([[x, y]])}
					sample_mu, = self.sess.run([self.sample_mu], feed)
					canvas[(x_dim-i-1)*28:(x_dim-i)*28, j*28:(j+1)*28] = sample_mu[0].reshape(28, 28)
			plt.figure(figsize=(12, 18))        
			Xi, Yi = np.meshgrid(x_values, y_values)
			plt.imshow(canvas, origin="upper")
			plt.title("MNIST Manifold")
			plt.tight_layout()
			plt.savefig(os.path.join(self.model_directory, "manifold.jpg"))
			plt.close()

		if self.scatter_visualize and self.N_z == 2:
			# Scatter plot
			x_batch, y_batch = self.mnist.test.next_batch(self.mnist.test.num_examples)
			feed 	   	  	 = {self.batch: x_batch}
			encode_mu, 	  	 = self.sess.run([self.encode_mu], feed)
			plt.figure(figsize=(12,18))
			plt.scatter(encode_mu[:, 0], encode_mu[:, 1], c=np.argmax(y_batch, 1))
			plt.colorbar()
			plt.savefig(os.path.join(self.model_directory, "scatter.jpg"))
			plt.close()

	def run(self):
		if self.is_test:
			self.test()
		else:
			self.train()

	def load(self):
		with open(self.outfile, 'a') as outfile:
			print(" [*] Reading checkpoints...", file=outfile)
			outfile.flush()
		ckpt = tf.train.get_checkpoint_state(self.checkpoint_directory)
		if ckpt and ckpt.model_checkpoint_path:
			self.saver.restore(self.sess, ckpt.model_checkpoint_path)
		else:
			raise Exception(" [!] Test mode but no checkpoint found")



