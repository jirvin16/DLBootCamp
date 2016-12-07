from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import rnn
import numpy as np
from length_analysis import process_files

import data
import datetime
import os
import time
import sys
from pprint import pprint


class AttentionNN(object):
	def __init__(self, config, sess):

		self.sess = sess

		# Training details
		self.batch_size            = config.batch_size
		self.max_size              = config.max_size
		self.epochs                = config.epochs
		self.current_learning_rate = config.init_learning_rate
		self.grad_max_norm 		   = config.grad_max_norm
		self.use_attention 		   = config.use_attention
		self.dropout 			   = config.dropout
		self.optimizer 			   = config.optimizer
		self.loss                  = None
		self.optim 				   = None

		self.data_directory 	   = "/deep/group/dlbootcamp/jirvin16/final_data/"
		self.is_test 			   = config.mode == 1
		self.is_sample 			   = config.mode == 2
		self.validate 			   = config.validate
		self.probabilities 		   = []
		self.alignments 		   = []
		
		self.save_every 		   = config.save_every
		self.model_name 		   = config.model_name
		self.model_directory 	   = self.model_name
		self.checkpoint_directory  = os.path.join(self.model_directory, "checkpoints")
		
		# Dimensions and initialization parameters
		self.init_min              = -0.1
		self.init_max 		       = 0.1
		self.hidden_dim 	       = config.hidden_dim
		self.embedding_dim         = config.embedding_dim		
		self.num_layers 	       = config.num_layers

		# Model placeholders
		self.source_batch 	       = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_size], name="source_batch")
		self.target_batch 	       = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_size], name="target_batch")
		self.dropout_var 	       = tf.placeholder(tf.float32, name="dropout_var")
		self.sequence_loss_weights = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_size - 1], name="sequence_loss_weights")
		
		if self.is_test:
			self.dropout = 0

		if not os.path.isdir(self.data_directory):
			raise Exception(" [!] Data directory %s not found" % self.data_directory)

		if not os.path.isdir(self.model_directory):
			os.makedirs(self.model_directory)

		if not os.path.isdir(self.checkpoint_directory):
			if self.is_test or self.is_sample:
				raise Exception(" [!] Checkpoints directory %s not found" % self.checkpoint_directory)
			else:
				os.makedirs(self.checkpoint_directory)

		if self.is_test:
			self.outfile  = os.path.join(self.model_directory, "test.out")
			self.bleu_outfile = os.path.join(self.model_directory, "bleu.out")
		elif self.is_sample:
			self.outfile = os.path.join(self.model_directory, "sample.out")
		else:
			self.outfile = os.path.join(self.model_directory, "train.out")

		with open(self.outfile, 'w') as outfile:
			pprint(config.__dict__['__flags'], stream=outfile)
			outfile.flush()

		# Data paths
		if config.sample:
			sample_prefix 		= "small_sample."
		else:
			sample_prefix 		= ""
		if config.use_unilingual:
			self.data_directory = "/deep/group/dlbootcamp/jirvin16/final_sample/"
			source_suffix 	 	= "fr"
			target_suffix 	 	= "en"
			test_target_suffix  = "en"
		else:
			source_suffix 	 	= "fr_en"
			target_suffix 	    = "en_de"
			test_target_suffix  = "de"

		self.train_source_data_path    = os.path.join(self.data_directory, sample_prefix + "train." + source_suffix)
		self.train_target_data_path    = os.path.join(self.data_directory, sample_prefix + "train." + target_suffix)
		if self.is_test:
			self.test_source_data_path = os.path.join(self.data_directory, "test.fr")
			self.test_target_data_path = os.path.join(self.data_directory, "test." + test_target_suffix)
		elif self.is_sample:
			self.test_source_data_path = os.path.join(self.model_directory, "example.txt")
			if not os.path.exists(self.test_source_data_path):
				raise Exception("If in sample mode, must create a file called 'example.txt'")
			# unused
			self.test_target_data_path = os.path.join(self.model_directory, "example.txt")
		else:
			self.test_source_data_path = os.path.join(self.data_directory, sample_prefix + "valid." + source_suffix)
			self.test_target_data_path = os.path.join(self.data_directory, sample_prefix + "valid." + target_suffix)


		self.vocab_index, self.index_vocab = data.read_vocabulary(os.path.join(self.data_directory, "segmented_all"))
		self.vocab_size    				   = len(self.vocab_index.keys())

	def build_model(self):
		initializer = tf.random_uniform_initializer(minval=self.init_min, maxval=self.init_max)

		with tf.variable_scope("encoding"):

			self.source_embed     = tf.get_variable("source_embed", shape=[self.vocab_size, self.embedding_dim],
													initializer=initializer)
			self.bidir_proj       = tf.get_variable("bidir_proj", shape=[2*self.hidden_dim, self.hidden_dim],
													initializer=initializer)
			self.bidir_proj_bias  = tf.get_variable("bidir_proj_bias", shape=[self.hidden_dim],
													initializer=initializer)
			encode_lstm_fw        = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim, 
																 state_is_tuple=True)
			encode_lstm_fw 		  = tf.nn.rnn_cell.DropoutWrapper(encode_lstm_fw, 
															      output_keep_prob=1-self.dropout_var)
			encode_lstm_bw        = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim, 
																 state_is_tuple=True)
			encode_lstm_bw 		  = tf.nn.rnn_cell.DropoutWrapper(encode_lstm_bw, 
															      output_keep_prob=1-self.dropout_var)
			if self.num_layers > 1:
				
				encode_lstm       = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim, 
																 state_is_tuple=True)
				encode_lstm 	  = tf.nn.rnn_cell.DropoutWrapper(encode_lstm, 
															      output_keep_prob=1-self.dropout_var)
				self.encoder 	  = tf.nn.rnn_cell.MultiRNNCell([encode_lstm] * (self.num_layers - 1), 
															    state_is_tuple=True)
			
		with tf.variable_scope("decoding"):

			self.target_embed 	  = tf.get_variable("target_embed", shape=[self.vocab_size, self.embedding_dim], 
													initializer=initializer)
			self.target_proj 	  = tf.get_variable("target_proj", shape=[self.embedding_dim, self.hidden_dim], 
													initializer=initializer)
			self.target_proj_bias = tf.get_variable("target_proj_bias", shape=[self.hidden_dim], 
													initializer=initializer)

			decode_lstm 		  = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim, 
															 state_is_tuple=True)
			decode_lstm 		  = tf.nn.rnn_cell.DropoutWrapper(decode_lstm, 
															  output_keep_prob=1-self.dropout_var)
			self.decoder 		  = tf.nn.rnn_cell.MultiRNNCell([decode_lstm] * self.num_layers,
																state_is_tuple=True)

			self.W_embed 		  = tf.get_variable("W_embed", shape=[self.hidden_dim, self.embedding_dim], 
										   			initializer=initializer)
			self.b_embed 		  = tf.get_variable("b_embed", shape=[self.embedding_dim], 
										   			initializer=initializer)
			self.W_proj 		  = tf.get_variable("W_proj", shape=[self.embedding_dim, self.vocab_size], 
										  			initializer=initializer)
			self.b_proj 		  = tf.get_variable("b_proj", shape=[self.vocab_size], 
										  			initializer=initializer)

			if self.use_attention:

				self.Wc = tf.get_variable("W_context", shape=[2 * self.hidden_dim, self.hidden_dim], 
										  initializer=initializer)
				self.bc = tf.get_variable("b_context", shape=[self.hidden_dim], 
										  initializer=initializer)

			# feed in previous outputs if testing or sampling
			# if training, feed in true values
			if not self.is_test and not self.is_sample:
				target_embeddings = tf.split(1, self.max_size, tf.nn.embedding_lookup(self.target_embed, self.target_batch))
			else:
				test_indices = [self.vocab_index["<s>"] for _ in range(self.batch_size)]
				test_embeddings = tf.nn.embedding_lookup(self.target_embed, test_indices)

		with tf.variable_scope("encoding"):
			source_embeddings = tf.split(1, self.max_size, tf.nn.embedding_lookup(self.source_embed, self.source_batch))
			source_embeddings = [tf.squeeze(source_embeddings[t], [1]) for t in range(self.max_size)]
			outputs, output_state_fw, output_state_bw = rnn.bidirectional_rnn(encode_lstm_fw, encode_lstm_bw, source_embeddings, dtype=tf.float32)
			source_hidden_states = []
			for t in xrange(self.max_size):
				projection = tf.matmul(outputs[t], self.bidir_proj) + self.bidir_proj_bias
				source_hidden_states.append(projection)
			
			if self.num_layers > 1:
				initial_hidden_state = self.encoder.zero_state(self.batch_size, dtype=tf.float32)
				hidden_state = initial_hidden_state
				new_source_hidden_states = []
				for t in xrange(self.max_size):
					if t >= 1:
						tf.get_variable_scope().reuse_variables()
					output, hidden_state = self.encoder(source_hidden_states[t], hidden_state)
					new_source_hidden_states.append(output)
				source_hidden_states = new_source_hidden_states

		packed_source_hidden_states = tf.pack(source_hidden_states)

		scores = []

		# for sampling
		probabilities = []
		alignments = []
			
		with tf.variable_scope("decoding"):

			for t in xrange(self.max_size):
				if t >= 1:
					tf.get_variable_scope().reuse_variables()
				
				if not self.is_test and not self.is_sample:
					projection = tf.matmul(tf.squeeze(target_embeddings[t]), self.target_proj) + self.target_proj_bias
				else:
					projection = tf.matmul(tf.squeeze(test_embeddings), self.target_proj) + self.target_proj_bias

				# First h is no longer h output by encoder
				# zero out that hidden state as in paper
				initial_hidden_state = self.decoder.zero_state(self.batch_size, dtype=tf.float32)
				output, hidden_state = self.decoder(projection, initial_hidden_state)

				if self.use_attention:

					attention_scores = tf.reduce_sum(tf.mul(output, packed_source_hidden_states), 2) # (M, B)

					a_t       = tf.nn.softmax(tf.transpose(attention_scores))

					alignments.append(a_t) # (B, M)

					c_t       = tf.batch_matmul(tf.transpose(packed_source_hidden_states, perm=[1, 2, 0]), tf.expand_dims(a_t, 2))

					h_tilde_t = tf.tanh(tf.matmul(tf.concat(1, [tf.squeeze(c_t, [2]), output]), self.Wc) + self.bc)

				else:

					h_tilde_t = output

				embed = tf.matmul(h_tilde_t, self.W_embed) + self.b_embed
				score = tf.matmul(embed, self.W_proj) + self.b_proj 

				# if sampling or testing, compute probabilities for bleu score
				if self.is_test or self.is_sample:
					
					probability     = tf.nn.softmax(score)
					probabilities.append(probability)
					test_indices    = tf.to_int32(tf.argmax(probability, 1))
					test_embeddings = tf.nn.embedding_lookup(self.target_embed, test_indices)

				scores.append(score)

		# dont compute loss if sampling
		if not self.is_sample:
			logits = scores[:-1]
			targets = tf.split(1, self.max_size, self.target_batch)[1:]
			sequence_loss_weights = [tf.squeeze(x) for x in tf.split(1, self.sequence_loss_weights.get_shape()[1], self.sequence_loss_weights)]
			# sequence_loss_weights = [tf.ones([self.batch_size]) for _ in xrange(self.max_size - 1)]
			self.loss = tf.nn.seq2seq.sequence_loss(logits, targets, sequence_loss_weights, average_across_timesteps=True)
		
		# if sampling or testing, compute probabilities for bleu score
		if self.is_sample or self.is_test:
			self.probabilities = tf.transpose(tf.pack(probabilities), [1, 0, 2])
			self.alignments = tf.transpose(tf.pack(alignments), [1, 0, 2]) # (M, B, M) -> (B, M, M)

		# only optimize if training
		if not self.is_test and not self.is_sample:
			self.optim = tf.contrib.layers.optimize_loss(self.loss, None, self.current_learning_rate, self.optimizer, clip_gradients=self.grad_max_norm, 
														 summaries=["learning_rate", "gradient_norm", "loss", "gradients"])
		
		self.sess.run(tf.initialize_all_variables())

		with open(self.outfile, "a") as outfile:
			for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
				print(var.name, file=outfile)
				print(var.get_shape(), file=outfile)	
				outfile.flush()

		self.saver = tf.train.Saver()

	def train(self):

		total_loss = 0.0

		merged_sum = tf.merge_all_summaries()
		t 	   	   = datetime.datetime.now()
		writer     = tf.train.SummaryWriter(os.path.join(self.model_directory, \
										    "logs", "{}-{}-{}-{}-{}".format(t.year, t.month, t.day, t.hour % 12, t.minute)), \
										    self.sess.graph)

		i 					= 0
		best_valid_loss 	= float("inf")
		for epoch in xrange(self.epochs):

			train_loss  = 0.0
			num_batches = 0
			for source_batch, target_batch, target_length in data.data_iterator(self.train_source_data_path, self.train_target_data_path, \
																				self.vocab_index, self.max_size, self.batch_size):

				# sequence_loss_weights = [[1.0 if j < t_l else 1.0 for j in range(self.max_size - 1)] for t_l in target_length]
				sequence_loss_weights = [[1.0 if j < t_l else 0.0 for j in range(self.max_size - 1)] for t_l in target_length]
				
				feed = {self.source_batch: source_batch, self.target_batch: target_batch, 
						self.dropout_var: self.dropout, self.sequence_loss_weights: sequence_loss_weights}

				_, batch_loss, summary = self.sess.run([self.optim, self.loss, merged_sum], feed)

				train_loss += batch_loss
				
				if i % 50 == 0:
					writer.add_summary(summary, i)
					with open(self.outfile, 'a') as outfile:
						print(batch_loss, file=outfile)
						outfile.flush()
				
				i += 1
				num_batches += 1

			perplexity = np.exp(train_loss / num_batches)

			state = {
				"train_loss" : train_loss / num_batches,
				"train_perplexity" : perplexity,
				"epoch" : epoch,
				"learning_rate" : self.current_learning_rate,
			}

			with open(self.outfile, 'a') as outfile:
				print(state, file=outfile)
				outfile.flush()
			
			if self.validate:
				valid_loss = self.test()

				# if validation loss increases, halt training
				# model in previous epoch will be saved in checkpoint
				if valid_loss > best_valid_loss:
					if tolerance >= 10:
						break
					else:
						tolerance += 1
				# save model after validation check
				else:
					tolerance = 0
					self.saver.save(self.sess,
									os.path.join(self.checkpoint_directory, "MemN2N.model")
									)
					best_valid_loss = valid_loss
			
			else:
				if epoch % self.save_every == 0:
					self.saver.save(self.sess,
									os.path.join(self.checkpoint_directory, "MemN2N.model")
									)

	def test(self):

		# only load if in test mode (rather than cv)
		if self.is_test:
			self.load()

		test_loss = 0

		num_batches = 0
		for source_batch, target_batch, target_length in data.data_iterator(self.test_source_data_path, self.test_target_data_path, \
																			self.vocab_index, self.max_size, self.batch_size):

			sequence_loss_weights = [[1.0 if i < t_l else 0.0 for i in range(self.max_size - 1)] for t_l in target_length]

			feed = {self.source_batch: source_batch, self.target_batch: target_batch, 
					self.dropout_var: 0.0, self.sequence_loss_weights:sequence_loss_weights}

			loss, = self.sess.run([self.loss], feed)

			if num_batches % 50 == 0:
				with open(self.outfile, 'a') as outfile:
					print(loss, file=outfile)
					outfile.flush()

			test_loss += loss

			num_batches += 1

		perplexity = np.exp(test_loss / num_batches)

		state = {
			"test_loss" : test_loss / num_batches,
			"test_perplexity" : perplexity,
		}

		with open(self.outfile, 'a') as outfile:
			print(state, file=outfile)
			outfile.flush()

		if self.is_test:
			
			self.sample()
			
			process_files(self.predictions_file, self.truth_file, self.bleu_outfile)

		return test_loss / num_batches


	def sample(self):
		
		# only load model again if we are sampling
		if self.is_sample:
			self.load()

		with open(self.test_source_data_path) as sample_data:
			sample_size = sum(1 for line in sample_data if len(line.replace("\n", "").split()) < self.max_size)

		num_batches = int(np.ceil(sample_size / self.batch_size))

		if self.is_test:
			self.predictions_file = os.path.join(self.model_directory, "predictions.txt")
			self.truth_file       = os.path.join(self.model_directory, "truth.txt")
		else:
			self.predictions_file = os.path.join(self.model_directory, "example.vi")
			self.truth_file 	  = os.path.join(self.model_directory, "ignore.txt")

		with open(self.predictions_file, 'w') as predictions, open(self.truth_file, 'w') as truth:

			for source_batch, target_batch, _ in data.data_iterator(self.test_source_data_path, self.test_target_data_path, \
														 self.vocab_index, self.max_size, self.batch_size):

				# unused
				psuedo_sequence_loss_weights = [[-1 for _ in range(self.max_size - 1)] for _ in range(len(source_batch))]
				pseudo_target_batch   		 = [[-1 for _ in range(self.max_size)] for _ in range(self.batch_size)]

				feed = {self.source_batch: source_batch, self.target_batch: pseudo_target_batch, 
						self.dropout_var: 0.0, self.sequence_loss_weights:psuedo_sequence_loss_weights}

				probabilities, alignments, = self.sess.run([self.probabilities, self.alignments], feed)

				# iterate through the batch examples
				assert probabilities.shape[0] == alignments.shape[0]
				for j in range(probabilities.shape[0]):

					# source_indices  = source_batch[j]
					# source_sentence = [self.source_index_vocab[i] for i in source_indices]
					# print(" ".join(source_sentence))

					target_probs    = probabilities[j]
					target_indices  = np.argmax(target_probs, 1)
					target_sentence = []
					k = 0
					for i in target_indices:
						next_word   = self.target_index_vocab[i]
						if next_word != "</s>":
							target_sentence.append(next_word)
							if self.is_sample:
								alignment_scores = alignments[j][k]
								k += 1
								print(" ".join([str(x) for x in alignment_scores]), file=predictions)
								predictions.flush()
						else:
							break
					if self.is_sample:
						print("SENTENCE ", end="", file=predictions)
						predictions.flush()

					print(" ".join(target_sentence), file=predictions)
					predictions.flush()

					if self.is_test:
						true_indices  = target_batch[j]
						true_sentence = [self.target_index_vocab[i] for i in true_indices \
										 if self.target_index_vocab[i] not in ["<pad>", "</s>", "<s>"]]
						print(" ".join(true_sentence), file=truth)
						truth.flush()

					# elif self.is_sample:
					# 	assert alignments.shape[1] == self.max_size
					# 	for k in range(self.max_size):
					# 		alignment_scores = alignments[j][k]
					# 		assert alignment_scores.shape[0] == self.max_size
					# 		print(",".join([str(x) for x in alignment_scores]), file=predictions)



	def run(self):
		if self.is_test:
			self.test()
		elif self.is_sample:
			self.sample()
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



