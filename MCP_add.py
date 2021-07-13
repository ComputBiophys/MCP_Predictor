# -*- coding:utf-8 -*-
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

from utils import *

training_steps = 0
n_hidden = 200
keep_prob = 0.7
batch_size = 1
train_rate_initial=0.001
test_set = './example/'+sys.argv[1]+'_CM.pkl'
checkpoint_dir='./checkpoint_MCP'
result_dir='./example'

def conv2d(x, W, b):
    x = tf.reshape(x, shape=[-1, 700, 26])
    x = tf.nn.conv1d(x, W, 1, padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def multi_conv(x, weights, biases):
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv2 = conv2d(x, weights['wc2'], biases['bc2'])
    conv3 = conv2d(x, weights['wc3'], biases['bc3'])
    convs = tf.concat([conv1, conv2, conv3], 2)
    return convs

def BiRNN(x, weights, biases, seq_length):
    stacked_fw_rnn, stacked_bw_rnn = [], []
    for i in range(3):
        lstm_fw_cell = tf.contrib.rnn.GRUCell(n_hidden)
        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        stacked_fw_rnn.append(lstm_fw_cell)
        lstm_bw_cell = tf.contrib.rnn.GRUCell(n_hidden)
        lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        stacked_bw_rnn.append(lstm_bw_cell)
    mcell_fw = tf.contrib.rnn.MultiRNNCell(stacked_fw_rnn, state_is_tuple=True)
    mcell_bw = tf.contrib.rnn.MultiRNNCell(stacked_bw_rnn, state_is_tuple=True)
    init_state_fw = mcell_fw.zero_state(1, dtype=tf.float32)
    init_state_bw = mcell_bw.zero_state(1, dtype=tf.float32)
    output, states = tf.nn.bidirectional_dynamic_rnn(mcell_fw, mcell_bw, x, sequence_length=seq_length, time_major=False, initial_state_fw=init_state_fw, initial_state_bw=init_state_bw, dtype=tf.float32)
    lstmoutputs = tf.concat(output, 2)
    lstmoutputs = tf.concat([lstmoutputs, x], 2)
    outputs = tf.reshape(lstmoutputs, [-1, 2 * n_hidden + 192])
    return tf.matmul((tf.matmul(outputs, weights['out']) + biases['out']), weights['out2']) + biases['out2']

weights = {
    'wc1': tf.Variable(tf.truncated_normal([3, 26, 64])),
    'wc2': tf.Variable(tf.truncated_normal([7, 26, 64])),
    'wc3': tf.Variable(tf.truncated_normal([9, 26, 64])),
    'out': tf.Variable(tf.truncated_normal([ 2 * n_hidden  + 192, 1])),
    'out2': tf.Variable(tf.truncated_normal([1, 1]))
}

biases = {
    'bc1': tf.Variable(tf.truncated_normal([64])),
    'bc2': tf.Variable(tf.truncated_normal([64])),
    'bc3': tf.Variable(tf.truncated_normal([64])),
    'out': tf.Variable(tf.truncated_normal([1])),
    'out2': tf.Variable(tf.truncated_normal([1]))
}

def loss_func(y, pred):
	loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred)
	return loss

def data_pack(seq_infos, predictions, length_subs, package):
        for i, length_sub in enumerate(length_subs):
                temp_del = predictions
                temp = {b'name': seq_infos[length_sub[1]][b'name'],
                                b'sequence': seq_infos[length_sub[1]][b'sequence'],
                                b'PSSM': seq_infos[length_sub[1]][b'PSSM'],
                                b'SS3': seq_infos[length_sub[1]][b'SS3'],
                                b'ACC': seq_infos[length_sub[1]][b'ACC'],
                                b'ccmpredZ': seq_infos[length_sub[1]][b'ccmpredZ'],
                                b'OtherPairs': seq_infos[length_sub[1]][b'OtherPairs'],
                                b'contactMatrix': seq_infos[length_sub[1]][b'contactMatrix'],
                                b'lipidcontact': temp_del,
                                b'lipidcontact2': temp_del}
                package.append(temp)
        return package

def modeling():

	with tf.Session() as sess:
		x = tf.placeholder(tf.float32, [batch_size, 700, 26])
		y = tf.placeholder(tf.float32, [batch_size, 700, 1])
		seq_length = tf.placeholder("int32", [None])
		mask_weight = tf.placeholder(tf.float32, [None, 700])
		mask = tf.reshape(mask_weight, [-1])		
		x_convs = multi_conv(x, weights, biases)
		pred = BiRNN(x_convs, weights, biases, seq_length)
		global_step = training_steps
		pred = tf.sigmoid(pred, name=None)
		train_rate = tf.train.exponential_decay(train_rate_initial, global_step=training_steps, decay_steps=10, decay_rate=0.99)		
		g_list = tf.global_variables()
		cross_entropy = tf.reduce_sum(tf.pow((pred-y),2))
		l2norm = 0.0001*tf.reduce_sum([ tf.nn.l2_loss(v) for v in g_list ])
		mean_loss = cross_entropy + l2norm
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			train_step = tf.train.AdamOptimizer(train_rate).minimize(mean_loss)  
		train_grad = tf.gradients(pred, x)
		saver = tf.train.Saver(var_list=g_list, max_to_keep=1000)
		with sess.as_default():
			seq_infos_valid = pkl_read(test_set)
			length_subs_valid = sort_len(seq_infos_valid) 
			one_train_step_valid = len(length_subs_valid) // batch_size
			tf.get_variable_scope()
			if os.path.exists(checkpoint_dir):
				modeldir = os.path.join(checkpoint_dir, 'Epoch')
				if os.path.exists(modeldir):  
					try:
						saver.restore(sess, os.path.join(modeldir, 'model.ckpt'))
						print('Loading the checkpoint: ' + os.path.join(modeldir, 'model.ckpt'))
					except:
						print('The check point directory does not exist.\nTest failed.')
						return 0
				else:
					print('The check point directory does not exist.\nTest failed.')
					return 0
			else:
				os.makedirs(ckptdir)  
				tf.global_variables_initializer().run()
				print('This is a new training.\nInitializing new parameters.')

			if not os.path.exists(result_dir):
 				os.makedirs(result_dir)

			print('Start testing!...')
			package = []
			for i in range(one_train_step_valid):
				sequence, train_in = sub2train_fl(length_subs_valid[i * batch_size:(i + 1) * batch_size], seq_infos_valid, out_length=700)
				seq_len = [1] * batch_size					
				mask = np.ones((batch_size, 700))
				for f,length in enumerate(seq_len):
					mask[f, length:] = 0	
				mask_wei = mask
				test_out = sess.run(pred, feed_dict={x: train_in[:,:,:26], seq_length: seq_len, mask_weight: mask_wei})
				pred_out = test_out[0:length_subs_valid[i][0]]
				package = data_pack(seq_infos_valid, pred_out, length_subs_valid[i * batch_size:(i + 1) * batch_size], package)

			pkl_save(package, os.path.join(result_dir, sys.argv[1]+'_MCP_add.pkl'))
	return 0

modeling()
