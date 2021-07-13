# -*- coding:utf-8 -*-
import os
import sys
import glob
import pickle
import collections
import tensorflow as tf
from utils import *
slim = tf.contrib.slim

test_set = './example/'+sys.argv[1]+'_MCP_add.pkl'
training_start = 20
training_steps = 0
train_rate = 0.001
batch_size = 1
reuse = None
checkpoint_dir = './checkpoint_CM'
result_dir = './result'

class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    '''
    Example：Block('block1', bottleneck, [(256,64,1),(256,64,1),(256,64,2)])
    '''

def subsample(inputs, factor, scope=None, reuse=None):
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope, reuse=reuse)

@slim.add_arg_scope
def stack_blocks_dense(net, blocks, outputs_collections=None, reuse=None):
    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            for i, unit in enumerate(block.args):
                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    unit_depth, unit_depth_bottleneck, unit_stride = unit
                    net = block.unit_fn(net, depth=unit_depth, depth_bottleneck=unit_depth_bottleneck,
                                        stride=unit_stride)
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

    return net

def resnet_arg_scope(is_training=True,
                     weight_decay=0.000001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True): 

    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),  
            weights_initializer=slim.variance_scaling_initializer(),  
            activation_fn=tf.nn.relu,  
            normalizer_fn=slim.batch_norm,  
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc  

@slim.add_arg_scope
def bottleneck_1d(inputs, depth, depth_bottleneck, stride, reuse=None, outputs_collections=None, scope='1d_ResNet'):
    with tf.variable_scope(scope, 'bottleneck_1d', reuse=reuse) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4) 
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact1', reuse=reuse)

        if depth == depth_in:
            shortcut = subsample(inputs, stride, scope='shortcut', reuse=reuse)
        elif depth > depth_in:
            shortcut = tf.pad(inputs, [[0, 0], [0, 0], [0, 0], [0, depth - depth_in]], name='shortcut')
        else:
            shortcut = slim.conv2d(inputs, depth, [1, 1], stride=stride, normalizer_fn=None, activation_fn=None, scope='shortcut2',  reuse=reuse)
        residual = slim.conv2d(preact, depth_bottleneck, [1, 17], stride=1, scope='conv1', reuse=reuse)
        residual = slim.conv2d(residual, depth, [1, 17], stride=1, scope='conv2', normalizer_fn=None, activation_fn=None, reuse=reuse)
        output = shortcut + residual
        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)

@slim.add_arg_scope
def bottleneck_2d(inputs, depth, depth_bottleneck, stride, reuse=None, outputs_collections=None, scope='2d_ResNet'):
    with tf.variable_scope(scope, 'bottleneck_2d', [inputs], reuse=reuse) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact1', reuse=reuse)

        if depth == depth_in:
            shortcut = subsample(inputs, stride, scope='shortcut')
        elif depth > depth_in:
            shortcut = tf.pad(inputs, [[0, 0], [0, 0], [0, 0], [0, depth - depth_in]], name='shortcut')
        else:
            shortcut = slim.conv2d(inputs, depth, [1, 1], stride=stride, normalizer_fn=None, activation_fn=None, scope='shortcut', reuse=reuse)
        residual = slim.conv2d(preact, depth_bottleneck, [5, 5], stride=1, scope='conv1', reuse=reuse)
        residual = slim.conv2d(residual, depth, [5, 5], stride=1, scope='conv2', normalizer_fn=None, activation_fn=None, reuse=reuse)
        output = shortcut + residual
        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)

def seq2pair(inputs, append, mask, concat_shape, scope=None):
    with tf.variable_scope(scope, 'Pairwise', [inputs]):
        copies = 700
        bottom_layer = tf.concat([inputs] * copies, axis=1)
        bottom_layer = tf.boolean_mask(tf.transpose(bottom_layer, perm=[1, 0, 2, 3]), mask)
        bottom_layer = tf.transpose(bottom_layer, perm=[1, 0, 2, 3])

        concat_unit = tf.reshape(tf.concat([inputs, inputs], axis=3), concat_shape)
        concat_unit = tf.pad(concat_unit, ((0, 0), (0, 0), (0, 1400), (0, 0)), 'constant')
        middle_layer = tf.concat([concat_unit[:, :, i:i + copies, :] for i in range(copies)], axis=1)
        middle_layer = tf.boolean_mask(tf.transpose(middle_layer, perm=[1, 0, 2, 3]), mask)
        middle_layer = tf.boolean_mask(tf.transpose(middle_layer, perm=[2, 1, 0, 3]), mask)
        middle_layer = tf.transpose(middle_layer, perm=[1, 2, 0, 3])

        top_layer = tf.transpose(bottom_layer, perm=[0, 2, 1, 3])

        append = tf.boolean_mask(tf.transpose(append, perm=[1, 0, 2, 3]), mask)
        append = tf.boolean_mask(tf.transpose(append, perm=[2, 1, 0, 3]), mask)
        append = tf.transpose(append, perm=[1, 2, 0, 3])
        return tf.concat([bottom_layer, middle_layer, top_layer, append], axis=3, name=scope)

def resnet_seq(inputs, blocks, dims, num_classes=None, reuse=None, scope=None):
    with tf.variable_scope(scope, 'ResNet_CM', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, dims, stack_blocks_dense],
                            outputs_collections=end_points_collection):

            net = stack_blocks_dense(inputs, blocks, reuse=reuse) 

            if num_classes is not None:  
                net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,  
                                  normalizer_fn=None, scope='logits', reuse=reuse)  
            end_points = slim.utils.convert_collection_to_dict(end_points_collection) 
            if num_classes is not None:
                end_points['predictions'] = slim.softmax(net, scope='predictions') 
            return net, end_points

def resnet_seq_64_64(batch_size, inputs, append, mask, reuse=None, scope='Forbidden_Block'):
    inputs = tf.boolean_mask(tf.transpose(inputs, perm=[2, 1, 0, 3]), mask)
    inputs = tf.transpose(inputs, perm=[2, 1, 0, 3])
    blocks_1d = [Block('block_1d_1', bottleneck_1d, [(50, 50, 1)] * 2),
                 Block('block_1d_final', bottleneck_1d, [(50, 50, 1)])]
    seq_info, end_points = resnet_seq(inputs, blocks_1d, bottleneck_1d, num_classes=None, reuse=reuse, scope=scope+'_1d')

    concat_shape = [batch_size, 1, -1, 50] 
    pair_info = seq2pair(seq_info, append, mask, concat_shape)

    blocks_2d = [Block('block_2d_1', bottleneck_2d, [(75, 75, 1)] * 30),
                 Block('block_2d_Final', bottleneck_2d, [(1, 1, 1)])]
    result, end_points2 = resnet_seq(pair_info, blocks_2d, bottleneck_2d, num_classes=None, reuse=reuse, scope=scope+'_2d')
    return result

def loss_func(y_gt, y_p, mask):
    y_gt = tf.boolean_mask(tf.transpose(y_gt, perm=[1, 0, 2, 3]), mask)
    y_gt = tf.boolean_mask(tf.transpose(y_gt, perm=[2, 1, 0, 3]), mask)
    y_gt = tf.transpose(y_gt, perm=[1, 2, 0, 3])  
    with tf.name_scope('Loss'):
        lost_pix = tf.equal(y_gt, -1)  
        cross_entropy = tf.nn.weighted_cross_entropy_with_logits(y_gt, y_p, 8, name='cross_entropy')  
        zeros_temp = tf.multiply(y_gt, 0)
        ones_temp = zeros_temp + 1
        cross_entropy = tf.where(lost_pix, zeros_temp, cross_entropy) 
        loss = tf.reduce_sum(cross_entropy)  
        tf.summary.scalar('Loss', loss)  
        y_p_b = tf.greater(y_p, 0)  
        y_p_r = tf.where(y_p_b, ones_temp, zeros_temp)
        pp_r = tf.where(lost_pix, zeros_temp, y_p_r)
        pp_b = tf.cast(pp_r, tf.bool)
        tp_r = tf.where(pp_b, y_gt, zeros_temp) 
        accuracy = tf.reduce_sum(tp_r)/(tf.reduce_sum(pp_r)+1e-16)  
        tf.summary.scalar('Accuracy', accuracy)  
    return loss, accuracy

def modeling():

    with tf.Session() as sess:
        seq_length = 700
        x = tf.placeholder(tf.float32, [batch_size, 1, seq_length, 28])
        x2 = tf.placeholder(tf.float32, [batch_size, seq_length, seq_length, 3])
        y = tf.placeholder(tf.float32, [batch_size, seq_length, seq_length, 1])
        mask = tf.placeholder(tf.bool, [seq_length]) 
        with slim.arg_scope(resnet_arg_scope(is_training=True)):
            y_res = resnet_seq_64_64(batch_size, x, x2, mask, reuse=reuse)
        cross_entropy, accuracy = loss_func(y, y_res, mask)
        l2norm = tf.add_n(tf.losses.get_regularization_losses()) 
        tf.summary.scalar('L2-norm', l2norm)
        loss = cross_entropy + l2norm
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer(train_rate).minimize(loss)  
        train_grad = tf.gradients(y_res, x)
        g_list = tf.global_variables()
        saver = tf.train.Saver(var_list=g_list, max_to_keep=1000)
        merged = tf.summary.merge_all()
        with sess.as_default():
            ckptdir = os.path.join(checkpoint_dir,'batch_' + str(batch_size))
            seq_infos_valid = pkl_read(test_set)
            length_subs_valid = sort_len(seq_infos_valid) 
            one_train_step_valid = len(length_subs_valid) // batch_size
            tf.get_variable_scope()
            if os.path.exists(ckptdir):
                modeldir = os.path.join(ckptdir, 'Epoch_' + str(training_start))
                if os.path.exists(modeldir): 
                    try:
                        saver.restore(sess, os.path.join(modeldir, 'model.ckpt'))
                    except:
                        return 0
                else:
                    return 0
            else:
                os.makedirs(ckptdir)  
                tf.global_variables_initializer().run()
                    
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            for i in range(one_train_step_valid):
                train_in, train_append, maps, train_mask = sub2train_fl_CM(length_subs_valid[i * batch_size:(i + 1) * batch_size], seq_infos_valid, out_length=seq_length)
                test_out, test_grad = sess.run([y_res, train_grad], feed_dict={x: train_in, x2: train_append, y: maps, mask: train_mask})
                test_out = 1/(1+np.exp(-test_out))
                test_out = test_out.reshape((np.shape(test_out)[1],np.shape(test_out)[1]))
                for j in range(np.shape(test_out)[0]):
                    for k in range(np.shape(test_out)[0]):
                        if abs(k-j) < 6:
                            test_out[k][j] = 0

                np.savetxt('./result/'+length_subs_valid[i][2]+'_CM.txt', test_out)
    return 0

modeling()
