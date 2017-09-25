# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 17:19:23 2017

@author: subhajit
"""

import tensorflow as tf
import numpy as np
import sys, os
import argparse
from data_utils import get_data, get_character_mapping, vectorize, characterize


parser = argparse.ArgumentParser(description='Train the name generator')
parser.add_argument('--batch_size', type=int, default=128,
                   help='batch size for training')
parser.add_argument('--input_noise_dim', type=int, default=1,
                   help='dimension (integer) of the input noise')
parser.add_argument('--learning_rate', type=float, default=0.01,
                   help='learning rate (float) for training the GAN')
parser.add_argument('--dis_dropout_keep_prob', type=float, default=0.5,
                   help='probability (float) of keeping (1 - dropout) nodes in discriminator')
parser.add_argument('--minibatch_num_kernels', type=int, default=5,
                   help='number of kernels (integer) for minibatch discrimination')
parser.add_argument('--minibatch_kernel_dim', type=int, default=5,
                   help='dimension (integer) of the kernels for minibatch discrimination')
parser.add_argument('--loss_control_ratio', type=float, default=0.3,
                   help='threshold for gen_loss to dis_loss ratio (and vice versa), to stop training the one which becomes significantly stronger')
parser.add_argument('--max_iter', type=int, default=200,
                   help='number of iterations (integer) for training for each length')
parser.add_argument('--start_len', type=int, default=1,
                   help='length (integer) at which curriculum learning starts')
parser.add_argument('--end_len', type=int, default=10,
                   help='length (integer) at which curriculum learning stops')

args = parser.parse_args()

#set training parameters here
BATCH_SIZE = args.batch_size
INPUT_NOISE_DIM = args.input_noise_dim
LEARNING_RATE = args.learning_rate
MAX_ITER = args.max_iter
DIS_DROPOUT_KEEP_PROB = args.dis_dropout_keep_prob
MINIBATCH_NUM_KERNELS = args.minibatch_num_kernels
MINIBATCH_KERNEL_DIM = args.minibatch_kernel_dim
GEN_DIS_LOSS_CONTROL_RATIO = args.loss_control_ratio
START_LEN = args.start_len
END_LEN = args.end_len

# load the data and all
names = get_data()
char_indices, indices_char = get_character_mapping(names)


# open a csv file for storing the training log
fp = open(os.path.join(os.path.dirname(__file__), 'log.csv'), 'w')
fp.write('len,iter,sample_output,gen_loss,dis_loss\n')

print('initiating the training...')
# curriculum learning: start training on shortest strings first and then move to longer ones
input_dim = len(char_indices)+1
for maxlen in range(START_LEN, END_LEN):
    # convert names to vectors
    X = np.zeros((len(names), maxlen, len(char_indices)+1), dtype=np.float32)
    for i, name in enumerate(names):
        X[i] = vectorize(name, maxlen, char_indices)
    
    timesteps = maxlen
    
    # define the model
    tf.reset_default_graph()
    # input
    input_noise = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_NOISE_DIM], name='input_noise')
    input_repeat = [input_noise]*timesteps
    # generator
    with tf.variable_scope('generator'):
        gen_rnn_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(num_units=input_dim) for _ in range(2)])
        gen_layers0, gen_state = tf.contrib.rnn.static_rnn(cell=gen_rnn_cell, inputs=input_repeat, dtype=tf.float32)
        weights1 = tf.Variable(tf.random_normal([input_dim, input_dim]))
        biases1 = tf.Variable(tf.random_normal([input_dim]))
        gen_layers1 = [tf.matmul(l, weights1) + biases1 for l in gen_layers0]
        gen_layers1_activation = [tf.nn.tanh(l) for l in gen_layers1]
        weights2 = tf.Variable(tf.random_normal([input_dim, input_dim]))
        biases2 = tf.Variable(tf.random_normal([input_dim]))
        logits = [tf.matmul(l, weights2) + biases2 for l in gen_layers1_activation]
        logit = tf.stack(logits, name='generated_name')
    # discriminator
    is_real_label = tf.placeholder(dtype=tf.float32, shape=[None, 2])
    with tf.variable_scope('discriminator'):
        logits_transform1 = [tf.nn.tanh(l) for l in logits]
        w_minibatch = tf.Variable(tf.random_normal([input_dim, MINIBATCH_NUM_KERNELS*MINIBATCH_KERNEL_DIM]))
        #b_minibatch = tf.Variable(tf.random_normal([5*5]))
        logits_transform2 = [tf.reshape(tf.matmul(l, w_minibatch), [-1, MINIBATCH_NUM_KERNELS, MINIBATCH_KERNEL_DIM]) for l in logits_transform1]
        diffs = [tf.expand_dims(l, 3) - tf.expand_dims(tf.transpose(l, [1, 2, 0]), 0) for l in logits_transform2]
        abs_diffs = [tf.reduce_sum(tf.abs(d), 2) for d in diffs]
        minibatch_features = [tf.reduce_sum(tf.exp(-ad), 2) for ad in abs_diffs]
        concatenated_features = [tf.concat([l, mf], 1) for l, mf in zip(logits_transform1, minibatch_features)]
        dis_rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=2)
        features_dropout0 = [tf.nn.dropout(x=l, keep_prob=DIS_DROPOUT_KEEP_PROB) for l in concatenated_features]
        dis_layers1, dis_state = tf.contrib.rnn.static_rnn(cell=dis_rnn_cell, inputs=features_dropout0, dtype=tf.float32)
        true_or_fake = dis_state[-1]
    
    # define cost function and optimizers
    gen_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=true_or_fake, labels=is_real_label))
    dis_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=true_or_fake, labels=is_real_label))
    
    gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
    dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
    
    # learning rates for G and D should be different (TTUR: two time-scale update rule)
    gen_optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE)
    dis_optimizer = tf.train.RMSPropOptimizer(learning_rate=3*LEARNING_RATE)
    
    optim_gen = gen_optimizer.minimize(gen_cost, var_list=gen_vars)
    optim_dis = dis_optimizer.minimize(dis_cost, var_list=dis_vars)
    
    gen_gradients = gen_optimizer.compute_gradients(gen_cost, gen_vars)
    dis_gradients = dis_optimizer.compute_gradients(dis_cost, dis_vars)
    
    # initiate training
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # load model trained on previous length
    if maxlen > START_LEN:
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(os.path.dirname(__file__), 'models/name_generator_'+str(maxlen-1)))
    
    # start training
    count_iter = 0
    # flag variables to stop training when one becomes significantly stronger than the other
    train_gen = True
    train_dis = True
    for _ in range(int(MAX_ITER)):
        count_iter += 1
        if train_gen:
            loss_gen_sum = 0
        if train_dis:
            loss_dis_sum = 0
        num_batches = int(len(X)/BATCH_SIZE)
        
        # training
        for i in range(0, len(X), BATCH_SIZE):
            rand_smpl = np.random.choice(len(X), BATCH_SIZE)
            real_logits = [5*X[rand_smpl, j, :] for j in range(timesteps)]
            random_input = np.random.normal(size=(real_logits[0].shape[0], INPUT_NOISE_DIM), scale=3.0)
            gen_logits = sess.run(logits, feed_dict={input_noise: random_input})
            all_logits = [np.vstack((r,g)) for r, g in zip(real_logits, gen_logits)]
            # providing soft labels instead of hard binary labels
            real_labels = np.zeros((real_logits[0].shape[0], 2))
            real_labels[:, 0] = np.random.uniform(low=0.7, high=1.2, size=real_labels.shape[0])
            real_labels[:, 1] = np.random.uniform(low=0.0, high=0.3, size=real_labels.shape[0])
            gen_labels = np.zeros((gen_logits[0].shape[0], 2))
            gen_labels[:, 0] = np.random.uniform(low=0.0, high=0.3, size=gen_labels.shape[0])
            gen_labels[:, 1] = np.random.uniform(low=0.7, high=1.2, size=gen_labels.shape[0])
            all_labels = np.vstack((real_labels, gen_labels))
            
            if train_dis:
                feed_dict = {k: d for k, d in zip(logits, real_logits)}
                #feed_dict.update({k: d for k, d in zip(x, all_logits)})
                feed_dict.update({is_real_label: real_labels})
                _, loss_dis, dis_grad = sess.run([optim_dis, dis_cost, dis_gradients], feed_dict=feed_dict)
                #loss_dis_sum += loss_dis
                
                feed_dict = {k: d for k, d in zip(logits, gen_logits)}
                #feed_dict.update({k: d for k, d in zip(x, all_logits)})
                feed_dict.update({is_real_label: gen_labels})
                _, loss_dis, dis_grad = sess.run([optim_dis, dis_cost, dis_gradients], feed_dict=feed_dict)
                loss_dis_sum += loss_dis
            
            if train_gen:
                gen_labels[:, 0], gen_labels[:, 1] = gen_labels[:, 1].copy(), gen_labels[:, 0].copy()
                
                feed_dict = {k: d for k, d in zip(logits, gen_logits)}
                feed_dict.update({input_noise: random_input})
                feed_dict.update({is_real_label: gen_labels})
                _, loss_gen, gen_grad = sess.run([optim_gen, gen_cost, gen_gradients], feed_dict=feed_dict)
                loss_gen_sum += loss_gen
            
            sys.stdout.write('\riter:'+str(count_iter)+'\tgenerator loss: '+str(loss_gen)+
                             '\tdiscriminator loss: '+str(loss_dis))
        sys.stdout.write('\riter:'+str(count_iter)+'\tgenerator loss: '+str(loss_gen_sum/num_batches)+
                         '\tdiscriminator loss: '+str(loss_dis_sum/num_batches)+'\n')
        
        train_gen = False if loss_gen_sum/loss_dis_sum < GEN_DIS_LOSS_CONTROL_RATIO else True
        train_dis = False if loss_dis_sum/loss_gen_sum < GEN_DIS_LOSS_CONTROL_RATIO else True
        
        # testing
        pred = sess.run(logit, feed_dict={input_noise: random_input[0].reshape(1,-1)})
        test_decode = characterize(pred[:, 0, :], maxlen, indices_char)
        print(test_decode)
        
        fp.write(','.join([str(maxlen), str(count_iter), test_decode, str(loss_gen_sum/num_batches), str(loss_dis_sum/num_batches)])+'\n')
    
    # save the trained model
    print('saving the model for length '+str(maxlen))
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(os.path.dirname(__file__), 'models/name_generator_'+str(maxlen)))
    
    sess.close()

fp.close()