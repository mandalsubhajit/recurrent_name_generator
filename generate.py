# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 21:41:48 2017

@author: subhajit
"""

import tensorflow as tf
import numpy as np
import os
from data_utils import get_data, get_character_mapping, vectorize, characterize

# load the data and all
names = get_data()
char_indices, indices_char = get_character_mapping(names)

# generate names of a desired length
# os.path.dirname(__file__)
def generate(length, session=None, noise_input=None, how_many=1):
    if not session:
        sess=tf.Session()    
        #First let's load meta graph and restore weights
        saver = tf.train.import_meta_graph(os.path.join(os.path.dirname(__file__), 'models/name_generator_'+str(length)+'.meta'))
        saver.restore(sess, os.path.join(os.path.dirname(__file__), 'models/name_generator_'+str(length)))
    else:
        sess = session
    
    graph = tf.get_default_graph()
    logit = graph.get_tensor_by_name('generator/generated_name:0')
    input_noise = graph.get_tensor_by_name('input_noise:0')
    
    if noise_input is None:
        rand_input = np.random.normal(size=(how_many, input_noise.get_shape()[1]), scale=3.0)
    else:
        rand_input = noise_input
    
    if noise_input is None and how_many == 1:
        pred = sess.run(logit, feed_dict={input_noise: rand_input.reshape(1,-1)})
        generated = characterize(pred[:, 0, :], length, indices_char)
    elif noise_input is not None or how_many > 1:
        pred = sess.run(logit, feed_dict={input_noise: rand_input})
        generated = [characterize(pred[:, i, :], length, indices_char) for i in range(pred.shape[1])]
    else:
        return []
    
    if not session:
        sess.close()
    return generated

if __name__ == '__main__':
    n = np.hstack([np.zeros((100, 1)), np.array([[i for i in range(-50,50)]]).reshape(-1,1), np.zeros((100, 8))])
    print(generate(2, None, n))
    
    
    sess=tf.Session()    
    #First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph(os.path.join(os.path.dirname(__file__), 'models/name_generator_2.meta'))
    saver.restore(sess, os.path.join(os.path.dirname(__file__), 'models/name_generator_2'))
    for _ in range(3):
        print(generate(2, sess, n))
    sess.close()