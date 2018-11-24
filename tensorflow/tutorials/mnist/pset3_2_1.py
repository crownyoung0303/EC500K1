# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
sess = tf.InteractiveSession()

FLAGS = None

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def mlp(x=tf.placeholder(tf.float32, [None, 784]), hidden_sizes= [64, 10], activation_fn= tf.nn.relu):
  h_in = x
  w= []
  b= []
  io_sizes=[784] + hidden_sizes

  for i in range(len(io_sizes)-1):
    with tf.variable_scope("layer"+str(i)):
            weights= tf.get_variable("weights", [io_sizes[i], io_sizes[i+1]], initializer=tf.random_normal_initializer(stddev= 0.1))
            biases = tf.get_variable("biases", [io_sizes[i+1]], initializer=tf.constant_initializer(0.1))
            h_out=activation_fn(tf.matmul(h_in, weights) + biases)
            h_in= h_out
        #if i == len(hidden_sizes)-1:
           #weights= tf.get_variable("weights", [int(28 * 28 / (2**(i+1))) * io_sizes[i], io_sizes[i+1]], initializer=tf.random_normal_initializer())
        #    weights= tf.get_variable("weights", [28 * 28 * io_sizes[i], io_sizes[i+1]], initializer=tf.random_normal_initializer())
            #h.append(tf.reshape(h[-1], [-1, int(28 * 28 / (2**(i+1))) * io_sizes[i]]))
        #    h_in=tf.reshape(h_in, [-1, 28 * 28 * io_sizes[i]])
        #    biases = tf.get_variable("biases", [io_sizes[i+1]], initializer=tf.constant_initializer(0.0))
        #    h_out=activation_fn(tf.matmul(h_in, weights) + biases)
       # else:
       #     weights= tf.get_variable("weights", [5, 5, io_sizes[i], io_sizes[i+1]], initializer=tf.random_normal_initializer())
       #     biases = tf.get_variable("biases", [io_sizes[i+1]], initializer=tf.constant_initializer(0.0))
       #     h_out=activation_fn(tf.nn.conv2d(h_in, weights, strides=[1, 1, 1, 1], padding='SAME') + biases)
       #     h_in=h_out
            #h.append(tf.nn.max_pool(h[-1], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME'))
            #h.append(max_pool_2x2(h[-1]))
  return h_out

def test_classification(model_function, learning_rate=0.1):
    # import data
    mnist = input_data.read_data_sets('./datasets/mnist/', one_hot=True)

    with tf.Graph().as_default() as g:
        # where are you going to allocate memory and perform computations
        with tf.device("/cpu:0"):
            
            # define model "input placeholders", i.e. variables that are
            # going to be substituted with input data on train/test time
            x_ = tf.placeholder(tf.float32, [None, 784])
            y_ = tf.placeholder(tf.float32, [None, 10])
            y_logits = model_function(x_)
            
            # naive implementation of loss:
            # > losses = y_ * tf.log(tf.nn.softmax(y_logits))
            # > tf.reduce_mean(-tf.reduce_sum(losses, 1))
            # can be numerically unstable.
            #
            # so here we use tf.nn.softmax_cross_entropy_with_logits on the raw
            # outputs of 'y', and then average across the batch.
            
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_logits)
            cross_entropy_loss = tf.reduce_mean(losses)
            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_loss)
            
            y_pred = tf.argmax(tf.nn.softmax(y_logits), dimension=1)
            correct_prediction = tf.equal(y_pred, tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with g.as_default(), tf.Session() as sess:
        # that is how we "execute" statements 
        # (return None, e.g. init() or train_op())
        # or compute parts of graph defined above (loss, output, etc.)
        # given certain input (x_, y_)
        sess.run(tf.global_variables_initializer())
        
        # train
        for iter_i in range(20001):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x_: batch_xs, y_: batch_ys})
            
            # test trained model
            if iter_i % 2000 == 0:
                tf_feed_dict = {x_: mnist.test.images, y_: mnist.test.labels}
                acc_value = sess.run(accuracy, feed_dict=tf_feed_dict)
                print('iteration %d\t accuracy: %.3f'%(iter_i, acc_value))
                
test_classification(lambda x: mlp(x, [64, 10], activation_fn=tf.nn.relu), learning_rate=0.1)

