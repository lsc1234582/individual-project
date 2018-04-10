import numpy as np
import tensorflow as tf

from Utils import *

class SPGMultiPerceptronValueEstimator():
    """ Stochastic Policy Gradient Value Function Approximator.

    The estimator in use deploys a multi-layer perceptron
    """

    def __init__(self, sess, layer_shapes, learning_rate=0.1, initial_weight_stddev=1.0, initial_bias_stddev=1.0, scope="value_estimator"):
        """ Initializer

        Args:
            layer_shapes (list): The shape of the multiperceptron arrange as
                                 [input_dim, hidden_l1, ..., hidden_ln, output_dim]
            learning_rate (float32): Learning rate
            initial_weight_stddev (float32): initial_weight_stddev
            initial_bias_stddev (float32): initial_bias_stddev
        """
        self._sess = sess
        with tf.variable_scope(scope):
            self._state = tf.placeholder(tf.float32, (None, layer_shapes[0]), "state")
            self._target = tf.placeholder(dtype=tf.float32, name="target")
            self._is_training = tf.placeholder(tf.bool, name="is_training")

            hidden_l = self._state

            for i in range(1, len(layer_shapes) - 1):
                hidden_l = tf.contrib.layers.fully_connected(
                            inputs=hidden_l,
                            num_outputs=layer_shapes[i],
                            activation_fn=None,
                            weights_regularizer=tf.contrib.layers.l2_regularizer,
                            biases_regularizer=tf.contrib.layers.l2_regularizer,
                            weights_initializer=tf.initializers.truncated_normal(stddev=initial_weight_stddev),
                            biases_initializer=tf.initializers.truncated_normal(stddev=initial_bias_stddev),
                            scope="hidden_l{}".format(i))

                hidden_l = tf.contrib.layers.batch_norm(
                            inputs=hidden_l,
                            is_training = self._is_training,
                            scope="hidden_l{}_bn".format(i))

                hidden_l = tf.nn.relu(hidden_l)

            self._value_estimate = tf.contrib.layers.fully_connected(
                                    inputs=hidden_l,
                                    num_outputs=layer_shapes[-1],
                                    activation_fn=None,
                                    weights_regularizer=tf.contrib.layers.l2_regularizer,
                                    biases_regularizer=tf.contrib.layers.l2_regularizer,
                                    weights_initializer=tf.initializers.truncated_normal(stddev=initial_weight_stddev),
                                    biases_initializer=tf.initializers.truncated_normal(stddev=initial_bias_stddev),
                                    scope="value_estimate")

            self._loss = tf.reduce_mean(tf.squared_difference(self._value_estimate, self._target), name="loss")

            self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self._train_op = self._optimizer.minimize(
                self._loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state):
        """ Estimate the value given current state

        Args:
            state (array(shape=(None, state dimension), dtype=float32)): Current state

        Return:
            value_estimate (array(shape=(None, 1), dtype=float32)): Value estimate
        """
        return self._sess.run(self._value_estimate, { self._state: state, self._is_training: False })

    def update(self, state, target):
        """ A single update step of the network with target (actual value)

        Args:
            state (array(shape=(None, state dimension), dtype=float32)): Current state
            target (array(shape=(None, 1), dtype=float32)): The target(value) of each state

        Return:
            loss (float32): Training loss

        """
        feed_dict = { self._state: state, self._target: target, self._is_training: True }
        _, loss = self._sess.run([self._train_op, self._loss], feed_dict)
        return loss
