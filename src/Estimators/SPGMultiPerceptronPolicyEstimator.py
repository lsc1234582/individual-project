import numpy as np
import tensorflow as tf

from Utils import *

class SPGMultiPerceptronPolicyEstimator():
    """ Stochastic Policy Gradient Policy Function Approximator

    The estimator in use deploys a multi-layer perceptron
    """

    def __init__(self, sess, layer_shapes, learning_rate=0.001, initial_weight_stddev=1.0,
            initial_bias_stddev=1.0, action_lo=-1.0, action_hi=1.0, scope="policy_estimator"):
        """ Initializer

        Args:
            layer_shapes (list): The shape of the multiperceptron arrange as
                                 [input_dim, hidden_l1, ..., hidden_ln, output_dim]
            learning_rate (float32): Learning rate
            initial_weight_stddev (float32): initial_weight_stddev
            initial_bias_stddev (float32): initial_bias_stddev
            action_lo (float32): action minimum
            action_hi (float32): action maximum
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

            self._mu = tf.contrib.layers.fully_connected(
                inputs=hidden_l,
                num_outputs=layer_shapes[-1],
                activation_fn=None,
                weights_regularizer=tf.contrib.layers.l2_regularizer,
                biases_regularizer=tf.contrib.layers.l2_regularizer,
                weights_initializer=tf.initializers.truncated_normal(stddev=initial_weight_stddev),
                biases_initializer=tf.initializers.truncated_normal(stddev=initial_bias_stddev),
                scope="mu")

            self._sigma = tf.contrib.layers.fully_connected(
                inputs=hidden_l,
                num_outputs=1,
                activation_fn=None,
                weights_regularizer=tf.contrib.layers.l2_regularizer,
                biases_regularizer=tf.contrib.layers.l2_regularizer,
                weights_initializer=tf.initializers.truncated_normal(stddev=initial_weight_stddev),
                biases_initializer=tf.initializers.truncated_normal(stddev=initial_bias_stddev))

            self._sigma = tf.add(tf.nn.softplus(self._sigma), 1e-5, name="sigma")
            self._normal_dist = tf.contrib.distributions.Normal(self._mu, self._sigma)
            self._action = self._normal_dist.sample()
            self._action = tf.clip_by_value(self._action, action_lo, action_hi, name="action")

            # Loss and train op
            self._loss = tf.multiply(-self._target, self._normal_dist.log_prob(self._action), name="loss_vanilla")
            # Add cross entropy cost to encourage exploration
            self._loss = tf.subtract(self._loss, 1e-1 * self._normal_dist.entropy(), name="loss_with_entropy")
            self._loss = tf.reduce_mean(self._loss, name="loss_with_entropy")

            self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self._train_op = self._optimizer.minimize(
                self._loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state):
        """ Estimate the best action given current state

        Args:
            state (array(shape=(None, state dimension), dtype=float32)): Current state

        Return:
            action (array(shape=(None, action dimension), dtype=float32)): Best action estimate
        """
        return self._sess.run(self._action, { self._state: state, self._is_training: False })

    def update(self, state, target, action):
        """ A single update step of the network with actual action taken and target (advantage)

        Args:
            state (array(shape=(None, state dimension), dtype=float32)): Current state
            target (array(shape=(None, 1), dtype=float32)): The target(advantage/return) of each state, action pair
            action (array(shape=(None, action dimension), dtype=float32)): Action sample

        Return:
            loss (float32): Training loss

        """
        feed_dict = { self._state: state, self._target: target, self._action: action, self._is_training: True }
        _, loss = self._sess.run([self._train_op, self._loss], feed_dict)
        return loss
