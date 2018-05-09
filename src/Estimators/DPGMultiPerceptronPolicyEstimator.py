import tensorflow as tf
import tflearn
from Utils import normalize, denormalize

class DPGMultiPerceptronPolicyEstimator(object):
    """ Deterministic Policy Gradient Policy Function Approximator

    The estimator in use deploys a multi-layer perceptron

    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_rms, state_dim, action_dim, h_layer_shapes, learning_rate, action_bound, state_range, tau,
            minibatch_size, imitation_learning_rate=-1, imitation_minibatch_size=-1, scope="policy_estimator"):
        self._sess = sess
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._h_layer_shapes = h_layer_shapes
        self._action_bound = action_bound
        self._state_range = state_range
        self._learning_rate = learning_rate
        self._tau = tau
        self._minibatch_size = minibatch_size

        self._imitation_learning_rate = imitation_learning_rate if imitation_learning_rate != -1 else learning_rate
        self._imitation_minibatch_size = imitation_minibatch_size if imitation_minibatch_size != -1 else\
        minibatch_size

        self._state_rms = state_rms

        # Actor Network
        self._inputs, _, self._scaled_out = self._create_actor_network(scope)

        self._network_params = tf.trainable_variables()

        # Target Network
        self._target_inputs, _, self._target_scaled_out = self._create_actor_network(scope + "_target")

        self._target_network_params = tf.trainable_variables()[
            len(self._network_params):]

        # Make tau a placeholder so we can plug in different values within this class
        self._tau_var = tf.placeholder(tf.float32, [], name="tau")
        with tf.name_scope(scope + "_target/"):
            # Op for periodically updating target network with online network
            # weights
            self._update_target_network_params = \
                [self._target_network_params[i].assign(tf.multiply(self._network_params[i], self._tau_var) +
                                                      tf.multiply(self._target_network_params[i], 1. - self._tau_var))
                    for i in range(len(self._target_network_params))]

        with tf.name_scope(scope):
            self._actual_outputs = tf.placeholder(tf.float32, [None, self._action_dim], name="actual_outputs")
            # This gradient will be provided by the critic network
            self._action_gradient = tf.placeholder(tf.float32, [None, self._action_dim], name="action_gradient")

            # Combine the gradients here
            self._unnormalized_actor_gradients = tf.gradients(
                self._scaled_out, self._network_params, -self._action_gradient)
            self._actor_gradients = list(map(lambda x: tf.div(x, self._minibatch_size), self._unnormalized_actor_gradients))

            # Optimization Op
            self._optimize = tf.train.AdamOptimizer(self._learning_rate).\
                apply_gradients(zip(self._actor_gradients, self._network_params), name="optimize")

            # MSE loss for imitation learning
            self._imitation_loss = tflearn.mean_square(self._actual_outputs, self._scaled_out)
            self._imitation_optimize = tf.train.AdamOptimizer(self._imitation_learning_rate).\
                    minimize(self._imitation_loss, name="imitation_optimize")

        self._num_trainable_vars = len(
            self._network_params) + len(self._target_network_params)

            #graph = tf.get_default_graph()

            #self._inputs = graph.get_tensor_by_name(scope + "/inputs:0")
            #self._scaled_out = graph.get_tensor_by_name(scope + "/scaled_out:0")
            #self._optimize = graph.get_tensor_by_name(scope + "/optimize:0")
            #self._action_gradient = graph.get_tensor_by_name(scope + "/action_gradient:0")
            #self._target_inputs = graph.get_tensor_by_name(scope + "_target/inputs:0")
            #self._target_scaled_out = graph.get_tensor_by_name(scope + "_target/scaled_out:0")

            #self._network_params = tf.trainable_variables()[num_actor_vars:]
            #self._target_network_params = tf.trainable_variables()[(len(self._network_params) + num_actor_vars):]
            #self._update_target_network_params = []
            #for i in range(len(self._target_network_params)):
            #    self._update_target_network_params[i] = graph.get_tensor_by_name(
            #            scope+"_target/update_target_network_params:{}".format(i))

    def _create_actor_network(self, scope):
        with tf.name_scope(scope):
            inputs = tflearn.input_data(shape=(None, self._state_dim), name="inputs")
            # Normalize input(states)
            normalized_inputs = tf.clip_by_value(normalize(inputs, self._state_rms), self._state_range[0],
                    self._state_range[1])
            net = normalized_inputs
            for i in range(len(self._h_layer_shapes)):
                net = tflearn.fully_connected(net, self._h_layer_shapes[i])
                # Add l2 regularizer
                tflearn.helpers.regularizer.add_weights_regularizer(net.W, "L2")
                net = tflearn.layers.normalization.batch_normalization(net)
                net = tflearn.activations.relu(net)
            # Final layer weights are init to Uniform[-3e-3, 3e-3]
            w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
            out = tflearn.fully_connected(
                net, self._action_dim, activation='tanh', weights_init=w_init, name="out")
            # Add l2 regularizer
            tflearn.helpers.regularizer.add_weights_regularizer(out.W, "L2")
            # Scale output to -action_bound to action_bound
            scaled_out = tf.multiply(out, self._action_bound, name="scaled_out")
        return inputs, out, scaled_out

    def updateImitation(self, inputs, actual_outputs):
        _, loss = self._sess.run([self._imitation_optimize, self._imitation_loss], feed_dict={
            self._inputs: inputs,
            self._actual_outputs: actual_outputs
            })
        # Also keep nn with target nn in sync
        self._sess.run(self._update_target_network_params, feed_dict={
            self._tau_var: 1.0
            })
        return loss

    def update(self, inputs, a_gradient):
        self._sess.run(self._optimize, feed_dict={
            self._inputs: inputs,
            self._action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self._sess.run(self._scaled_out, feed_dict={
            self._inputs: inputs
        })

    def predict_target(self, inputs):
        return self._sess.run(self._target_scaled_out, feed_dict={
            self._target_inputs: inputs
        })

    def update_target_network(self):
        self._sess.run(self._update_target_network_params, feed_dict={
            self._tau_var: self._tau
            })

    def get_num_trainable_vars(self):
        return self._num_trainable_vars
